import os
import base64
import json
import logging
import redis
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("audio-inference-service")

app = FastAPI(title="NIM-Driven Audio Event Detection")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_CHANNEL = "audio_alerts"

try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# NVIDIA NIM Configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "google/gemma-3n-e4b-it"

client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY
)

@app.post("/detect", response_model=Dict[str, Any])
async def detect_audio_event(file: UploadFile = File(...)):
    """
    Receives an audio file, sends it to NVIDIA NIM for zero-shot classification,
    and publishes critical alerts to Redis.
    """
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="NVIDIA_API_KEY is not configured.")

    try:
        # Read file and encode to Base64
        audio_content = await file.read()
        audio_b64 = base64.b64encode(audio_content).decode('utf-8')
        
        # Exact prompt as specified
        system_prompt = (
            "Analyze this audio. Detect if it contains an ambulance siren, police siren, "
            "firetruck siren, breaking glass, or traffic noise. You MUST respond ONLY with "
            "a valid JSON object in this exact format: "
            "{\"event_type\": \"detected_sound\", \"confidence\": 0.95, \"is_critical\": true, \"description\": \"brief reason\"}. "
            "Do not include any markdown formatting, backticks, or extra text."
        )

        logger.info(f"Sending audio file '{file.filename}' to NVIDIA NIM...")

        # Call NVIDIA NIM API via OpenAI client
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": f"data:audio/wav;base64,{audio_b64}"}
                        }
                    ]
                }
            ],
            temperature=0.1,
            top_p=1.0,
            max_tokens=512
        )

        raw_response = response.choices[0].message.content.strip()
        logger.info(f"Raw NIM response: {raw_response}")

        # Robust JSON parsing
        try:
            # Modern LLMs sometimes include markdown even when told not to
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[-1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[-1].split("```")[0].strip()
            
            result_json = json.loads(raw_response)
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse NIM response as JSON: {je}")
            raise HTTPException(status_code=500, detail="Invalid response format from NIM API.")

        # Publish to Redis if critical
        if result_json.get("is_critical") is True and redis_client:
            redis_client.publish(REDIS_CHANNEL, json.dumps(result_json))
            logger.info(f"Published critical alert to Redis: {result_json['event_type']}")

        return result_json

    except Exception as e:
        logger.error(f"Error during audio detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "redis_connected": redis_client is not None if redis_client else False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
