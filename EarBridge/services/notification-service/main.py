import os
import json
import logging
import redis
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("notification-service")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_CHANNEL = "audio_alerts"

def main():
    """
    Subscribes to Redis channel and processes audio event alerts.
    """
    logger.info("Starting Notification Service...")
    
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe(REDIS_CHANNEL)
        logger.info(f"Subscribed to channel: {REDIS_CHANNEL}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        return

    logger.info("Waiting for messages...")
    
    for message in pubsub.listen():
        if message["type"] == "message":
            try:
                data = json.loads(message["data"])
                event_type = data.get("event_type", "Unknown")
                confidence = data.get("confidence", 0.0)
                description = data.get("description", "No description provided.")
                is_critical = data.get("is_critical", False)
                
                # Format the alert
                logger.warning("=" * 60)
                logger.warning(f"[ALERT TRIGGERED] Event: {event_type}")
                logger.warning(f"Confidence: {confidence:.2f} | Critical: {is_critical}")
                logger.warning(f"Reason: {description}")
                logger.warning("=" * 60)
                
            except json.JSONDecodeError:
                logger.error(f"Received malformed JSON message: {message['data']}")
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
        
        # Slight sleep to prevent CPU spiking in tight loop if listen() behaves unexpectedly
        time.sleep(0.01)

if __name__ == "__main__":
    main()
