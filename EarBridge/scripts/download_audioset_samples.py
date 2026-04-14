"""
download_audioset_samples.py
----------------------------
Parses the Google AudioSet ontology and balanced_train_segments.csv to extract
specific critical audio events for End-to-End API testing.

Target Classes: "Glass", "Shatter", "Fire alarm", "Screaming"

Downloads 3 random segments (start to end time) per class using yt-dlp
and saves them as: data/test_samples/audioset/<class_name>_<YTID>.wav
"""

import os
import sys
import json
import random
import logging
import subprocess
from pathlib import Path

import pandas as pd
from pydub import AudioSegment
from static_ffmpeg import add_paths

# Inject portable ffmpeg/ffprobe into the system PATH
add_paths()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("audioset-downloader")

# -------------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ONTOLOGY_JSON = DATA_DIR / "audioset" / "ontology-master" / "ontology-master" / "ontology.json"
AUDIOSET_CSV = DATA_DIR / "audioset" / "balanced_train_segments.csv"
OUTPUT_DIR = DATA_DIR / "test_samples" / "audioset"

TARGET_CLASSES = ["Glass", "Shatter", "Fire alarm", "Screaming"]
SAMPLES_PER_CLASS = 3


def load_ontology() -> dict:
    """Loads AudioSet ontology and returns a mapping of MID -> Display Name."""
    if not ONTOLOGY_JSON.exists():
        logger.error(f"Ontology JSON not found: {ONTOLOGY_JSON}")
        return {}
        
    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)
        
    return {entry["id"]: entry["name"] for entry in ontology}


def find_target_mids(mid_to_name: dict) -> dict[str, str]:
    """Finds MIDs for the target classes. Returns {mid: class_name_slug}."""
    target_map = {}
    for mid, name in mid_to_name.items():
        for target in TARGET_CLASSES:
            if target.lower() in name.lower():
                # Create a safe filename slug (e.g. "Fire alarm" -> "fire_alarm")
                slug = target.lower().replace(" ", "_")
                target_map[mid] = slug
                logger.info(f"Found target mapping: '{name}' (MID: {mid}) -> mapped to '{slug}'")
    return target_map


def parse_audioset_csv() -> pd.DataFrame:
    """Parses the AudioSet balanced_train_segments.csv (skipping comment headers)."""
    if not AUDIOSET_CSV.exists():
        logger.error(f"AudioSet CSV not found at {AUDIOSET_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(
        AUDIOSET_CSV,
        sep=", ",
        engine="python",
        skiprows=3,  # Ignore the first 3 lines starting with #
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
        skipinitialspace=True,
    )
    # Strip whitespace/quotes
    df["YTID"] = df["YTID"].str.strip(' "')
    df["positive_labels"] = df["positive_labels"].str.strip(' "')
    return df


def download_segment(ytid: str, start: float, end: float, output_path: Path) -> bool:
    """
    Downloads the full audio from YouTube using yt-dlp, 
    then slices exactly [start:end] using pydub, saving as .wav.
    """
    url = f"https://www.youtube.com/watch?v={ytid}"
    temp_output = str(output_path.with_name(f"temp_{ytid}")) 
    
    # 1. Download full audio natively
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--quiet", "--no-warnings",
        "-f", "bestaudio",
        "-o", temp_output + ".%(ext)s",
        "--", url,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        logger.error(f"  ✗ Timeout downloading full video {ytid}")
        return False
        
    # Find the downloaded temp file (extension might be .webm or .m4a)
    downloaded_files = list(output_path.parent.glob(f"temp_{ytid}.*"))
    if not downloaded_files:
        logger.warning(f"  ✗ Failed to download source audio for {ytid}")
        return False
        
    source_file = downloaded_files[0]
    final_file = output_path.with_suffix(".wav")

    # 2. Slice the audio using pydub
    try:
        logger.info(f"    Slicing {start}s - {end}s...")
        
        # Load the downloaded file
        audio = AudioSegment.from_file(str(source_file))
        
        # pydub works in milliseconds
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        
        sliced_audio = audio[start_ms:end_ms]
        
        # Export as wav
        sliced_audio.export(str(final_file), format="wav")
        logger.info(f"  ✓ Processed and saved -> {final_file.name}")
        
    except Exception as e:
        logger.error(f"  ✗ Error slicing audio for {ytid}: {e}")
        return False
    finally:
        # Cleanup the massive source file
        if source_file.exists():
            source_file.unlink()

    return final_file.exists()


def main():
    print("\n" + "=" * 60)
    print("  AUDIOSET DOWNLOADER: EXTRACTING TEST SAMPLES")
    print("=" * 60 + "\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Map target strings to MIDs
    mid_to_name = load_ontology()
    if not mid_to_name:
        return
        
    target_map = find_target_mids(mid_to_name)
    if not target_map:
        logger.error("Could not cross-reference target classes in ontology.")
        return

    # 2. Parse the CSV database
    logger.info("Parsing segments CSV metadata...")
    df = parse_audioset_csv()
    if df.empty:
        return

    # 3. Process each target class up to SAMPLES_PER_CLASS
    for mid, class_slug in target_map.items():
        logger.info(f"\n--- Processing class: {class_slug} (MID: {mid}) ---")
        
        # Filter all rows containing this specific MID
        class_rows = df[df["positive_labels"].apply(lambda labels: mid in labels.split(","))]
        
        if class_rows.empty:
            logger.warning(f"No segments found for MID {mid} in the structured dataset.")
            continue
            
        logger.info(f"Found {len(class_rows)} available segments. Picking {SAMPLES_PER_CLASS} randomly.")
        samples = class_rows.sample(n=min(SAMPLES_PER_CLASS, len(class_rows)), random_state=42)
        
        # 4. Attempt downloading the selected rows
        successful_downloads = 0
        for _, row in samples.iterrows():
            ytid = row["YTID"]
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            
            # Format: glass_breaking_jHsdf33A.wav
            filename = f"{class_slug}_{ytid}.wav"
            output_filepath = OUTPUT_DIR / filename
            
            # Skip if we already downloaded it previously
            if output_filepath.exists():
                logger.info(f"  → Skipping {filename} (Already exists)")
                successful_downloads += 1
                continue

            logger.info(f"Attempting to slice {ytid} [{start}s - {end}s] ...")
            if download_segment(ytid, start, end, output_filepath):
                successful_downloads += 1
                
        logger.info(f"Finished {class_slug}: successfully downloaded {successful_downloads}/{len(samples)}.")

    print("\n" + "=" * 60)
    print(f"  ✅ SCRIPT COMPLETE. Samples saved to {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
