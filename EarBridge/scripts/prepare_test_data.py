"""
prepare_test_data.py
--------------------
Prepares test audio samples for the NIM-driven Audio Event Detection system.

1. Extracts sireNNet.zip into data/test_samples/sirennet/
2. Downloads 5 random AudioSet segments for 'Glass breaking' or 'Fire alarm'
   using yt-dlp, saving them as .wav into data/test_samples/audioset/

Requirements: yt-dlp, pandas
"""

import os
import json
import random
import zipfile
import subprocess
import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("prepare-test-data")

# ──────────────────────────────────────────────
# Paths (relative to project root c:\Users\mod30\audio)
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

SIRENNET_ZIP = DATA_DIR / "sireNNet-Emergency Vehicle Siren Classification Dataset For Urban Applications" / "sireNNet.zip"
AUDIOSET_CSV = DATA_DIR / "audioset" / "balanced_train_segments.csv"
ONTOLOGY_JSON = DATA_DIR / "audioset" / "ontology-master" / "ontology-master" / "ontology.json"

OUTPUT_SIRENNET = DATA_DIR / "test_samples" / "sirennet"
OUTPUT_AUDIOSET = DATA_DIR / "test_samples" / "audioset"

NUM_SAMPLES = 5
TARGET_CLASSES = ["Glass, glass breaking", "Fire alarm"]


# ──────────────────────────────────────────────
# Step 1: Extract sireNNet
# ──────────────────────────────────────────────
def extract_sirennet() -> None:
    """Extracts sireNNet.zip into the test_samples directory."""
    if not SIRENNET_ZIP.exists():
        logger.error(f"sireNNet.zip not found at {SIRENNET_ZIP}")
        return

    OUTPUT_SIRENNET.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if any(OUTPUT_SIRENNET.iterdir()):
        logger.info(f"sireNNet already extracted at {OUTPUT_SIRENNET}. Skipping.")
        return

    logger.info(f"Extracting {SIRENNET_ZIP} -> {OUTPUT_SIRENNET} ...")
    try:
        with zipfile.ZipFile(SIRENNET_ZIP, "r") as zf:
            zf.extractall(OUTPUT_SIRENNET)
        logger.info(f"✓ sireNNet extracted to {OUTPUT_SIRENNET}")
    except zipfile.BadZipFile:
        logger.error("sireNNet.zip is corrupted or not a valid zip file.")


# ──────────────────────────────────────────────
# Step 2: Download AudioSet samples
# ──────────────────────────────────────────────
def load_ontology() -> dict:
    """Loads AudioSet ontology.json and returns a mapping of mid -> display name."""
    if not ONTOLOGY_JSON.exists():
        logger.error(f"ontology.json not found at {ONTOLOGY_JSON}")
        return {}

    with open(ONTOLOGY_JSON, "r", encoding="utf-8") as f:
        ontology = json.load(f)

    mid_to_name = {entry["id"]: entry["name"] for entry in ontology}
    return mid_to_name


def find_target_mids(mid_to_name: dict) -> list[str]:
    """Finds the MID codes for target classes."""
    target_mids = []
    for mid, name in mid_to_name.items():
        for target in TARGET_CLASSES:
            if target.lower() in name.lower():
                target_mids.append(mid)
                logger.info(f"  Found target class: '{name}' -> {mid}")
    return target_mids


def parse_audioset_csv() -> pd.DataFrame:
    """Parses the AudioSet balanced_train_segments.csv (skipping comment headers)."""
    if not AUDIOSET_CSV.exists():
        logger.error(f"AudioSet CSV not found at {AUDIOSET_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(
        AUDIOSET_CSV,
        sep=", ",
        engine="python",
        skiprows=3,
        names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
        skipinitialspace=True,
    )
    # Strip whitespace and quotes from YTID
    df["YTID"] = df["YTID"].str.strip()
    return df


def download_segment(ytid: str, start: float, end: float, output_path: Path) -> bool:
    """
    Downloads a specific segment of a YouTube video as .wav using yt-dlp.
    Returns True on success, False on failure.
    """
    url = f"https://www.youtube.com/watch?v={ytid}"
    output_template = str(output_path.with_suffix(""))  # yt-dlp adds extension

    cmd = [
        "yt-dlp",
        "--no-warnings",
        "--quiet",
        "-x",                          # extract audio
        "--audio-format", "wav",        # convert to wav
        "--download-sections", f"*{start}-{end}",
        "-o", output_template + ".%(ext)s",
        url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # yt-dlp may produce the file with .wav extension
        expected = output_path.with_suffix(".wav")
        if expected.exists():
            logger.info(f"  ✓ Downloaded {ytid} ({start}-{end}s) -> {expected.name}")
            return True
        else:
            logger.warning(f"  ✗ File not created for {ytid}. stderr: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning(f"  ✗ Timeout downloading {ytid}")
        return False
    except FileNotFoundError:
        logger.error("yt-dlp is not installed. Install with: pip install yt-dlp")
        return False
    except Exception as e:
        logger.warning(f"  ✗ Error downloading {ytid}: {e}")
        return False


def download_audioset_samples() -> None:
    """Downloads NUM_SAMPLES AudioSet segments for target classes."""
    logger.info("Loading AudioSet ontology...")
    mid_to_name = load_ontology()
    if not mid_to_name:
        return

    logger.info(f"Searching for target classes: {TARGET_CLASSES}")
    target_mids = find_target_mids(mid_to_name)
    if not target_mids:
        logger.error("No matching MIDs found in ontology for target classes.")
        return

    logger.info("Parsing AudioSet CSV...")
    df = parse_audioset_csv()
    if df.empty:
        return

    # Filter rows containing any target MID
    def has_target(labels_str: str) -> bool:
        labels = labels_str.strip('"').split(",")
        return any(mid in labels for mid in target_mids)

    filtered = df[df["positive_labels"].apply(has_target)]
    logger.info(f"Found {len(filtered)} segments matching target classes.")

    if filtered.empty:
        logger.warning("No matching segments found in the CSV.")
        return

    # Pick random samples
    samples = filtered.sample(n=min(NUM_SAMPLES, len(filtered)), random_state=42)
    OUTPUT_AUDIOSET.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {len(samples)} AudioSet samples...")
    success_count = 0
    for _, row in samples.iterrows():
        ytid = row["YTID"]
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        filename = f"{ytid}_{int(start)}_{int(end)}"
        output_path = OUTPUT_AUDIOSET / filename

        # Skip if already downloaded
        if output_path.with_suffix(".wav").exists():
            logger.info(f"  → Already exists: {filename}.wav")
            success_count += 1
            continue

        if download_segment(ytid, start, end, output_path):
            success_count += 1

    logger.info(f"AudioSet download complete: {success_count}/{len(samples)} succeeded.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 60)
    logger.info("PREPARING TEST DATA FOR AUDIO EVENT DETECTION SYSTEM")
    logger.info("=" * 60)

    logger.info("\n[Step 1/2] Extracting sireNNet dataset...")
    extract_sirennet()

    logger.info("\n[Step 2/2] Downloading AudioSet samples...")
    download_audioset_samples()

    logger.info("\n" + "=" * 60)
    logger.info("TEST DATA PREPARATION COMPLETE")
    logger.info(f"  sireNNet samples : {OUTPUT_SIRENNET}")
    logger.info(f"  AudioSet samples : {OUTPUT_AUDIOSET}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
