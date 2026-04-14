"""
test_pipeline.py
----------------
Automated E2E test for the NIM-driven Audio Event Detection system.

This script selects audio files from the sireNNet dataset, sends them
via HTTP POST to the local inference API, and saves the results into
a JSON file for later review.

Usage:
    python scripts/test_pipeline.py

Requires:
    pip install requests
"""

import os            # For interacting with the operating system (e.g., paths)
import sys           # For system-specific parameters and functions (e.g., exiting)
import json          # For parsing and saving JSON data
import random        # For randomly selecting sample files
import logging       # For emitting log messages during execution
import argparse      # For parsing command-line arguments
from pathlib import Path  # For object-oriented file system paths
from datetime import datetime # For timestamping the output file

# Attempt to import the 'requests' library, which is needed for HTTP calls.
try:
    import requests
except ImportError:
    # If not installed, print an error message and exit the script with a status code of 1.
    print("ERROR: 'requests' library not found. Install with: pip install requests")
    sys.exit(1)

# Configure the logging mechanism to output informational messages with timestamps.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Create a specific logger instance for this script.
logger = logging.getLogger("test-pipeline")

# -------------------------------------------------------------------------
# Configuration Constants
# -------------------------------------------------------------------------
# Determine the root directory of the project (two levels up from this script).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The URL where the FastAPI inference service is listening.
API_URL = "http://localhost:8000/detect"

# How many random audio files to test per category.
SAMPLES_PER_CLASS = 2

# The specific categories (subfolders) within the sireNNet dataset to look for.
CATEGORIES = ["ambulance", "firetruck", "police", "traffic"]

# Determine the absolute path to where the sireNNet dataset was extracted.
SIRENNET_DIR = (
    PROJECT_ROOT
    / "data"
    / "sireNNet-Emergency Vehicle Siren Classification Dataset For Urban Applications"
    / "sireNNet"
)

# Output directory for the JSON results
OUTPUT_DIR = PROJECT_ROOT / "data" / "test_results"


def find_sirennet_dir() -> Path:
    """
    Dynamically locate the extracted sireNNet directory.
    Returns:
        The Path object pointing to the root of the sireNNet dataset.
    """
    # If the default predicted path exists, return it immediately.
    if SIRENNET_DIR.exists():
        return SIRENNET_DIR

    # Fallback mechanism: Search dynamically under the 'data/' directory.
    data_dir = PROJECT_ROOT / "data"
    
    # Iterate recursively through all directories named 'sireNNet'.
    for candidate in data_dir.rglob("sireNNet"):
        # Check if it's a directory and if it contains all 4 category folders.
        if candidate.is_dir() and all((candidate / c).is_dir() for c in CATEGORIES):
            return candidate

    # If the directory cannot be found, log an error and exit.
    logger.error("Could not find extracted sireNNet directory under data/.")
    sys.exit(1)


def select_samples(base_dir: Path) -> list[tuple[str, Path]]:
    """
    Selects a defined number of random .wav files from each category folder.
    
    Args:
        base_dir: The root Path of the sireNNet dataset.
        
    Returns:
        A list of tuples containing (category_name, file_path).
    """
    selected = []  # Initialize an empty list to hold the selected file paths.
    
    # Iterate over each category (ambulance, firetruck, etc.).
    for category in CATEGORIES:
        # Construct the path to the specific category subfolder.
        cat_dir = base_dir / category
        
        # Check if the folder exists.
        if not cat_dir.exists():
            logger.warning(f"Category folder not found: {cat_dir}")
            continue

        # Find all files ending in '.wav' in this category folder.
        wav_files = list(cat_dir.glob("*.wav"))
        
        # If the folder is empty or has no .wav files, warn and move on.
        if not wav_files:
            logger.warning(f"No .wav files found in {cat_dir}")
            continue

        # Determine how many files to pick (either SAMPLES_PER_CLASS or the max available).
        count = min(SAMPLES_PER_CLASS, len(wav_files))
        
        # Randomly select the specified number of files without replacement.
        picks = random.sample(wav_files, count)
        
        # Add the selected tuple (category string, file Path) to the list.
        for f in picks:
            selected.append((category, f))

    return selected


def send_to_api(filepath: Path) -> dict | None:
    """
    Sends an audio file to the local API /detect endpoint.
    
    Args:
        filepath: The Path to the .wav file.
        
    Returns:
        The parsed JSON dictionary response from the server, or None if it failed.
    """
    try:
        # Open the file in binary read mode.
        with open(filepath, "rb") as f:
            # Construct the multipart/form-data payload expected by FastAPI UploadFile.
            files = {"file": (filepath.name, f, "audio/wav")}
            
            # Perform the HTTP POST request to the inference API.
            response = requests.post(API_URL, files=files, timeout=120)

        # HTTP 200 implies the request was successfully processed.
        if response.status_code == 200:
            return response.json()  # Return the parsed JSON body.
        else:
            # Log an error if the server returned a 4xx or 5xx HTTP status code.
            logger.error(f"API returned {response.status_code}: {response.text[:200]}")
            return None

    # Handle the case where the server is offline or unreachable.
    except requests.exceptions.ConnectionError:
        logger.error(
            f"CONNECTION ERROR: Could not reach {API_URL}. "
            "Make sure the inference service is running (e.g., docker-compose up)."
        )
        return None
    # Handle the case where the request took more than 120 seconds.
    except requests.exceptions.Timeout:
        logger.error(f"TIMEOUT: Request took too long for {filepath.name}")
        return None
    # Catch any other unforeseen exceptions preventing the HTTP call.
    except Exception as e:
        logger.error(f"Unexpected error making request: {e}")
        return None


def main() -> None:
    """
    Main function to execute the automated testing pipeline.
    """
    # Setup argparse to allow the user to optionally specify a single file to test.
    parser = argparse.ArgumentParser(description="Test NIM-driven Audio Event Detection API")
    parser.add_argument("--file", type=str, default=None, help="Path to a specific .wav file to test")
    parser.add_argument("--expected", type=str, default="unknown", help="Expected class category (used in the report)")
    args = parser.parse_args()

    # Print a header to mark the beginning of the test execution.
    print("\n" + "=" * 65)
    print("   AUDIO EVENT DETECTION - E2E TEST PIPELINE")
    print("=" * 65 + "\n")

    # Check if the user passed a specific file via the command line
    if args.file:
        specific_file = Path(args.file)
        if not specific_file.exists():
            logger.error(f"The provided file does not exist: {specific_file}")
            sys.exit(1)
            
        logger.info(f"Testing single specified file: {specific_file.name}")
        # Build the samples list with just this one file
        samples = [(args.expected, specific_file)]
    else:
        # Step 1: Locate the dataset folder dynamically.
        base_dir = find_sirennet_dir()
        logger.info(f"Using sireNNet dataset at: {base_dir}")

        # Step 2: Randomly select the .wav files to use for testing.
        samples = select_samples(base_dir)
        
        # If no files were found across all categories, abort.
        if not samples:
            logger.error("No samples selected. Exiting.")
            sys.exit(1)

        logger.info(f"Selected {len(samples)} samples across {len(CATEGORIES)} categories.\n")

    # Initialize a list to hold the test results for the JSON output report.
    report_data = []

    # Step 3: Loop through each selected file, query the API, and log the result.
    for expected_class, filepath in samples:
        logger.info(f"Sending {filepath.name} (Expected true class: {expected_class})...")
        
        # Send the file via HTTP POST.
        result_json = send_to_api(filepath)
        
        # Print basic console feedback for the user in real-time.
        if result_json:
            print(f"  -> Detected: {result_json.get('event_type')} | Critical: {result_json.get('is_critical')} ")
        else:
            print(f"  -> Failed to get response.")

        # Create a dictionary summarizing the request and response for the output file.
        test_record = {
            "file_name": filepath.name,
            "expected_category": expected_class,
            "api_response": result_json
        }
        
        # Append the record to our full report dataset.
        report_data.append(test_record)

    # Print a footer to signify the completion of API requests.
    print("\n" + "=" * 65)
    print("   TEST PIPELINE RUN COMPLETE")
    print("=" * 65)

    # Step 4: Save the aggregated results to a structured JSON file.
    
    # Ensure the output directory (data/test_results/) actually exists.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate a unique filename using the current date and time.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"test_report_{timestamp}.json"
    
    try:
        # Open the new file in text-write mode.
        with open(output_file, "w", encoding="utf-8") as f:
            # Dump the Python list as a beautifully indented JSON structure.
            json.dump(report_data, f, indent=4)
        print(f"\nSaved detailed JSON report to:\n{output_file}")
    except Exception as e:
        # Catch file permission errors or directory issues when saving.
        logger.error(f"Failed to save JSON report: {e}")


# Run the main() function if this script is executed directly via python.
if __name__ == "__main__":
    main()
