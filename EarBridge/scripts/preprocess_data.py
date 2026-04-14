import os
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def process_file(self, file_path: Path):
        """
        Loads audio, resamples, and extracts Mel-Spectrogram.
        """
        try:
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            # Normalize length to 10 seconds (standard for AudioSet)
            target_len = self.sample_rate * 10
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = np.pad(y, (0, target_len - len(y)))

            # Extract Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            return mel_spec_db
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

def main():
    preprocessor = AudioPreprocessor()
    data_dir = Path("data")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = []
    
    # Process SirenNet (Assuming subdirectories are class names)
    sirennet_dir = data_dir / "sirennet"
    if sirennet_dir.exists():
        classes = [d.name for d in sirennet_dir.iterdir() if d.is_dir()]
        for cls in classes:
            files = list((sirennet_dir / cls).glob("*.wav"))
            print(f"Processing SirenNet class: {cls} ({len(files)} files)")
            for f in tqdm(files):
                spec = preprocessor.process_file(f)
                if spec is not None:
                    dataset.append({'spec': spec, 'label': cls})

    # Process AudioSet Raw (Downloader saves them here)
    audioset_raw = data_dir / "audioset/raw"
    if audioset_raw.exists():
        files = list(audioset_raw.glob("*.wav"))
        print(f"Processing AudioSet Raw ({len(files)} files)")
        for f in tqdm(files):
            spec = preprocessor.process_file(f)
            if spec is not None:
                # In a real scenario, we'd map filename back to labels from CSV
                # For simplicity, let's assume class name is in the filename or managed via a lookup
                dataset.append({'spec': spec, 'label': 'audioset_event'})

    if not dataset:
        print("No data found to process.")
        return

    # Save dataset
    df = pd.DataFrame(dataset)
    df.to_pickle(output_dir / "dataset.pkl")
    print(f"Preprocessing complete. Saved {len(dataset)} items to {output_dir / 'dataset.pkl'}")

if __name__ == "__main__":
    main()
