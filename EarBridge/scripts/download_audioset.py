import os
import pandas as pd
import subprocess
import yt_dlp
from pathlib import Path

def download_segment(youtube_id: str, start_time: float, end_time: float, output_path: Path):
    """
    Downloads a specific segment of a YouTube video and converts it to wav.
    """
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path.with_suffix('')), # temporary template
        'quiet': True,
        'no_warnings': True,
        'external_downloader': 'ffmpeg',
        'external_downloader_args': [
            '-ss', str(start_time),
            '-to', str(end_time),
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # yt-dlp might append .wav because of postprocessor
        temp_file = output_path.with_suffix('.wav')
        if temp_file.exists():
            temp_file.rename(output_path)
        return True
    except Exception as e:
        print(f"Failed to download {youtube_id}: {e}")
        return False

def main():
    # Define classes of interest
    target_classes = ['Glass breaking', 'Fire alarm', 'Screaming']
    
    csv_path = Path("data/audioset/balanced_train_segments.csv")
    labels_path = Path("data/audioset/class_labels_indices.csv")
    output_dir = Path("data/audioset/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists() or not labels_path.exists():
        print("AudioSet CSV files not found in data/audioset/. Please download them first.")
        return

    # Load labels
    labels_df = pd.read_csv(labels_path)
    # Filter mid for target classes
    target_mids = labels_df[labels_df['display_name'].isin(target_classes)]['mid'].tolist()
    
    # Load segments
    # AudioSet CSV has header rows starting with #
    df = pd.read_csv(csv_path, sep=', ', engine='python', skiprows=3, names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
    
    # Filter segments containing any of the target labels
    def has_target_label(labels_str):
        labels = labels_str.strip('"').split(',')
        return any(mid in labels for mid in target_mids)

    filtered_df = df[df['positive_labels'].apply(has_target_label)]
    
    print(f"Found {len(filtered_df)} segments to download.")

    for idx, row in filtered_df.iterrows():
        ytid = row['YTID']
        start = row['start_seconds']
        end = row['end_seconds']
        filename = f"{ytid}_{int(start)}.wav"
        save_path = output_dir / filename
        
        if save_path.exists():
            continue
            
        print(f"Downloading {ytid} ({start}-{end})...")
        download_segment(ytid, start, end, save_path)

if __name__ == "__main__":
    main()
