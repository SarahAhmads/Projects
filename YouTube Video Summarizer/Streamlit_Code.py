import streamlit as st
import requests
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥",
    layout="wide",
)

st.title("🎥 YouTube Video Summarizer")
st.markdown("Summarize any YouTube video using a remote AI model — no local download needed.")

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a full or short URL."""
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        video_id = parsed.path.strip("/")
        if video_id:
            return video_id
    qs = parse_qs(parsed.query)
    video_ids = qs.get("v")
    if not video_ids:
        raise ValueError(f"Could not find a video ID in: {url}")
    return video_ids[0]


def get_transcript(video_id: str) -> str:
    """Fetch the English transcript for a YouTube video."""
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=["en"])
    return "\n".join(snippet.text for snippet in fetched)


def call_summarize_api(
    base_url: str,
    text: str,
    max_length: int,
    min_length: int,
    timeout: int = 300,
) -> list[str]:
    """
    POST the transcript to the ngrok /summarize endpoint.
    Returns a list of summary strings (one per chunk).
    """
    endpoint = base_url.rstrip("/") + "/summarize"
    payload = {
        "text": text,
        "max_length": max_length,
        "min_length": min_length,
    }
    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    if "error" in data:
        raise RuntimeError(data["error"])

    return data["summaries"]


def check_server_health(base_url: str) -> bool:
    """Ping /health and return True if the server is up."""
    try:
        r = requests.get(base_url.rstrip("/") + "/health", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar — server configuration ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Server Configuration")
    st.markdown(
        "Run **`kaggle_backend_server.ipynb`** on Kaggle or Colab, "
        "then paste the ngrok URL it prints below."
    )

    ngrok_url = st.text_input(
        "ngrok URL",
        placeholder="https://xxxx-xx-xx-xxx-xx.ngrok-free.app",
        help="The public URL printed by the backend notebook.",
    )

    if ngrok_url:
        if st.button("Check server health"):
            with st.spinner("Pinging server..."):
                alive = check_server_health(ngrok_url)
            if alive:
                st.success("Server is up and ready!")
            else:
                st.error("Could not reach the server. Check the URL and make sure the notebook is running.")

    st.divider()
    st.header("ℹ️ About")
    st.markdown(
        """
        This app uses:
        - **YouTube Transcript API** — fetches the video transcript locally
        - **BART Large CNN** — runs *remotely* on Kaggle/Colab GPU via ngrok
        - No heavy model download on your machine

        ### How to use
        1. Run `kaggle_backend_server.ipynb` on Kaggle (GPU enabled)
        2. Copy the ngrok URL it prints into the field above
        3. Click **Check server health** to confirm it's reachable
        4. Paste a YouTube URL below and click **Generate Summary**

        ### Requirements
        - Video must have English captions
        - Keep the Kaggle notebook running while you use this app
        """
    )

    st.header("💡 Tips")
    st.markdown(
        """
        - Longer videos take more time — the backend processes them in 300-word chunks
        - Adjust max/min length to get more or less detail
        - Use the transcript expander to verify the source text
        """
    )

# ── Main form ─────────────────────────────────────────────────────────────────
with st.form("video_form"):
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Supports both youtube.com and youtu.be formats.",
    )

    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length (tokens)", 50, 200, 120)
    with col2:
        min_length = st.slider("Minimum summary length (tokens)", 20, 100, 40)

    submit = st.form_submit_button("🚀 Generate Summary", type="primary")

# ── Processing ────────────────────────────────────────────────────────────────
if submit:
    if not ngrok_url:
        st.error("Please enter the ngrok URL in the sidebar first.")
        st.stop()

    if not video_url:
        st.error("Please enter a YouTube video URL.")
        st.stop()

    try:
        # 1. Extract video ID
        with st.spinner("Extracting video ID..."):
            video_id = extract_video_id(video_url)
        st.success(f"Video ID: `{video_id}`")

        # 2. Embed the video
        st.video(video_url)

        # 3. Fetch transcript
        with st.spinner("Fetching transcript from YouTube..."):
            transcript = get_transcript(video_id)
        word_count = len(transcript.split())
        st.success(f"Transcript fetched — {word_count:,} words")

        with st.expander("📄 View Original Transcript"):
            st.text_area("Transcript", transcript, height=300, label_visibility="collapsed")

        # 4. Call the remote model
        st.info(f"Sending transcript to remote model at `{ngrok_url}` …")
        progress = st.progress(0, text="Waiting for server response...")

        summaries = call_summarize_api(ngrok_url, transcript, max_length, min_length)

        progress.progress(100, text="Done!")
        progress.empty()

        # 5. Display results
        st.subheader("📝 Summary by Section")
        for idx, summary in enumerate(summaries, 1):
            st.markdown(f"**Part {idx}:**")
            st.write(summary)
            if idx < len(summaries):
                st.divider()

        st.subheader("🎯 Complete Summary")
        combined = " ".join(summaries)
        st.write(combined)

        st.download_button(
            label="⬇️ Download Summary",
            data=combined,
            file_name=f"summary_{video_id}.txt",
            mime="text/plain",
        )

    except ValueError as e:
        st.error(f"Invalid URL: {e}")
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the remote server. "
            "Make sure the Kaggle notebook is running and the ngrok URL is correct."
        )
    except requests.exceptions.Timeout:
        st.error(
            "The request timed out. The video may be very long — try again or use a shorter video."
        )
    except requests.exceptions.HTTPError as e:
        st.error(f"Server returned an error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.info("Check that the video has English captions and the server is reachable.")
