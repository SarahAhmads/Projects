import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Page configuration
st.set_page_config(
    page_title="YouTube Video Summarizer",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 YouTube Video Summarizer")
st.markdown("Automatically generate summaries from YouTube videos using AI")

# Initialize summarizer with caching
@st.cache_resource
def load_summarizer():
    """Load the summarization model"""
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    
    return tokenizer, model

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a URL"""
    parsed = urlparse(url)
    
    # Handle youtu.be short links
    if 'youtu.be' in parsed.netloc:
        # Extract video ID from path (after the /)
        video_id = parsed.path.strip('/')
        if video_id:
            return video_id
    
    # Handle standard youtube.com links
    qs = parse_qs(parsed.query)
    video_ids = qs.get('v')
    
    if not video_ids:
        raise ValueError(f"No video id found in URL: {url}")
    
    return video_ids[0]

def get_transcript(video_id: str) -> str:
    """Fetch transcript from YouTube video"""
    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=['en'])
    text = "\n".join(snippet.text for snippet in fetched)
    return text

def chunk_text(text: str, max_words: int = 300) -> list:
    """Split text into chunks for processing"""
    text_parts = []
    words = text.split()
    for i in range(0, len(words), max_words):
        part = " ".join(words[i:i + max_words])
        text_parts.append(part)
    return text_parts

def summarize_text(text: str, tokenizer, model, max_length: int = 120, min_length: int = 40):
    """Generate summary from text"""
    chunks = chunk_text(text)
    summaries = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, part in enumerate(chunks):
        status_text.text(f"Processing chunk {idx + 1} of {len(chunks)}...")
        
        # Tokenize input
        inputs = tokenizer(part, max_length=1024, truncation=True, return_tensors="pt")
        
        # Generate summary
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)
        
        progress_bar.progress((idx + 1) / len(chunks))
    
    status_text.empty()
    progress_bar.empty()
    
    return summaries

# Main interface
with st.form("video_form"):
    video_url = st.text_input(
        "YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=... or https://youtu.be/...",
        help="Enter the full YouTube video URL (supports both youtube.com and youtu.be formats)"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        max_length = st.slider("Maximum summary length", 50, 200, 120)
    with col2:
        min_length = st.slider("Minimum summary length", 20, 100, 40)
    
    submit_button = st.form_submit_button("Generate Summary", type="primary")

if submit_button and video_url:
    try:
        with st.spinner("Loading AI model..."):
            tokenizer, model = load_summarizer()
        
        with st.spinner("Extracting video ID..."):
            video_id = extract_video_id(video_url)
            st.success(f"Video ID: {video_id}")
        
        # Display video
        st.video(video_url)
        
        with st.spinner("Fetching transcript..."):
            transcript = get_transcript(video_id)
            st.success(f"Transcript fetched! ({len(transcript.split())} words)")
        
        # Show original transcript in expander
        with st.expander("View Original Transcript"):
            st.text_area("Transcript", transcript, height=300)
        
        with st.spinner("Generating summary..."):
            summaries = summarize_text(transcript, tokenizer, model, max_length, min_length)
        
        # Display summaries
        st.subheader("📝 Summary")
        for idx, summary in enumerate(summaries, 1):
            st.markdown(f"**Part {idx}:**")
            st.write(summary)
            st.divider()
        
        # Combined summary
        st.subheader("🎯 Complete Summary")
        combined_summary = " ".join(summaries)
        st.write(combined_summary)
        
        # Download button
        st.download_button(
            label="Download Summary",
            data=combined_summary,
            file_name=f"summary_{video_id}.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the URL and try again. Make sure the video has English captions available. Both youtube.com and youtu.be links are supported.")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This application uses:
    - **YouTube Transcript API** to fetch video transcripts
    - **BART Large CNN** model for summarization
    - Processes videos in chunks for optimal performance
    
    ### How to use:
    1. Paste a YouTube video URL
    2. Adjust summary length preferences
    3. Click "Generate Summary"
    4. View and download the summary
    
    ### Note:
    - Video must have English captions
    - Processing time depends on video length
    - Longer videos will be split into multiple parts
    """)
    
    st.header("🎯 Tips")
    st.markdown("""
    - For best results, use videos with clear speech
    - Adjust max/min length for different detail levels
    - Check the original transcript if summary seems off
    """)
