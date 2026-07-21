import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
import requests

# ─── ngrok Tunnel Configuration ─────────────────────────────────────────────
# Paste the tunnel URL printed by the Kaggle notebook here.
# Example: "https://xxxx-xx-xx-xxx-xx.ngrok-free.app"
NGROK_TUNNEL_URL = "https://pseudoindependently-uninterpreted-fairy.ngrok-free.dev"  # <-- paste your tunnel URL here

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Q&A Assistant with RAG")
st.markdown(
    "Ask questions about your PDF documents using AI-powered "
    "Retrieval Augmented Generation — LLM runs remotely via ngrok, "
    "no local GPU needed."
)


# ─── Helpers: ngrok tunnel discovery ────────────────────────────────────────

@st.cache_data(ttl=60)
def get_ngrok_tunnel_url() -> str:
    """Return the configured tunnel URL."""
    return NGROK_TUNNEL_URL.rstrip("/")


# ─── Helpers: embedding (local, lightweight ~90 MB) ─────────────────────────

@st.cache_resource
def load_embedding_model():
    """Load the small sentence-transformer model locally."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ─── Helpers: PDF processing ─────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    step = max(size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + size])
        if chunk:
            chunks.append(chunk)
    return chunks


def create_embeddings(chunks: list[str], model) -> np.ndarray:
    return model.encode(chunks, convert_to_numpy=True)


def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def search_relevant_chunks(
    query: str, model, index, chunks: list[str], k: int = 3
) -> tuple[list[str], np.ndarray]:
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]], distances[0]


# ─── Helpers: remote LLM via ngrok ──────────────────────────────────────────

def generate_answer_remote(
    question: str,
    context: str,
    tunnel_url: str,
    max_new_tokens: int = 300,
) -> str:
    """
    Calls the remote Mistral-Nemo server exposed via ngrok.

    The notebook is expected to expose a /generate endpoint that accepts:
        POST /generate
        { "prompt": "...", "max_new_tokens": 300 }
    and returns:
        { "generated_text": "..." }

    If your notebook exposes a different schema, adjust the payload/key below.
    """
    if not tunnel_url:
        return "⚠️ Could not reach the remote model — no active ngrok tunnel found."

    prompt = (
        f"Answer the following question based only on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
    }

    try:
        resp = requests.post(
            f"{tunnel_url}/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        # Support common response keys
        raw = (
            data.get("generated_text")
            or data.get("text")
            or data.get("answer")
            or str(data)
        )

        # Strip the echoed prompt if the model returns it
        if "Answer:" in raw:
            raw = raw.split("Answer:")[-1].strip()

        return raw

    except requests.exceptions.Timeout:
        return "⚠️ Request timed out. The remote model may be busy — try again."
    except requests.exceptions.ConnectionError:
        return "⚠️ Could not connect to the remote model. Is the ngrok tunnel still active?"
    except Exception as e:
        return f"⚠️ Error calling remote model: {e}"


# ─── Session state init ──────────────────────────────────────────────────────

for key, default in {
    "processed": False,
    "chunks": [],
    "index": None,
    "chat_history": [],
    "embedding_model": None,
    "tunnel_url": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🌐 Remote Model Status")

    if st.button("🔄 Refresh Tunnel URL"):
        get_ngrok_tunnel_url.clear()

    tunnel_url = get_ngrok_tunnel_url()
    st.session_state.tunnel_url = tunnel_url

    if tunnel_url:
        st.success(f"Tunnel active:\n`{tunnel_url}`")
    else:
        st.warning("No tunnel URL set. Paste your ngrok URL into NGROK_TUNNEL_URL at the top of the script.")

    st.divider()

    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Upload a PDF document to ask questions about",
    )

    st.divider()

    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size (words)", 100, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 10, 200, 50, 10)
    num_results = st.slider("Relevant chunks to retrieve", 1, 5, 3)
    max_new_tokens = st.slider("Max tokens in answer", 100, 600, 300, 50)

    st.divider()

    if uploaded_file and st.button("Process PDF", type="primary"):
        with st.spinner("Extracting text from PDF…"):
            text = extract_text_from_pdf(uploaded_file)
            if not text.strip():
                st.error("Could not extract text — the PDF may be scanned/image-only.")
            else:
                st.success(f"Extracted {len(text.split()):,} words")

                st.session_state.chunks = chunk_text(text, chunk_size, chunk_overlap)
                st.success(f"Created {len(st.session_state.chunks)} chunks")

                with st.spinner("Loading embedding model (first time only)…"):
                    st.session_state.embedding_model = load_embedding_model()

                with st.spinner("Building vector index…"):
                    embeddings = create_embeddings(
                        st.session_state.chunks, st.session_state.embedding_model
                    )
                    st.session_state.index = build_faiss_index(embeddings)

                st.session_state.processed = True
                st.success("✅ PDF ready — start asking questions!")

    if st.session_state.processed:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.header("ℹ️ How it works")
    st.markdown(
        """
        1. **Upload** your PDF  
        2. Text is chunked & embedded locally (tiny model)  
        3. Your question retrieves the most relevant chunks via FAISS  
        4. The chunks + question are sent to **Mistral-Nemo** running remotely on Kaggle via ngrok  
        5. The answer is returned — no big model download needed locally  
        """
    )


# ─── Main area ───────────────────────────────────────────────────────────────

if not st.session_state.processed:
    st.info("👈 Upload a PDF and click **Process PDF** to get started.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload your PDF in the sidebar")
    with col2:
        st.markdown("### 2️⃣ Process")
        st.write("Click Process PDF — embedding happens locally")
    with col3:
        st.markdown("### 3️⃣ Ask")
        st.write("Type your question; Mistral answers via ngrok")

else:
    st.subheader("💬 Chat with your document")

    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and "chunks" in message:
                with st.expander("View retrieved context"):
                    for idx, chunk in enumerate(message["chunks"], 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.write(chunk)
                        st.divider()

    # Question input
    question = st.chat_input("Ask a question about your document…")

    if question:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant chunks…"):
                relevant_chunks, distances = search_relevant_chunks(
                    question,
                    st.session_state.embedding_model,
                    st.session_state.index,
                    st.session_state.chunks,
                    k=num_results,
                )

            context = "\n\n".join(relevant_chunks)

            with st.spinner("Sending to remote Mistral via ngrok…"):
                answer = generate_answer_remote(
                    question,
                    context,
                    st.session_state.tunnel_url,
                    max_new_tokens=max_new_tokens,
                )

            st.write(answer)

            with st.expander("View retrieved context"):
                for idx, (chunk, dist) in enumerate(
                    zip(relevant_chunks, distances), 1
                ):
                    relevance = 1 / (1 + float(dist))
                    st.markdown(f"**Chunk {idx}** — relevance: `{relevance:.2f}`")
                    st.write(chunk)
                    st.divider()

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "chunks": relevant_chunks,
                }
            )


