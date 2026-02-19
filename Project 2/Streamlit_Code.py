import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# Title and description
st.title("📄 PDF Q&A Assistant with RAG")
st.markdown("Ask questions about your PDF documents using AI-powered Retrieval Augmented Generation")

# Initialize models with caching
@st.cache_resource
def load_models():
    """Load all required models"""
    with st.spinner("Loading models... This may take a few minutes on first run."):
        # Set HuggingFace token for faster downloads (optional)
        import os
        hf_token = os.getenv("HF_TOKEN")
        
        # Load embedding model
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load LLM
        model_name = "mistralai/Mistral-Nemo-Instruct-2407"
        
        # Use token if available
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto",
                token=hf_token
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            llm_model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        
    return embedding_model, tokenizer, llm_model

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
    return chunks

def create_embeddings(chunks, model):
    """Create embeddings for text chunks"""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings):
    """Build FAISS index for similarity search"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_relevant_chunks(query, model, index, chunks, k=3):
    """Search for most relevant chunks"""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]], distances[0]

def generate_answer(question, context, tokenizer, model, max_length=300):
    """Generate answer using LLM"""
    prompt = f"Answer the following question based on the provided context:\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the answer part (after "Answer:")
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return answer

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=['pdf'],
        help="Upload a PDF document to ask questions about"
    )
    
    st.divider()
    
    st.header("⚙️ Settings")
    chunk_size = st.slider("Chunk size (words)", 100, 1000, 500, 50)
    chunk_overlap = st.slider("Chunk overlap (words)", 10, 200, 50, 10)
    num_results = st.slider("Number of relevant chunks", 1, 5, 3)
    
    st.divider()
    
    if uploaded_file and st.button("Process PDF", type="primary"):
        with st.spinner("Processing PDF..."):
            # Extract text
            text = extract_text_from_pdf(uploaded_file)
            st.success(f"Extracted {len(text.split())} words")
            
            # Chunk text
            st.session_state.chunks = chunk_text(text, chunk_size, chunk_overlap)
            st.success(f"Created {len(st.session_state.chunks)} chunks")
            
            # Load models
            embedding_model, tokenizer, llm_model = load_models()
            st.session_state.embedding_model = embedding_model
            st.session_state.tokenizer = tokenizer
            st.session_state.llm_model = llm_model
            
            # Create embeddings and index
            with st.spinner("Creating embeddings..."):
                embeddings = create_embeddings(st.session_state.chunks, embedding_model)
                st.session_state.index = build_faiss_index(embeddings)
            
            st.session_state.processed = True
            st.success("✅ PDF processed successfully!")
    
    if st.session_state.processed:
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system:
    - Extracts text from PDFs
    - Splits into semantic chunks
    - Creates vector embeddings
    - Finds relevant context
    - Generates AI answers
    """)

# Main content area
if not st.session_state.processed:
    st.info("👈 Please upload a PDF document and click 'Process PDF' to get started")
    
    # Display example usage
    st.subheader("📖 How it works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload your PDF document")
    
    with col2:
        st.markdown("### 2️⃣ Process")
        st.write("AI analyzes and indexes the content")
    
    with col3:
        st.markdown("### 3️⃣ Ask")
        st.write("Ask questions and get accurate answers")

else:
    # Display chat interface
    st.subheader("💬 Chat with your document")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "chunks" in message:
                with st.expander("View relevant context"):
                    for idx, chunk in enumerate(message["chunks"], 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.write(chunk)
                        st.divider()
    
    # Question input
    question = st.chat_input("Ask a question about your document...")
    
    if question:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        
        with st.chat_message("user"):
            st.write(question)
        
        # Process question
        with st.chat_message("assistant"):
            with st.spinner("Searching for relevant information..."):
                relevant_chunks, distances = search_relevant_chunks(
                    question,
                    st.session_state.embedding_model,
                    st.session_state.index,
                    st.session_state.chunks,
                    k=num_results
                )
            
            # Combine chunks for context
            context = "\n\n".join(relevant_chunks)
            
            with st.spinner("Generating answer..."):
                answer = generate_answer(
                    question,
                    context,
                    st.session_state.tokenizer,
                    st.session_state.llm_model
                )
            
            st.write(answer)
            
            # Show relevant chunks
            with st.expander("View relevant context"):
                for idx, (chunk, distance) in enumerate(zip(relevant_chunks, distances), 1):
                    st.markdown(f"**Chunk {idx}** (Relevance score: {1/(1+distance):.2f})")
                    st.write(chunk)
                    st.divider()
            
            # Add assistant message to chat
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "chunks": relevant_chunks
            })

# Sample questions section
if st.session_state.processed:
    with st.expander("💡 Need inspiration? Try these sample questions"):
        st.markdown("""
        - What is the main topic of this document?
        - Can you summarize the key points?
        - What are the important dates mentioned?
        - Who are the people mentioned in this document?
        - What conclusions are drawn in this document?
        """)
