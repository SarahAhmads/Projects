import streamlit as st
from pypdf import PdfReader
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import torch
import re
import json

# Page configuration
st.set_page_config(
    page_title="CV Parser",
    page_icon="📋",
    layout="wide"
)

# Title and description
st.title("📋 AI-Powered CV/Resume Parser")
st.markdown("Automatically extract structured information from CVs and resumes using AI")

# Initialize model with caching
@st.cache_resource
def load_model():
    """Load the LLM model"""
    with st.spinner("Loading AI model... This may take a few minutes on first run."):
        import os
        hf_token = os.getenv("HF_TOKEN")
        
        model_name = "mistralai/Mistral-Nemo-Instruct-2407"
        
        if hf_token:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto",
                token=hf_token
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
    return tokenizer, model

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_text(prompt, tokenizer, model, max_length=1500):
    """Generate text using LLM"""
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
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def extract_json_block(text):
    """Extract JSON block from model response"""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return f"```json\n{matches[-1]}\n```"
    return text

def parse_cv(cv_text, tokenizer, model):
    """Parse CV and extract structured information"""
    
    # Define the expected JSON structure
    cv_extraction_template = """
You are a smart assistant that extracts information from CVs/resumes.

Extract the following information from the CV and respond ONLY with a valid JSON object (no markdown, no code blocks, just pure JSON):

{{
    "FullName": "candidate's full name",
    "Email": "candidate's email address",
    "Phone": "candidate's phone number",
    "Education": "education history with degree, institution, and year",
    "Skills": ["skill1", "skill2", "skill3"],
    "Experience": ["experience1 with role, company, duration", "experience2"]
}}

CV Text:
{cv_text}

Remember: Respond with ONLY the JSON object, nothing else.
"""
    
    prompt = cv_extraction_template.format(cv_text=cv_text)
    
    # Generate response
    response = generate_text(prompt, tokenizer, model, max_length=1500)
    
    # Extract JSON from response
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed_data = json.loads(json_str)
        else:
            # If no JSON found, try to parse the whole response
            parsed_data = json.loads(response)
    except json.JSONDecodeError:
        # If parsing fails, create a fallback structure
        st.error("Failed to parse JSON. Showing raw response.")
        st.text(response)
        parsed_data = {
            "FullName": "Could not extract",
            "Email": "Could not extract",
            "Phone": "Could not extract",
            "Education": "Could not extract",
            "Skills": [],
            "Experience": []
        }
    
    return parsed_data

# Initialize session state
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None

# Sidebar
with st.sidebar:
    st.header("📁 Upload CV/Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a CV or resume in PDF format"
    )
    
    st.divider()
    
    st.header("ℹ️ About")
    st.markdown("""
    This parser uses AI to extract:
    - ✅ Personal Information
    - ✅ Contact Details
    - ✅ Education History
    - ✅ Work Experience
    - ✅ Skills & Competencies
    
    ### Supported Formats:
    - PDF resumes/CVs
    - Multi-page documents
    - Various CV templates
    """)
    
    st.divider()
    
    st.header("🔧 Features")
    st.markdown("""
    - Automatic information extraction
    - Structured JSON output
    - Easy-to-read display
    - Downloadable results
    - Support for various CV formats
    """)

# Main content
if not uploaded_file:
    st.info("👈 Please upload a CV/Resume PDF to get started")
    
    # Display example
    st.subheader("📖 How it works")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Upload")
        st.write("Upload your CV/Resume in PDF format")
    
    with col2:
        st.markdown("### 2️⃣ Process")
        st.write("AI analyzes and extracts information")
    
    with col3:
        st.markdown("### 3️⃣ Review")
        st.write("View structured, organized results")
    
    st.divider()
    
    st.subheader("✨ Key Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        #### 🎯 Accurate Extraction
        - Uses state-of-the-art AI models
        - Handles various CV formats
        - Extracts key information reliably
        """)
        
        st.markdown("""
        #### 📊 Structured Output
        - Clean, organized data
        - JSON format for easy integration
        - Ready for databases or APIs
        """)
    
    with feature_col2:
        st.markdown("""
        #### ⚡ Fast Processing
        - Quick analysis
        - Real-time results
        - Efficient parsing
        """)
        
        st.markdown("""
        #### 💾 Export Options
        - Download parsed data
        - JSON format
        - Easy to save and share
        """)

else:
    # Process button
    if st.button("🔍 Parse CV", type="primary"):
        try:
            # Extract text from PDF
            with st.spinner("Extracting text from PDF..."):
                cv_text = extract_text_from_pdf(uploaded_file)
                word_count = len(cv_text.split())
                st.success(f"✅ Extracted {word_count} words from CV")
            
            # Display extracted text in expander
            with st.expander("📄 View Extracted Text"):
                st.text_area("CV Text", cv_text, height=300)
            
            # Load model and parse
            tokenizer, model = load_model()
            
            with st.spinner("🤖 AI is analyzing the CV... This may take a minute."):
                parsed_data = parse_cv(cv_text, tokenizer, model)
                st.session_state.parsed_data = parsed_data
            
            st.success("✅ CV parsed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check the CV format and try again.")
    
    # Display parsed data
    if st.session_state.parsed_data:
        st.divider()
        st.subheader("📊 Parsed Information")
        
        data = st.session_state.parsed_data
        
        # Personal Information Section
        st.markdown("### 👤 Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Full Name:**")
            st.info(data.get("FullName", "Not found"))
            
            st.markdown("**Email:**")
            st.info(data.get("Email", "Not found"))
        
        with col2:
            st.markdown("**Phone:**")
            st.info(data.get("Phone", "Not found"))
        
        st.divider()
        
        # Education Section
        st.markdown("### 🎓 Education")
        education = data.get("Education", "Not found")
        if isinstance(education, list):
            for idx, edu in enumerate(education, 1):
                st.markdown(f"**{idx}.** {edu}")
        else:
            st.write(education)
        
        st.divider()
        
        # Skills Section
        st.markdown("### 💡 Skills")
        skills = data.get("Skills", [])
        if isinstance(skills, list) and skills:
            # Display skills as tags
            skills_html = " ".join([f'<span style="background-color: #1f77b4; color: white; padding: 5px 10px; margin: 2px; border-radius: 5px; display: inline-block;">{skill}</span>' for skill in skills])
            st.markdown(skills_html, unsafe_allow_html=True)
        else:
            st.write(skills if skills else "Not found")
        
        st.divider()
        
        # Experience Section
        st.markdown("### 💼 Work Experience")
        experience = data.get("Experience", [])
        if isinstance(experience, list) and experience:
            for idx, exp in enumerate(experience, 1):
                st.markdown(f"**{idx}.** {exp}")
        else:
            st.write(experience if experience else "Not found")
        
        st.divider()
        
        # Download section
        st.subheader("💾 Download Parsed Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON download
            json_str = json.dumps(data, indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_str,
                file_name="parsed_cv.json",
                mime="application/json",
                type="primary"
            )
        
        with col2:
            # Text format download
            text_output = f"""
CV PARSED INFORMATION
=====================

PERSONAL INFORMATION
--------------------
Name: {data.get('FullName', 'N/A')}
Email: {data.get('Email', 'N/A')}
Phone: {data.get('Phone', 'N/A')}

EDUCATION
---------
{data.get('Education', 'N/A')}

SKILLS
------
{', '.join(data.get('Skills', [])) if isinstance(data.get('Skills', []), list) else data.get('Skills', 'N/A')}

EXPERIENCE
----------
{chr(10).join(data.get('Experience', [])) if isinstance(data.get('Experience', []), list) else data.get('Experience', 'N/A')}
"""
            st.download_button(
                label="📥 Download as Text",
                data=text_output,
                file_name="parsed_cv.txt",
                mime="text/plain"
            )
        
        # Display raw JSON in expander
        with st.expander("🔍 View Raw JSON"):
            st.json(data)
