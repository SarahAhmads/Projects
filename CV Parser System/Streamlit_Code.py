import streamlit as st
from pypdf import PdfReader
import requests
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Parser",
    page_icon="📋",
    layout="wide"
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📋 AI-Powered CV/Resume Parser")
st.markdown("Automatically extract structured information from CVs and resumes using a remote AI backend (Kaggle GPU)")

# ── Session state defaults ────────────────────────────────────────────────────
if "parsed_data" not in st.session_state:
    st.session_state.parsed_data = None
if "backend_url" not in st.session_state:
    st.session_state.backend_url = ""
if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = False


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalise_url(url: str) -> str:
    """Strip trailing slash so we can append paths cleanly."""
    return url.rstrip("/")


def check_health(base_url: str) -> tuple[bool, str]:
    """Ping /health and return (ok, message)."""
    try:
        r = requests.get(f"{base_url}/health", timeout=10)
        if r.status_code == 200:
            data = r.json()
            return True, data.get("model", "unknown model")
        return False, f"Server returned HTTP {r.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Could not connect. Is the ngrok tunnel running?"
    except requests.exceptions.Timeout:
        return False, "Request timed out. Try again."
    except Exception as exc:
        return False, str(exc)


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from an uploaded PDF file object."""
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)


def call_parse_api(base_url: str, pdf_file) -> dict:
    """
    POST the PDF to the remote /parse endpoint.
    Returns the parsed JSON dict or raises on error.
    """
    pdf_file.seek(0)
    files = {"file": (pdf_file.name, pdf_file, "application/pdf")}
    response = requests.post(
        f"{base_url}/parse",
        files=files,
        timeout=120       # model inference can take ~1 min
    )
    response.raise_for_status()
    return response.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # --- Backend connection ---
    st.header("🔌 Backend Connection")
    st.markdown(
        "Run **`kaggle_backend_server.ipynb`** on Kaggle, then paste the "
        "ngrok URL printed in the last cell here."
    )

    url_input = st.text_input(
        "ngrok URL",
        value=st.session_state.backend_url,
        placeholder="https://xxxx-xx-xx-xxx-xx.ngrok-free.app",
    )

    col_connect, col_status = st.columns([1, 2])
    with col_connect:
        connect_btn = st.button("Connect", use_container_width=True)

    if connect_btn and url_input.strip():
        with st.spinner("Pinging backend…"):
            clean_url = normalise_url(url_input.strip())
            ok, msg = check_health(clean_url)
            st.session_state.backend_url = clean_url
            st.session_state.backend_ok = ok
            if ok:
                st.success(f"Connected  ✅\n{msg}")
            else:
                st.error(f"Failed ❌\n{msg}")
    elif connect_btn:
        st.warning("Please enter a URL first.")

    # Show persistent status badge
    with col_status:
        if st.session_state.backend_ok:
            st.markdown(
                "<div style='background:#1a7a4a;color:white;border-radius:6px;"
                "padding:6px 10px;margin-top:4px;font-size:13px'>● Online</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='background:#7a1a1a;color:white;border-radius:6px;"
                "padding:6px 10px;margin-top:4px;font-size:13px'>● Offline</div>",
                unsafe_allow_html=True
            )

    st.divider()

    # --- File upload ---
    st.header("📁 Upload CV / Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
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
    - ✅ Projects (if present)

    ### How it works
    1. Start **`kaggle_backend_server.ipynb`** on Kaggle (free T4 GPU)
    2. Paste the printed **ngrok URL** above and click **Connect**
    3. Upload a CV PDF and click **Parse CV**
    """)


# ── Main area ─────────────────────────────────────────────────────────────────
if not uploaded_file and not st.session_state.backend_ok:
    # Landing page
    st.info("👈 Connect to the Kaggle backend, then upload a CV/Resume PDF to get started")

    st.subheader("📖 How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 1️⃣ Kaggle")
        st.write("Run the backend notebook on a free Kaggle GPU")
    with col2:
        st.markdown("### 2️⃣ Connect")
        st.write("Paste the ngrok URL in the sidebar and click Connect")
    with col3:
        st.markdown("### 3️⃣ Upload")
        st.write("Upload your CV/Resume PDF")
    with col4:
        st.markdown("### 4️⃣ Review")
        st.write("View and download the structured results")

    st.divider()
    st.subheader("✨ Key Features")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("""
        #### 🎯 Accurate Extraction
        - State-of-the-art Mistral-Nemo model
        - Handles various CV formats
        - Extracts key information reliably

        #### 📊 Structured Output
        - Clean, organised data
        - JSON format for easy integration
        - Ready for databases or APIs
        """)
    with fc2:
        st.markdown("""
        #### ☁️ No Local GPU Required
        - Model runs on Kaggle's free T4 GPU
        - Lightweight Streamlit client
        - Works on any machine

        #### 💾 Export Options
        - Download as JSON
        - Download as plain text
        - Easy to save and share
        """)

elif not st.session_state.backend_ok:
    st.warning("⚠️ Backend not connected. Enter the ngrok URL in the sidebar and click **Connect**.")

elif not uploaded_file:
    st.info("👈 Upload a CV/Resume PDF in the sidebar to get started")

else:
    # ── Parse button ──────────────────────────────────────────────────────────
    if st.button("🔍 Parse CV", type="primary", use_container_width=False):
        try:
            # Show extracted text preview
            with st.spinner("Extracting text from PDF…"):
                cv_text = extract_text_from_pdf(uploaded_file)
                word_count = len(cv_text.split())
            st.success(f"✅ Extracted {word_count} words from PDF")

            with st.expander("📄 View Extracted Text"):
                st.text_area("CV Text", cv_text, height=250)

            # Send to Kaggle backend
            with st.spinner("🤖 Sending to AI backend… (this may take ~60 s)"):
                parsed_data = call_parse_api(
                    st.session_state.backend_url,
                    uploaded_file
                )
                st.session_state.parsed_data = parsed_data

            st.success("✅ CV parsed successfully!")

        except requests.exceptions.HTTPError as exc:
            st.error(f"❌ Backend error: {exc.response.status_code} – {exc.response.text}")
        except requests.exceptions.Timeout:
            st.error("❌ Request timed out. The model may still be loading — try again in 30 s.")
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.parsed_data:
        data = st.session_state.parsed_data

        # Surface any backend-side error field
        if "error" in data:
            st.error(f"Backend reported an error: {data['error']}")
        else:
            st.divider()
            st.subheader("📊 Parsed Information")

            # Personal info
            st.markdown("### 👤 Personal Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Full Name**")
                st.info(data.get("FullName", "Not found"))
            with col2:
                st.markdown("**Email**")
                st.info(data.get("Email", "Not found"))
            with col3:
                st.markdown("**Phone**")
                st.info(data.get("Phone", "Not found"))

            st.divider()

            # Education
            st.markdown("### 🎓 Education")
            education = data.get("Education", "Not found")
            if isinstance(education, list):
                for i, edu in enumerate(education, 1):
                    st.markdown(f"**{i}.** {edu}")
            else:
                st.write(education)

            st.divider()

            # Skills
            st.markdown("### 💡 Skills")
            skills = data.get("Skills", [])
            if isinstance(skills, list) and skills:
                skills_html = " ".join([
                    f'<span style="background-color:#1f77b4;color:white;'
                    f'padding:5px 10px;margin:3px;border-radius:5px;'
                    f'display:inline-block;">{s}</span>'
                    for s in skills
                ])
                st.markdown(skills_html, unsafe_allow_html=True)
            else:
                st.write(skills or "Not found")

            st.divider()

            # Experience
            st.markdown("### 💼 Work Experience")
            experience = data.get("Experience", [])
            if isinstance(experience, list) and experience:
                for i, exp in enumerate(experience, 1):
                    st.markdown(f"**{i}.** {exp}")
            else:
                st.write(experience or "Not found")

            st.divider()

            # Projects
            st.markdown("### 🚀 Projects")
            projects = data.get("Projects", [])
            if isinstance(projects, list) and projects:
                for i, proj in enumerate(projects, 1):
                    st.markdown(f"**{i}.** {proj}")
            else:
                st.write("Not found")

            st.divider()

            # Downloads
            st.subheader("💾 Download Parsed Data")
            dl1, dl2 = st.columns(2)

            with dl1:
                st.download_button(
                    label="📥 Download as JSON",
                    data=json.dumps(data, indent=2),
                    file_name="parsed_cv.json",
                    mime="application/json",
                    type="primary",
                    use_container_width=True,
                )

            with dl2:
                skills_str = (
                    ", ".join(data.get("Skills", []))
                    if isinstance(data.get("Skills"), list)
                    else str(data.get("Skills", "N/A"))
                )
                exp_str = (
                    "\n".join(data.get("Experience", []))
                    if isinstance(data.get("Experience"), list)
                    else str(data.get("Experience", "N/A"))
                )
                proj_str = (
                    "\n".join(data.get("Projects", []))
                    if isinstance(data.get("Projects"), list)
                    else str(data.get("Projects", "N/A"))
                )
                text_output = f"""CV PARSED INFORMATION
=====================

PERSONAL INFORMATION
--------------------
Name:  {data.get('FullName', 'N/A')}
Email: {data.get('Email', 'N/A')}
Phone: {data.get('Phone', 'N/A')}

EDUCATION
---------
{data.get('Education', 'N/A')}

SKILLS
------
{skills_str}

EXPERIENCE
----------
{exp_str}

PROJECTS
--------
{proj_str}
"""
                st.download_button(
                    label="📥 Download as Text",
                    data=text_output,
                    file_name="parsed_cv.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with st.expander("🔍 View Raw JSON"):
                st.json(data)
