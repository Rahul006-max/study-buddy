import streamlit as st
import base64
import os
from sentence_transformers import SentenceTransformer
from pdf_reader import extract_text_from_pdf
# Import the new Groq Summarizer
from summarizer import GroqSummarizer
# Keep the local concept extractor for now (or swap for AI version if you updated keywords.py)
from keywords import ConceptExtractor
from qa import SemanticQA

# --- Configuration ---
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

# --- API Key Handling ---
# We check for an environment variable first (for Cloud), but allow manual entry in Sidebar for local testing
api_key = os.environ.get("GROQ_API_KEY")

with st.sidebar:
    st.header("Settings")
    # If no key found in env, let user input one temporarily
    input_key = st.text_input("Groq API Key (Optional if set in Env)", type="password")
    
    if input_key:
        api_key = input_key
    
    if api_key:
        st.success("Groq API Key loaded!", icon="✅")
    else:
        st.warning("Please enter API Key to use AI features.", icon="⚠️")

# --- Session State Initialization ---
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'keywords' not in st.session_state:
    st.session_state.keywords = []
if 'history' not in st.session_state:
    st.session_state.history = [] 
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'filename' not in st.session_state:
    st.session_state.filename = ""

# --- Helper Functions ---

def display_pdf(file):
    """
    Displays a PDF uploaded via Streamlit using an HTML iframe.
    """
    # Read file as bytes
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    
    # Embed using HTML Iframe
    pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="1000px" 
                type="application/pdf">
        </iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_embedding_model():
    """Loads the local all-MiniLM-L6-v2 model for Semantic Search (QA)."""
    with st.spinner("Loading Local Search Model..."):
        return SentenceTransformer('all-MiniLM-L6-v2')

# --- UI Layout ---
st.title("📚 AI Study Assistant")
st.markdown("Upload a PDF to generate a summary, extract concepts, and ask questions.")

# File Upload Section
uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])

if uploaded_file is not None:
    current_filename = uploaded_file.name
    
    # Logic: Check if we need to process a new file
    if st.session_state.filename != current_filename:
        
        with st.spinner("Processing PDF..."):
            # 1. Extract Text
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if not raw_text:
                st.error("Could not extract text from this PDF. It might be an image-based scan.")
                st.stop()

            # 2. Initialize Models
            # We keep the local SentenceTransformer for fast QA (Retrieval)
            local_model = load_embedding_model()
            
            # 3. Summarize (Using Groq API)
            summary = "Error: API Key missing."
            if api_key:
                summarizer = GroqSummarizer(api_key=api_key)
                summary = summarizer.summarize(raw_text)
            else:
                summary = "Please enter the Groq API Key in the sidebar to generate a summary."

            # 4. Keywords
            # You can switch this to AI later, for now we use the local extractor
            extractor = ConceptExtractor()
            keywords = extractor.extract_keywords(summary)

            # 5. Setup QA (Using Local Semantic Search)
            qa_system = SemanticQA(local_model)
            qa_system.prepare_context(raw_text)

            # 6. Update Session State
            st.session_state.text = raw_text
            st.session_state.summary = summary
            st.session_state.keywords = keywords
            st.session_state.qa_system = qa_system
            st.session_state.filename = current_filename
            st.session_state.history = [] 
            
            st.success("Document processed successfully!")
            st.rerun() # Rerun to display results immediately

# --- PDF Viewer ---
if uploaded_file is not None:
    with st.expander("📄 View Original PDF"):
        uploaded_file.seek(0) 
        display_pdf(uploaded_file)

# --- Display Results (Only if text exists) ---
if st.session_state.text:
    st.divider()
    
    # Column Layout for Summary and Keywords
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Document Summary")
        st.write(st.session_state.summary)
    
    with col2:
        st.subheader("🔑 Key Concepts")
        st.write(", ".join([f"`{kw}`" for kw in st.session_state.keywords]))

    st.divider()

    # --- Q&A Section ---
    st.subheader("💬 Ask Questions")
    
    # Container for chat history
    chat_container = st.container()
    
    # Input area
    with st.form("q_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about the document:", placeholder="e.g., What is the main conclusion?")
        submit_button = st.form_submit_button("Send")
    
    if submit_button and user_input:
        if st.session_state.qa_system:
            answer = st.session_state.qa_system.ask(user_input)
            # Append to history
            st.session_state.history.append((user_input, answer))
            st.rerun() 

    # Display Chat History
    with chat_container:
        if not st.session_state.history:
            st.info("No questions asked yet.")
        else:
            for i, (q, a) in enumerate(reversed(st.session_state.history)):
                # User Message
                st.markdown(f"**You:** {q}")
                # AI Message
                st.markdown(f"**Assistant:** {a}")
                st.divider() if i < len(st.session_state.history) - 1 else None

else:
    st.info("Please upload a PDF file to begin.")