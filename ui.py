import streamlit as st
import base64
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pdf_reader import extract_text_from_pdf
from summarizer import SemanticSummarizer
from keywords import ConceptExtractor
from qa import SemanticQA
import time

# --- Load Environment Variables ---
load_dotenv()

# --- BRUTALIST THEME CSS ---
theme_css = """
<style>
    /* --- CSS VARIABLES --- */
    :root {
        --bg-color: #E8F5E8;
        --card-bg: #FFFFFF;
        --text-color: #000000;
        --accent-green: #4CAF50;
        --accent-purple: #8E24AA;
        --border-thick: #000000;
        --shadow-color: #000000;
    }

    /* --- FONTS --- */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;900&family=Arial+Black&display=swap');

    /* --- GLOBAL BODY & BACKGROUND --- */
    body {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Arial Black', 'Arial', sans-serif;
        font-weight: 900;
        margin: 0;
        padding: 20px;
    }

    ::selection {
        background-color: var(--accent-purple);
        color: #ffffff;
    }

    code, pre, input {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700;
    }

    /* --- TABS STYLING --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        border-bottom: 8px solid var(--text-color);
        padding-left: 0px;
        background-color: var(--accent-green);
        padding: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--card-bg);
        border: 4px solid var(--text-color);
        border-bottom: none;
        color: var(--text-color);
        font-family: 'Arial Black', sans-serif;
        font-weight: 900;
        text-transform: uppercase;
        font-size: 1.2rem;
        padding: 20px 40px;
        margin-right: 5px;
        box-shadow: 8px 8px 0px var(--text-color);
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-purple);
        color: #ffffff !important;
        box-shadow: 8px 8px 0px var(--accent-green);
        border: 4px solid var(--text-color);
        transform: translateY(4px);
    }

    /* --- TYPOGRAPHY --- */
    h1 {
        font-size: 4rem;
        font-weight: 900;
        color: var(--text-color) !important;
        background-color: var(--accent-green);
        display: inline-block;
        padding: 30px 50px;
        border: 8px solid var(--text-color);
        box-shadow: 12px 12px 0px var(--accent-purple);
        text-transform: uppercase;
        letter-spacing: -2px;
        margin: 20px 0;
        font-family: 'Arial Black', sans-serif;
    }
    
    h2 {
        font-size: 2rem;
        font-weight: 900;
        text-transform: uppercase;
        border: 4px solid var(--text-color);
        padding: 15px 25px;
        margin: 20px 0;
        color: var(--text-color) !important;
        display: inline-block;
        background: var(--accent-purple);
        color: #ffffff !important;
        box-shadow: 6px 6px 0px var(--accent-green);
        font-family: 'Arial Black', sans-serif;
    }
    
    h3 {
        color: var(--text-color) !important;
        font-weight: 900;
        font-family: 'Arial Black', sans-serif;
    }

    /* --- CONTENT CARDS --- */
    .summary-card {
        border: 6px solid var(--text-color);
        background-color: var(--card-bg);
        color: var(--text-color) !important;
        padding: 40px;
        border-left: 20px solid var(--accent-green);
        box-shadow: 12px 12px 0px var(--text-color);
        font-size: 1.2rem;
        line-height: 1.4;
        font-weight: 700;
        margin-bottom: 30px;
        width: 100%;
        font-family: 'Arial', sans-serif;
    }

    /* --- KEYWORDS (HIERARCHICAL GRID) --- */
    .kw-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 15px;
        margin-bottom: 50px;
        width: 100%;
    }

    /* Tier 1: CRITICAL (Top 2) */
    .kw-tier-1 {
        font-size: 1.5rem;
        font-weight: 900;
        background-color: var(--accent-purple);
        color: #ffffff !important;
        border: 6px solid var(--text-color);
        padding: 30px 50px;
        text-transform: uppercase;
        letter-spacing: 0px;
        box-shadow: 10px 10px 0px var(--accent-green);
        display: flex;
        justify-content: center;
        align-items: center;
        grid-column: span 2;
        font-family: 'Arial Black', sans-serif;
    }

    /* Tier 2: IMPORTANT (Next 3) */
    .kw-tier-2 {
        font-size: 1.2rem;
        font-weight: 900;
        background-color: var(--accent-green);
        color: var(--text-color) !important;
        border: 4px solid var(--text-color);
        padding: 20px 30px;
        text-transform: uppercase;
        letter-spacing: 0px;
        box-shadow: 6px 6px 0px var(--text-color);
        display: flex;
        align-items: center;
        font-family: 'Arial Black', sans-serif;
    }
    .kw-tier-2:hover {
        background-color: var(--accent-purple);
        color: #ffffff !important;
    }

    /* Tier 3: STANDARD (Rest) */
    .kw-standard {
        background-color: var(--card-bg);
        color: var(--text-color) !important;
        border: 3px solid var(--text-color);
        padding: 15px 25px;
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        font-size: 1rem;
        box-shadow: 4px 4px 0px var(--text-color);
        display: flex;
        align-items: center;
    }
    .kw-standard:hover {
        background-color: var(--accent-green);
        color: var(--text-color) !important;
    }

    /* --- CHAT BUBBLES --- */
    .chat-bubble {
        border: 4px solid var(--text-color);
        background-color: var(--card-bg);
        color: var(--text-color);
        padding: 30px;
        margin-bottom: 20px;
        font-family: 'Arial', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 6px 6px 0px var(--text-color);
    }
    .user-bubble {
        background-color: var(--card-bg);
        color: var(--text-color) !important;
        border-right: 20px solid var(--accent-green);
        text-align: right;
        margin-left: 20%;
    }
    .ai-bubble {
        background-color: var(--accent-purple);
        color: #ffffff !important;
        border-left: 20px solid var(--accent-green);
        box-shadow: 6px 6px 0px var(--accent-green);
        margin-right: 20%;
    }

    /* --- INPUTS & BUTTONS --- */
    .stTextInput > div > div > input {
        background-color: var(--card-bg) !important;
        color: var(--text-color) !important;
        border: 4px solid var(--text-color);
        border-radius: 0px;
        padding: 20px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 4px 4px 0px var(--text-color);
        font-family: 'JetBrains Mono', monospace;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 4px 4px 0px var(--accent-purple) !important;
    }
    
    .stButton > button {
        background-color: var(--accent-purple) !important;
        color: #ffffff !important;
        border: 4px solid var(--text-color) !important;
        border-radius: 0px !important;
        font-weight: 900 !important;
        text-transform: uppercase !important;
        padding: 20px 50px !important;
        font-size: 16px !important;
        font-family: 'Arial Black', sans-serif !important;
        box-shadow: 6px 6px 0px var(--accent-green) !important;
    }
    .stButton > button:hover {
        background-color: var(--accent-green) !important;
        border-color: var(--text-color) !important;
        box-shadow: 6px 6px 0px var(--accent-purple) !important;
        transform: translateY(-2px) !important;
        color: #ffffff !important;
    }

    /* --- EXPANDER & PDF --- */
    .streamlit-expanderHeader {
        border: 4px solid var(--accent-black);
        background-color: var(--accent-orange);
        color: var(--accent-black);
        font-weight: 900;
        text-transform: uppercase;
        font-family: 'Arial Black', sans-serif;
        padding: 15px;
        box-shadow: 4px 4px 0px var(--accent-black);
    }
    .streamlit-expanderContent {
        border: 4px solid var(--accent-black);
        border-top: none;
        background-color: var(--card-bg);
        padding: 20px;
        box-shadow: 4px 4px 0px var(--accent-black);
    }

    [data-testid="stSidebar"] {
        background-color: var(--accent-black);
        border-right: 8px solid var(--accent-orange);
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-family: 'Arial Black', sans-serif;
        font-weight: 700;
    }
    
    .stProgress > div > div > div {
        background-color: var(--accent-orange);
        box-shadow: 4px 4px 0px var(--accent-black);
        height: 15px;
    }
    
    /* --- ADDITIONAL BRUTALIST ELEMENTS --- */
    .stAlert {
        border: 4px solid var(--accent-black) !important;
        border-radius: 0px !important;
        background-color: var(--accent-orange) !important;
        color: var(--accent-black) !important;
        font-weight: 900 !important;
        box-shadow: 6px 6px 0px var(--accent-black) !important;
    }
    
    .stSpinner {
        border: 4px solid var(--accent-black) !important;
    }
    
    /* Fix file uploader text */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        border-radius: 0px !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background-color: var(--card-bg) !important;
        border: 6px solid var(--text-color) !important;
        padding: 30px !important;
        box-shadow: 8px 8px 0px var(--text-color) !important;
        border-radius: 0px !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: var(--text-color) !important;
        font-weight: 900 !important;
        font-family: 'Arial Black', sans-serif !important;
        font-size: 1.5rem !important;
        text-transform: uppercase !important;
        margin-bottom: 20px !important;
        display: block !important;
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
        background-color: var(--accent-purple) !important;
        color: #ffffff !important;
        border: 4px dashed var(--accent-green) !important;
        padding: 40px !important;
        text-align: center !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        font-family: 'Arial', sans-serif !important;
        border-radius: 0px !important;
    }
    
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] p {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        margin: 0 !important;
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] small {
        color: var(--accent-green) !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background-color: var(--accent-green) !important;
        color: #ffffff !important;
        border: 3px solid var(--text-color) !important;
        font-weight: 900 !important;
        padding: 15px 30px !important;
        font-size: 1rem !important;
        text-transform: uppercase !important;
        border-radius: 0px !important;
    }
    
    /* Fix all Streamlit text elements */
    .stMarkdown {
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
        margin: 10px 0 !important;
        box-shadow: none !important;
    }
    
    .stMarkdown p {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        line-height: 1.5 !important;
        margin-bottom: 10px !important;
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    .stMarkdown ul {
        background-color: var(--card-bg) !important;
        padding: 20px !important;
        border-left: 8px solid var(--accent-green) !important;
        margin: 15px 0 !important;
        border-radius: 0px !important;
        box-shadow: 4px 4px 0px var(--text-color) !important;
    }
    
    .stMarkdown li {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
        margin-bottom: 8px !important;
        padding-left: 10px !important;
        background-color: transparent !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-color) !important;
        font-weight: 900 !important;
        font-family: 'Arial Black', sans-serif !important;
        background-color: transparent !important;
        padding: 0 !important;
        margin: 10px 0 !important;
    }
    
    /* Fix info messages */
    [data-testid="stInfo"] {
        background-color: var(--accent-green) !important;
        color: var(--text-color) !important;
        border: 4px solid var(--text-color) !important;
        font-weight: 900 !important;
        padding: 25px !important;
        font-size: 1.3rem !important;
        text-align: center !important;
        box-shadow: 6px 6px 0px var(--text-color) !important;
        font-family: 'Arial Black', sans-serif !important;
        text-transform: uppercase !important;
        margin: 20px 0 !important;
        border-radius: 0px !important;
    }
    
    [data-testid="stInfo"] div {
        color: var(--text-color) !important;
        font-weight: 900 !important;
        background-color: transparent !important;
    }
    
    /* Fix success messages */
    [data-testid="stSuccess"] {
        background-color: var(--accent-purple) !important;
        color: #ffffff !important;
        border: 4px solid var(--text-color) !important;
        font-weight: 900 !important;
        padding: 25px !important;
        font-size: 1.2rem !important;
        box-shadow: 6px 6px 0px var(--accent-green) !important;
        border-radius: 0px !important;
    }
    
    /* Fix all text elements with better targeting */
    .main p, .main span {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    
    .main div:not(.chat-bubble):not(.kw-tier-1):not(.ai-bubble):not(.kw-tier-2):not(.kw-standard) {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    .main label {
        color: var(--text-color) !important;
        font-weight: 700 !important;
        background-color: transparent !important;
    }
    
    /* Fix expander text */
    [data-testid="stExpander"] {
        background-color: var(--card-bg) !important;
        border: 4px solid var(--text-color) !important;
        box-shadow: 4px 4px 0px var(--text-color) !important;
        border-radius: 0px !important;
    }
    
    [data-testid="stExpander"] p {
        color: var(--text-color) !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    
    /* Override Streamlit defaults with better approach */
    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
    }
    
    .main .block-container {
        background-color: var(--bg-color) !important;
        color: var(--text-color) !important;
        padding-top: 2rem !important;
        max-width: 100% !important;
    }
    
    /* Remove default containers that create white boxes */
    .element-container {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 5px 0 !important;
    }
    
    /* Target main content area text */
    .main .stMarkdown, .main .element-container {
        color: var(--text-color) !important;
        background-color: transparent !important;
    }
    
    /* But keep specific styled elements */
    .ai-bubble * {
        color: #ffffff !important;
    }
    
    .kw-tier-1 * {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    .stButton > button {
        color: #ffffff !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
    }
    
    /* Fix form elements */
    [data-testid="stForm"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    
    /* Fix dividers */
    hr {
        border: 2px solid var(--text-color) !important;
        background-color: var(--text-color) !important;
        margin: 30px 0 !important;
    }
    
    /* Remove unwanted white containers */
    .stContainer {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        h1 { 
            font-size: 2.5rem; 
            padding: 20px 30px; 
        }
        
        h2 {
            font-size: 1.5rem;
            padding: 10px 20px;
        }
        
        .kw-grid { 
            grid-template-columns: 1fr; 
            gap: 10px;
        }
        
        .stTabs [data-baseweb="tab"] { 
            padding: 12px 16px; 
            font-size: 1rem; 
        }
        
        .stTextInput > div > div > input { 
            padding: 15px; 
            font-size: 16px;
        }
        
        .chat-bubble { 
            margin-left: 0; 
            margin-right: 0; 
            padding: 20px;
        }
        
        .summary-card {
            padding: 25px;
            font-size: 1rem;
        }
        
        [data-testid="stFileUploader"] {
            padding: 20px !important;
        }
        
        [data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
            padding: 25px !important;
            font-size: 1rem !important;
        }
    }
    
    @media (max-width: 480px) {
        h1 {
            font-size: 2rem;
            padding: 15px 25px;
        }
        
        .kw-tier-1 {
            font-size: 1.2rem;
            padding: 20px 30px;
        }
        
        .kw-tier-2, .kw-standard {
            font-size: 0.9rem;
            padding: 12px 18px;
        }
    }
</style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# --- API Key Validation ---
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    st.error("🚫 GROQ_API_KEY not found.", icon="🔑")
    st.stop()

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
    """Displays PDF with System/Blueprint styling."""
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" 
                height="1200px" 
                type="application/pdf"
                style="border: 2px solid var(--neon-green); background: var(--bg-color); box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);">
        </iframe>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_embedding_model():
    with st.spinner("INITIALIZING SYSTEM..."):
        return SentenceTransformer('all-MiniLM-L6-v2')

# --- UI Layout ---
st.markdown("# PDF ANALYZER")
st.markdown("##DOCUMENT INSIGHT SYSTEM")

uploaded_file = st.file_uploader("UPLOAD SOURCE DOCUMENT", type=['pdf'])

if uploaded_file is not None:
    current_filename = uploaded_file.name
    
    if st.session_state.filename != current_filename:
        status_placeholder = st.empty()
        bar = st.progress(0)
        
        try:
            status_placeholder.info("STATUS: READING FILE...")
            raw_text = extract_text_from_pdf(uploaded_file)
            
            if not raw_text:
                st.error("ERROR: COULD NOT READ TEXT.")
                st.stop()

            bar.progress(10)
            status_placeholder.info("STATUS: LOADING NEURAL NETWORKS...")
            
            local_model = load_embedding_model()
            summarizer = SemanticSummarizer(api_key=api_key)
            extractor = ConceptExtractor(api_key=api_key)
            
            bar.progress(20)
            status_placeholder.info("STATUS: ANALYZING CONTENT...")

            summary = summarizer.summarize(raw_text)
            
            bar.progress(70)
            status_placeholder.info("STATUS: EXTRACTING CONCEPTS...")

            keywords = extractor.extract_keywords(summary)

            bar.progress(90)
            status_placeholder.info("STATUS: PREPARING INTERFACE...")

            qa_system = SemanticQA(local_model, api_key=api_key)
            qa_system.prepare_context(raw_text)

            bar.progress(100)
            status_placeholder.success("STATUS: SYSTEM ONLINE.")

            st.session_state.text = raw_text
            st.session_state.summary = summary
            st.session_state.keywords = keywords
            st.session_state.qa_system = qa_system
            st.session_state.filename = current_filename
            st.session_state.history = [] 
            
            time.sleep(1)
            st.rerun() 

        except Exception as e:
            status_placeholder.error(f"SYSTEM ERROR: {str(e)}")
            st.stop()

if uploaded_file is not None:
    with st.expander(">> VIEW RAW SOURCE DOCUMENT"):
        uploaded_file.seek(0) 
        display_pdf(uploaded_file)

if st.session_state.text:
    st.divider()
    
    # --- BRUTALIST TAB STRUCTURE ---
    tab1, tab2, tab3 = st.tabs(["⚡ SUMMARY", "⚡ KEY TOPICS", "⚡ INTERROGATE"])
    
    # TAB 1: SUMMARY
    with tab1:
        if st.session_state.summary:
            with st.container():
                st.markdown('<div class="summary-card">', unsafe_allow_html=True)
                st.markdown(st.session_state.summary, unsafe_allow_html=False)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("⚡ NO SUMMARY AVAILABLE")

    # TAB 2: KEY TOPICS
    with tab2:
        st.markdown("### SYSTEM ANALYSIS COMPLETE")
        st.markdown("###")
        if st.session_state.keywords:
            # Render first 2 as Critical
            st.markdown('<div class="kw-grid">', unsafe_allow_html=True)
            for i, kw in enumerate(st.session_state.keywords[:2]):
                st.markdown(f'<span class="kw-tier-1">CRITICAL: {kw}</span>', unsafe_allow_html=True)
            
            # Render remaining as Standard
            for kw in st.session_state.keywords[2:]:
                st.markdown(f'<span class="kw-standard">TOPIC: {kw}</span>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("⚡ NO TOPICS EXTRACTED")

    # TAB 3: INTERROGATE
    with tab3:
        st.markdown("### DOCUMENT INTERROGATION")
        
        with st.form("q_form", clear_on_submit=True):
            user_input = st.text_input("ENTER QUERY:", placeholder="ASK ANYTHING ABOUT THE DOCUMENT...")
            submit_button = st.form_submit_button("⚡ EXECUTE QUERY")
        
        if submit_button and user_input:
            if st.session_state.qa_system:
                answer = st.session_state.qa_system.ask(user_input)
                st.session_state.history.append((user_input, answer))
                st.rerun() 

        chat_container = st.container()
        with chat_container:
            if not st.session_state.history:
                st.markdown('<div style="text-align:center; opacity: 0.7; font-family: Arial Black; font-size: 1.5rem; margin-top: 50px; color: var(--accent-black); font-weight: 900;">⚡ NO ACTIVE INTERROGATIONS ⚡</div>', unsafe_allow_html=True)
            else:
                for q, a in reversed(st.session_state.history):
                    st.markdown(f'<div class="chat-bubble user-bubble">USER QUERY: {q}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chat-bubble ai-bubble">SYSTEM RESPONSE: {a}</div>', unsafe_allow_html=True)

else:
    st.info("⚡ SYSTEM READY - UPLOAD DOCUMENT TO BEGIN ⚡")