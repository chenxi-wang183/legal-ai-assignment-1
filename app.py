import streamlit as st
import os
import tempfile
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

# --- PAGE CONFIGURATION (Must be the first command) ---
st.set_page_config(page_title="Legal AI Assistant", layout="wide")


# --- CUSTOM STYLES & ICONS ---
scales_svg = """
<svg width="80" height="80" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M21 6L18.5 5M18.5 5L15 6M18.5 5V12M18.5 12L21 13M18.5 12L15 13M3 6L5.5 5M5.5 5L9 6M5.5 5V12M5.5 12L3 13M5.5 12L9 13M12 3V21M3 21H21" stroke="#1e3a8a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""
sources_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
</svg>
"""
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&display=swap');
    .stApp { background-color: #ffffff; }
    .main-container { display: flex; flex-direction: column; align-items: center; padding: 1rem 1rem 5rem 1rem; }
    .title { font-family: 'Cormorant Garamond', serif; font-weight: 600; font-size: 5rem; color: #1e3a8a; text-align: center; margin-top: 2rem; margin-bottom: 0rem; }
    .subtitle { font-family: 'Cormorant Garamond', serif; font-weight: 400; color: #9ca3af; text-align: center; margin-top: 3rem; margin-bottom: 3rem; font-size: 1.25rem; line-height: 1.6; }
    [data-testid="stHorizontalBlock"] { align-items: center; }
    .stTextInput input { background-color: #ffffff !important; border: 1px solid #d1d5db; height: 3.5rem; border-radius: 0.5rem; padding-left: 1rem; }
    [data-testid="stFileUploader"] { width: 3.5rem; height: 3.5rem; }
    [data-testid="stFileUploader"] section { border: none; padding: 0; }
    [data-testid="stFileUploader"] section > input + div { display: none; }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] { display: none; }
    [data-testid="stFileUploader"] section button { display: flex; justify-content: center; align-items: center; background-color: transparent !important; border: none !important; color: #9ca3af !important; padding: 0 !important; width: 3.5rem; height: 3.5rem; font-size: 0; }
    [data-testid="stFileUploader"] section button:hover { background-color: #eef2ff !important; color: #2563eb !important; }
    [data-testid="stFileUploader"] section button::after { content: '+'; font-size: 2.5rem; font-weight: 300; line-height: 1; }
    .stButton>button { background-color: transparent !important; color: #9ca3af !important; border: none !important; width: 3.5rem !important; height: 3.5rem !important; font-size: 1.5rem !important; padding: 0 !important; margin: 0 !important; }
    .stButton>button:hover { background-color: #eef2ff !important; color: #2563eb !important; }
    .response-container { font-size: 1.1rem; line-height: 1.7; }
    .stSidebar { background-color: #ffffff; }
    blockquote { background-color: #f3f4f6; border-left: 5px solid #d1d5db; padding: 1rem; border-radius: 0.25rem; color: #4b5563; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- SESSION STATE & CALLBACKS ---
# (Unchanged from the final version)
# ...

# --- SIDEBAR ---
with st.sidebar:
    st.header("API Credentials")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key", type="password", key="api_key_input_sidebar",
        label_visibility="collapsed", placeholder="Enter your OpenAI API Key..."
    )

    if api_key_input:
        st.session_state.api_key = api_key_input
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.info(f"Key received, ending in: ...{api_key_input[-4:]}")

    st.header("Search History")
    # (History logic unchanged)
    # ...

# --- MAIN PAGE UI ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div style="text-align: center;">{scales_svg}</div>', unsafe_allow_html=True)
st.markdown('<p class="title">Legal AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a legal document.<br>Ask a question.<br>Get a structured analysis.</p>', unsafe_allow_html=True)

if not st.session_state.api_key:
    st.warning("Please enter your OpenAI API Key in the sidebar to begin.")
else:
    # --- SEARCH BAR AND FILE UPLOADER ---
    # (Unchanged)
    # ...

    # --- DOCUMENT INDEXING ---
    if uploaded_file:
        # (Unchanged)
        # ...

    # --- QUERY LOGIC ---
    if analyze_button_clicked:
        if st.session_state.rag_engine and user_question:
            with st.spinner("Thinking..."):
                try:
                    # --- DEBUGGING BLOCK AS PER TEACHER'S SUGGESTION ---
                    st.write("---")
                    st.subheader("🕵️‍♂️ Debugging Information")
                    
                    llm_model_name = "gpt-4o" # Forcing gpt-4o for this test
                    
                    # Configure LlamaIndex Settings right before the call
                    Settings.llm = OpenAI(model=llm_model_name, api_key=st.session_state.api_key)
                    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=st.session_state.api_key)

                    st.info(f"**Attempting to use LLM:** `{Settings.llm.model}`")
                    st.info(f"**Using API Key ending in:** `...{st.session_state.api_key[-4:]}`")
                    st.write("---")
                    # --- END DEBUGGING BLOCK ---

                    response_obj = st.session_state.rag_engine.query(user_question)
                    
                    # (Response parsing and display logic remains the same)
                    # ...

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        # (Warning logic remains the same)
        # ...

    # (Result display logic remains the same)
    # ...

st.markdown('</div>', unsafe_allow_html=True)

    # --- DISPLAY RESPONSE ---
    if st.session_state.analysis_result:
        response_obj, formatted_answer = st.session_state.analysis_result
        st.write("---")
        st.markdown('<div class="response-container">', unsafe_allow_html=True)
        st.markdown(formatted_answer, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        expander_title = f'<div style="display: flex; align-items: center; gap: 10px;">{sources_svg}<span>Show Cited Sources</span></div>'
        st.markdown(expander_title, unsafe_allow_html=True)
        with st.expander(" ", expanded=False):
            for i, node in enumerate(response_obj.source_nodes):
                with st.container(border=True):
                    page_label = node.metadata.get('page_label')
                    if page_label:
                        st.markdown(f"**Source from Page: {page_label}** (Similarity: {node.score:.4f})")
                    else:
                        st.markdown(f"**Source {i+1}** (Similarity: {node.score:.4f})")
                    st.write(node.get_text())

st.markdown('</div>', unsafe_allow_html=True)