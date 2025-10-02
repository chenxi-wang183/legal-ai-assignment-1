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
if "rag_engine" not in st.session_state: st.session_state.rag_engine = None
if "history" not in st.session_state: st.session_state.history = []
if "run_analysis" not in st.session_state: st.session_state.run_analysis = False
if "analysis_result" not in st.session_state: st.session_state.analysis_result = None
if "api_key_configured" not in st.session_state: st.session_state.api_key_configured = False

def trigger_analysis():
    if st.session_state.get("question_input"):
        st.session_state.run_analysis = True

# --- SIDEBAR ---
with st.sidebar:
    st.header("API Credentials")
    api_key_input = st.text_input(
        "Enter your OpenAI API Key", type="password", key="api_key_input_sidebar",
        label_visibility="collapsed", placeholder="Enter your OpenAI API Key..."
    )

    if api_key_input:
        try:
            Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=api_key_input)
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key_input)
            if not st.session_state.api_key_configured:
                st.session_state.api_key_configured = True
                st.rerun()
        except Exception as e:
            st.error(f"Invalid API Key: {e}")
            st.session_state.api_key_configured = False
    
    st.header("Search History")
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
    
    if not st.session_state.history:
        st.write("No searches yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            with st.expander(f"**{i+1}. {item['question'][:50]}...**"):
                st.markdown(item.get('formatted_answer', item.get('answer', 'No answer recorded.')), unsafe_allow_html=True)

# --- MAIN PAGE UI ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown(f'<div style="text-align: center;">{scales_svg}</div>', unsafe_allow_html=True)
st.markdown('<p class="title">Legal AI Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a legal document.<br>Ask a question.<br>Get a structured analysis.</p>', unsafe_allow_html=True)

if not st.session_state.api_key_configured:
    st.warning("Please enter your OpenAI API Key in the sidebar to begin.")
else:
    search_bar_cols = st.columns([1, 8, 1])
    with search_bar_cols[0]:
        uploaded_file = st.file_uploader("Upload", type=["pdf", "txt", "docx"], key="file_uploader", label_visibility="collapsed")
    with search_bar_cols[1]:
        user_question = st.text_input("Enter your question...", key="question_input", label_visibility="collapsed")
    with search_bar_cols[2]:
        st.button("➤", on_click=trigger_analysis, key="analyze_button", help="Analyze the document", use_container_width=True)

    system_prompt = (
        "You are an expert legal AI assistant. Your task is to provide a detailed and structured "
        "analysis of a user's question based *only* on the provided text from a legal document. "
        "You must follow this four-step process for every answer:\n\n"
        "[ANALYSIS]\n"
        "First, break down the user's question into its core components...\n\n"
        "[RELEVANT_CLAUSES]\n"
        "Next, identify and quote the exact clause or clauses...\n\n"
        "[DIRECT_ANSWER]\n"
        "After quoting the relevant text, provide a direct, one-sentence summary answer...\n\n"
        "[REASONING]\n"
        "Finally, explain *why* the clause(s) you quoted lead to the direct answer...\n\n"
        "**Crucial Rule:** If you cannot find any relevant information..., state only: "
        "'Based on the provided document, I could not find a specific answer to this question.'"
    )
    
    if uploaded_file:
        if "last_uploaded_filename" not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
            with st.spinner("Indexing the document..."):
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()
                    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
                    index = VectorStoreIndex.from_documents(documents, transformations=[text_splitter])
                    st.session_state.rag_engine = index.as_query_engine(similarity_top_k=5)
                    st.session_state.last_uploaded_filename = uploaded_file.name
                st.success("✅ Document indexed successfully!")
                st.session_state.analysis_result = None

    if st.session_state.run_analysis:
        st.session_state.run_analysis = False
        question_to_run = st.session_state.get("question_input", "")

        if st.session_state.rag_engine and question_to_run:
            with st.spinner("Thinking..."):
                try:
                    # Make sure the prompt is set correctly on the LLM before querying
                    Settings.llm.system_prompt = system_prompt

                    response_obj = st.session_state.rag_engine.query(question_to_run)
                    response_text = str(response_obj)
                    
                    parts = response_text.split('[')
                    analysis, clauses, answer, reasoning = "", "", "", ""
                    for part in parts:
                        if part.startswith("ANALYSIS]"): analysis = part.replace("ANALYSIS]", "").strip()
                        elif part.startswith("RELEVANT_CLAUSES]"): clauses = part.replace("RELEVANT_CLAUSES]", "").strip()
                        elif part.startswith("DIRECT_ANSWER]"): answer = part.replace("DIRECT_ANSWER]", "").strip()
                        elif part.startswith("REASONING]"): reasoning = part.replace("REASONING]", "").strip()
                    
                    formatted_answer = ""
                    if analysis or reasoning:
                        formatted_answer += f"<blockquote><b>Analysis & Reasoning:</b><br>{analysis}<br><br>{reasoning}</blockquote><hr>"
                        formatted_answer += f"<b>Direct Answer:</b><br>{answer}<br><hr>"
                        formatted_answer += f"<b>Relevant Legal Clause(s):</b><br>{clauses}"
                    else:
                        formatted_answer = response_text
                    
                    st.session_state.analysis_result = (response_obj, formatted_answer)

                    if not any(d['question'] == question_to_run for d in st.session_state.history):
                        st.session_state.history.append({'question': question_to_run, 'formatted_answer': formatted_answer})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
        elif not st.session_state.rag_engine:
             st.warning("⚠️ Please upload a document first.")
        else:
            st.warning("⚠️ Please enter a question.")

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