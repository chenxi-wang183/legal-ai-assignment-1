import streamlit as st

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None

def render_sidebar():
    st.sidebar.title("API Key Input")
    api_key = st.sidebar.text_input("Enter your API key:")
    update_button = st.sidebar.button("Update API Key")
    return api_key, update_button

api_key, update_button = render_sidebar()

if update_button:
    st.session_state.openai_api_key = api_key

st.write(f"Current API Key: {st.session_state.openai_api_key}")