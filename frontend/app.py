"""
AgentXploit — frontend entry point.

Start the backend first:
    cd backend && uvicorn main:app --reload

Then start the frontend (from the project root):
    streamlit run frontend/app.py
"""

import streamlit as st

import views.chat as chat
import views.home as home
from state import init_session_state

st.set_page_config(page_title="AgentXploit", page_icon="🔐", layout="wide")

init_session_state()

if st.session_state.page == "home":
    home.show()
else:
    chat.show()
