import streamlit as st
from typing import Annotated, Sequence, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import os
import re
from main import chat_with_airtel  # Import the chat function from your original file

# Page config
st.set_page_config(
    page_title="Airtel Customer Service Bot",
    page_icon="📱",
    layout="centered"
)

# Custom CSS for chat interface
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        background-color: white;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to clean HTML tags from text
def clean_html(text):
    # Remove HTML tags
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Initialize session state for chat history and bot state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bot_state" not in st.session_state:
    st.session_state.bot_state = None

# Callback function to handle message sending
def process_input():
    user_message = st.session_state.user_input
    if user_message:
        # Add user message to chat history
        st.session_state.chat_history.append(("user", user_message))
        
        # Get bot response
        st.session_state.bot_state = chat_with_airtel(
            user_message, 
            st.session_state.bot_state
        )
        
        # Extract bot response from state and clean HTML tags
        bot_response = clean_html(st.session_state.bot_state["messages"][-1].content)
        
        # Add bot response to chat history
        st.session_state.chat_history.append(("bot", bot_response))

# Header
st.title("Airtel Customer Service Bot 📱")
st.markdown("""
    Welcome to Airtel's customer service! I can help you with:
    - General information about Airtel services
    - SIM swap requests
    - Checking your plan details
""")

# Chat interface
chat_container = st.container()

# Display chat history
with chat_container:
    for role, message in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'<div class="user-message">You: {message}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">Airtel Bot: {message}</div>', 
                       unsafe_allow_html=True)

# User input
st.text_input(
    "Type your message here...", 
    key="user_input",
    on_change=process_input
)

# Footer
st.markdown("---")
st.markdown("*Pratham Batra*")