import streamlit as st
import random
import time
from RAG_system import *

# run this with "streamlit run RAG_chat.py"

# Streamed response emulator
def response_generator(prompt, constraint=None):
    response = Query(prompt, constraint)  # query with constraint of 1000
    
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("RAG chat")

constraint = st.number_input(
    "Choose constraint", value=None, placeholder="Enter a constraint..."
)

if constraint != None:
    st.write("The current number constraint is ", constraint)
else:
    st.write("Starting with no constraint!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, constraint))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})