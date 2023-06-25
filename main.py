import streamlit as st
import random
from streamlit_chat import message
from app import get_response

def process_input(user_input):
    response = get_response(user_input)
    return response

st.header("HR Chatbot")
st.markdown("Ask your HR-related questions here.")

st.markdown("What is my name and employee id?")
st.markdown("Are employees on probation allowed to have vacation leaves?")
st.markdown("How many vacation leaves do I have left and what is the policy on unused VLs?")
st.markdown("How much will I be paid if I encash my unused VLs?")
st.markdown("My Salary is 15000. How much will it be I get a 7 pct increase?")

if "past" not in st.session_state:
    st.session_state["past"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "input_message_key" not in st.session_state:
    st.session_state["input_message_key"] = str(random.random())

chat_container = st.container()

user_input = st.text_input("Type your message and press Enter to send.", key=st.session_state["input_message_key"])

if st.button("Send"):
    response = process_input(user_input)

    st.session_state["past"].append(user_input)
    st.session_state["generated"].append(response)

    st.session_state["input_message_key"] = str(random.random())

    st.experimental_rerun()

if st.session_state["generated"]:
    with chat_container:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
