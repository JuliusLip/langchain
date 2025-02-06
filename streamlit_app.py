import streamlit as st
from sql_agent_final import graph
from langchain_core.messages import HumanMessage, AIMessage

config = {"configurable": {"thread_id": "1"}}

# >>>>>>>>>>STREAMLIT PART<<<<<<<<<<<

st.set_page_config(
    page_icon="🤖",
    page_title="SQL agent",
    layout="centered"
)

st.title("SQL agent")

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

# left_col, main_col, right_col = st.columns([1, 2, 1])

with st.sidebar:
    if st.button('Clear Chat'):
        st.session_state.message_history = []

# with main_col:

# conversation
for message in st.session_state.message_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:   
        with st.chat_message("AI"):
            st.markdown(message.content)

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state.message_history.append(HumanMessage(content=user_input))

    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        response = graph.invoke({
            'messages': st.session_state.message_history
        }, config=config)

        st.markdown(response['messages'][-1].content)

    st.session_state.message_history.append(AIMessage(content=response['messages'][-1].content))