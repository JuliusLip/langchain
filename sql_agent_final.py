import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)


# Load environment variables from .env file
load_dotenv()

# Create a Database object
sqlite_uri = "sqlite:///Chinook.db"
db = SQLDatabase.from_uri(sqlite_uri)

# Instantiate the LLM model
llm = ChatOpenAI(model='gpt-4o-mini')

# Load the SQL tools for AI agent to use
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Construct the Tools agent
memory = MemorySaver()
langgraph_agent_executor = create_react_agent(llm, tools, checkpointer=memory)
config = {"configurable": {"thread_id": 1}}

# Create a function to process a chat
def process_chat(agentExecutor, user_input, history):
    response = agentExecutor.invoke({"messages": history + [("human", user_input)]}, config)
    return response["messages"][-1].content

# >>>>>>>>>>STREAMLIT PART<<<<<<<<<<<

st.set_page_config(
    page_icon="🤖",
    page_title="Chat with SQL database",
    layout="centered"
)

st.title("Chat with SQL database")

history = StreamlitChatMessageHistory(key="chat_history")

# conversation
for message in history.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# user input
user_input = st.chat_input('Chat with your mysql database')
if user_input:
    history.add_user_message(user_input)

    with st.chat_message("Human"):
        st.markdown(user_input)

    with st.chat_message("AI"):
        ai_response = process_chat(langgraph_agent_executor, user_input, history.messages)
        st.markdown(ai_response)

    history.add_ai_message(ai_response)
