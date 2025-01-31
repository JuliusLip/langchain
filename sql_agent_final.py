import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environment variables from .env file
load_dotenv()

# Create a Database object
sqlite_uri = "sqlite:///Chinook.db"
db = SQLDatabase.from_uri(sqlite_uri)

# Instantiate the LLM model
llm = ChatOpenAI(model='gpt-4o-mini')

# Use the SQL agent system prompt from langchain hub
system_message = hub.pull("langchain-ai/sql-agent-system-prompt")

# Create a full prompt template with system message, chat history and user input
prompt = ChatPromptTemplate.from_messages([
    system_message.format(dialect=db.dialect, top_k=10),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Load the SQL tools for AI agent to use
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)

agentExecutor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Create a function to process a chat
def process_chat(agentExecutor, user_input, chat_history):
    response = agentExecutor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    return response["output"]


# >>>>>>>>>>STREAMLIT PART<<<<<<<<<<<

st.set_page_config(
    page_icon="🤖",
    page_title="Chat with SQL database",
    layout="centered"
)

st.title("Chat with SQL database")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# conversation
for message in st.session_state.chat_history:
    # st.chat_message(chat['role']).markdown(chat['content'])
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# user input
question = st.chat_input('Chat with your mysql database')
if question:
    st.session_state.chat_history.append(HumanMessage(question))

    with st.chat_message("Human"):
        st.markdown(question)

    with st.chat_message("AI"):
        ai_response = process_chat(agentExecutor, question, st.session_state.chat_history)

        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))