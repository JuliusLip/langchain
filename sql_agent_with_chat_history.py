from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent
# from langchain_core.messages import HumanMessage, AIMessage
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
    # MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad") # the placeholder for an agent to keep track it's goals, steps and outputs
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
def process_chat(agentExecutor, user_input): # , chat_history
    response = agentExecutor.invoke({
        "input": user_input,
        # "chat_history": chat_history
    })
    return response["output"]

if __name__ == '__main__':
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(agentExecutor, user_input) # , chat_history
        # chat_history.append(HumanMessage(content=user_input))
        # chat_history.append(AIMessage(content=response))

        print("Assistant:", response)