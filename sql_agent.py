from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environment variables from .env file
load_dotenv()

# Create a Database object
# sqlite_uri = "sqlite:///Chinook.db"
sqlite_uri = "sqlite:///elections.db"
db = SQLDatabase.from_uri(sqlite_uri)

# Instantiate the LLM model
llm = ChatOpenAI(model='gpt-4o-mini')

# Load the SQL tools for AI agent to use
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Construct the Tools agent
langgraph_agent_executor = create_react_agent(llm, tools)
config = {"configurable": {"thread_id": 1}}

# Create a function to process a chat
def process_chat(agentExecutor, user_input):
    response = agentExecutor.invoke({"messages": [("human", user_input)]}, config)
    return response["messages"][-1].content

if __name__ == '__main__':

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        response = process_chat(langgraph_agent_executor, user_input)
        print("Assistant:", response)
