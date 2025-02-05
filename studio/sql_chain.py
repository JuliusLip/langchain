from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain import hub
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph


# Define a State class to preserve all the required valriables accross the steps
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


# Create a function for SQL queries generation
class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


# Create a function for SQL queries generation
def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    
    return {"query": result["query"]}


# Create a function for query executing
def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


# Create a function for answer generation
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


# Define the database URI
db_uri = "sqlite:///chinook.db"

# Create a database object
db = SQLDatabase.from_uri(db_uri)

# Instantiate LLM model
llm = ChatOpenAI(model="gpt-4o-mini")

# Pull the prebuilt prompt template for sql query generation
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

# Build a LangGraph graph (chain) by adding all the steps into the sequence
graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)
graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()
