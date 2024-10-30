from dotenv import load_dotenv
import os
import textwrap
import openai
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
import warnings

# Warning control
warnings.filterwarnings("ignore")

# Load from environment
load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
# NEO4J_PASSWORD = "inMorphis24"
NEO4J_DATABASE = 'neo4j'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Global constants
VECTOR_INDEX_NAME = 'form_10k_chunks'
VECTOR_NODE_LABEL = 'Chunk'
VECTOR_SOURCE_PROPERTY = 'text'
VECTOR_EMBEDDING_PROPERTY = 'textEmbedding'

# Initialize Neo4j connection
kg = Neo4jGraph(
    url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, database=NEO4J_DATABASE
)
kg.refresh_schema()
print(textwrap.fill(kg.schema, 60))

# Test query to ensure connection
kg.query("""
MATCH (n:NotificationEmailScript)
RETURN COUNT(n) AS NumberOfNotifications
LIMIT 2
""")

# Cypher query generation template
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.
Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
{schema}
Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples:
# How many transform maps are there in itsm framework?
MATCH (tm:TransformMap)
RETURN COUNT(tm) AS NumberOfTransformMaps

# how many notifications are there?
MATCH (n:NotificationEmailScript)
RETURN COUNT(n) AS NumberOfNotifications

# how many dashboards are there in itsm framework?
MATCH (d:Dashboard)
RETURN COUNT(d) AS NumberOfDashboards

# Name the dashboards
MATCH (d:Dashboard)
RETURN d.name AS DashboardName, COUNT(d) AS NumberOfDashboards

# Name the available notifications
MATCH (n:NotificationEmailScript)
RETURN n.name AS NotificationName

# "What aspects of the ITSM Ace framework help in improving efficiency, effectiveness, and visibility?"
MATCH (d:Dashboard)
WHERE d.name IN ["ITSM Agent Dashboard", "ITSM Group Manager Dashboard", "ITSM Director Dashboard", "ITSM CXO Dashboard"]
RETURN d

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], 
    template=CYPHER_GENERATION_TEMPLATE
)

# Create Cypher Chain using Langchain
cypherChain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0),
    graph=kg,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
    allow_dangerous_requests=True
)

# Function to query using the chain
def prettyCypherChain(question: str) -> str:
    response = cypherChain.run(question)
    return textwrap.fill(response, 60)

# Example query
print(prettyCypherChain("what do you know about ITSM Framework?"))
