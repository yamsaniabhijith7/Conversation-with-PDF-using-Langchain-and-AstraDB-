# Quickstart: Querying PDF With Astra and LangChain
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from datasets import load_dataset
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

# Pre-requisites
!pip install -q cassio datasets langchain openai tiktoken
!pip install PyPDF2

# Setup
ASTRA_DB_APPLICATION_TOKEN = "Your_AstraDB_Token"
ASTRA_DB_ID = "Your_Database_ID"
OPENAI_API_KEY = "Your_OpenAI_API_Key"

# Provide your secrets:
pdfreader = PdfReader('budget_speech.pdf')

# Read text from PDF
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# Initialize the connection to your database
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Create the LangChain embedding and LLM objects for later usage
llm = OpenAI(openai_api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Create your LangChain vector store backed by Astra DB!
astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None,
)

# Split the text using Character Text Splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Load the dataset into the vector store
astra_vector_store.add_texts(texts[:50])
print("Inserted %i headlines." % len(texts[:50]))
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

# Run the QA cycle
first_question = True
while True:
    if first_question:
        query_text = input("\nEnter your question (or type 'quit' to exit): ").strip()
    else:
        query_text = input("\nWhat's your next question (or type 'quit' to exit): ").strip()
    if query_text.lower() == "quit":
        break
    if query_text == "":
        continue
    first_question = False
    print("\nQUESTION: \"%s\"" % query_text)
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print("ANSWER: \"%s\"\n" % answer)
    print("FIRST DOCUMENTS BY RELEVANCE:")
    for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
        print(" [%0.4f] \"%s ...\"" % (score, doc.page_content[:84]))
