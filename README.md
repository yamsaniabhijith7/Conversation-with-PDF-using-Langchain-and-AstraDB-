# PDFQuery_With_Astra_LangChain

## Introduction
This project demonstrates how to query PDFs using Astra DB and LangChain. It involves setting up a vector store in Astra DB, indexing PDF content, and querying it using LangChain and OpenAI.

## Prerequisites
- Astra DB account with a Serverless Cassandra and Vector Search database.
- Astra DB Token with Database Administrator role.
- OpenAI API Key.
- Python packages: cassio, datasets, langchain, openai, tiktoken, PyPDF2.

## Setup
1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Replace the placeholders in `main.py` with your Astra DB connection details and OpenAI API key.

3. Place the PDF file (`budget_speech.pdf`) in the project directory.

## Running the Project
1. Run the script:
    ```bash
    python main.py
    ```

2. Follow the prompts to ask questions and get answers from the PDF content.

## Note
This project is intended for demonstration purposes and should be used with appropriate caution and security measures.
