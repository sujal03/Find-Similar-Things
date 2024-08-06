import streamlit as st
import os
import logging
from dotenv import load_dotenv
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Ensure COHERE_API_KEY is set
if not os.getenv("COHERE_API_KEY"):
    raise ValueError("COHERE_API_KEY environment variable not set")

st.set_page_config(page_title="Educate Kids", page_icon=":robot:")
st.header("Hey, ask me something & I will give similar things...")

model_name = "embed-english-v3.0"
user_agent = "my-app/1.0"  # Replace with your user agent

embeddings = CohereEmbeddings(model=model_name, user_agent=user_agent)

loader = CSVLoader(file_path="myData.csv", csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

data = loader.load()

# Print and log data for debugging
logging.debug(f"Loaded data: {data}")
print(f"Loaded data: {data}")

# Extract texts from data and ensure they are valid
texts = [doc.page_content for doc in data]

# Validate the data being passed to the embeddings
for text in texts:
    logging.debug(f"Document text: {text}")

# Initialize FAISS database
try:
    db = FAISS.from_documents(data, embeddings)
    st.write("FAISS database created successfully")
except ValueError as e:
    logging.error(f"Error occurred: {e}")
    for text in texts:
        try:
            embedding = embeddings.embed_documents([text])
            logging.debug(f"Embedding: {embedding}")
        except ValueError as ve:
            logging.error(f"Failed to embed document: {text} with error: {ve}")

def get_input():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_input()
submit = st.button("Find Similar Things")

if submit:
    docs = db.similarity_search(user_input)
    print(docs)
    st.subheader("Top Matches:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
