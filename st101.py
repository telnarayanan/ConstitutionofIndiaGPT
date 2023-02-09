# List of Imports
import config
import os
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA


os.environ["OPENAI_API_KEY"] = st.secrets["api_key"]
with open("COI_English.txt", "r", encoding="utf8") as f:
    constitution_text = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(constitution_text)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)
    #query = "What is this document about? When was this document last updated?"

st.title('DocQuery Sample')
query = st.text_input('Enter your query to be asked to the Indian Constitution', 'What is this document about?')
st.write(qa.run(query))
