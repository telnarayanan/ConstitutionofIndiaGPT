# Include Imports Here
import os
import streamlit as st
from pdfminer.high_level import extract_text
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA



os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

with open('COI_English.txt') as f:
    sebitext = f.read()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(sebitext)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

query = "Which is the version of the document, and when was this updated?"
st.write(qa.run(query))
