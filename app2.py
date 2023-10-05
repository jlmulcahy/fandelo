# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = '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'

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()


@st.cache_resource
def load_pdf(): 
    pdf_name = 'annualreportshort.pdf'
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding = OpenAIEmbeddings, 
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

index = load_pdf()

chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=index.vectorstore.as_retriever(), input_key='question')

st.title('Ask Watsonx 🤖')

if 'messages' not in st.session_state: 
    st.session_state.messages = [] 

for message in st.session_state.messages: 
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})

    response = chain.run(prompt)

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})