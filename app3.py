# Standard imports
import os

# Streamlit imports
import streamlit as st

# Langchain imports
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Import chroma as the vector store 
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set API key for OpenAI Service from environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('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')

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

@st.cache_resource
def load_pdf(): 
    pdf_name = 'annualreportshort.pdf'
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding=OpenAIEmbeddings, 
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

index = load_pdf()

# Create and load PDF Loader
loader = PyPDFLoader('annualreportshort.pdf')
pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='trialdata')

vectorstore_info = VectorStoreInfo(
    name="trialdata",
    description="trialpropertydata",
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

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
