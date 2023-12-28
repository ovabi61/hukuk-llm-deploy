import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

def generate_response( openai_api_key, query_text):
    # API KEY

    import pinecone

    pinecone.init(
        api_key="5f0d4072-6003-4bca-b183-1d35f312c804",
        environment="gcp-starter",
    )

    ## Model initilization
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0)

    # Template creation
    template = """Use the following pieces of context to answer the question at the end. Answer in Turkish. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Use five sentences maximum.
     Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Vector Database mounting
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    persist_directory = 'docs/chroma/'
    # vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    index_name = "llm-hukuk"
    vectordb = Pinecone.from_existing_index(index_name, embeddings)

    # Chain initilization
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),  # search_type='mmr'
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain({"query": query_text})

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
#uploaded_file = st.file_uploader('Upload an article', type='txt')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.')#, disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (query_text))
    submitted = st.form_submit_button('Submit', disabled=not(query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
