import streamlit as st

st.title('🎈 Hukuk LLM')

st.write('Hello world!')

import os
import openai
import sys
import datetime

from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import chainlit as cl

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig

from dotenv import load_dotenv
load_dotenv()

#API KEY
api_key = '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' #Lawyer KEy
os.environ["OPENAI_API_KEY"] = api_key

@cl.on_chat_start
async def on_chat_start():

    await cl.Message(content="Plus Lawyer danışmanlık hizmetine hoşgeldiniz! Sorularınızı cevaplamak için buradayım :)").send()

    ## Model initilization
    llm_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=llm_name, temperature=0, streaming =True)

    # Template creation
    template = """Use the following pieces of context to answer the question at the end. Answer in Turkish. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Use five sentences maximum.
     Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Vector Database mounting
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    persist_directory = 'docs/chroma/'
    #vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    index_name = "llm-hukuk"
    vectordb = Pinecone.from_existing_index(index_name, embeddings)

    # Chain initilization
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),  # search_type='mmr'
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    cl.user_session.set("chain",qa_chain)
    """
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable legal advisor answering questions based on the documents.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    """

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler() #stream_final_answer=True,answer_prefix_tokens=["FINAL", "ANSWER"]
    #cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]

    """
    source_documents = res["source_documents"]
    from typing import List
    text_elements = []  # type: List[cl.Text]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    """
    await cl.Message(content=answer).send() #, elements=text_elements
    #chainlit run app.py -w

