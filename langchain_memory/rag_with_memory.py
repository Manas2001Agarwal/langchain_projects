import os
import sys
from uuid import uuid4
from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory 
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import trim_messages
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv(dotenv_path="/Users/mukulagarwal/Desktop/Projects/langchain/.env",override=True)

history = {}

def model_loader():
    groq_model = ChatGroq(model = "llama-3.1-8b-instant", temperature = 0,
                          max_tokens=400)
    
    return groq_model

def document_loader(path:str):
    loader = DirectoryLoader(path=path,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)   # type: ignore
    docs = loader.load()
    return docs

def text_splitter(docs:List[Document]) -> List[Document]:
    text_spl = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_spl.split_documents(docs)
    return docs

def get_embedding_model():

    embed_model = GoogleGenerativeAIEmbeddings(
        model = "models/text-embedding-004"
    )
    return embed_model

def load_documents(index_name:str, docs:List[Document]) -> None:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embed_model = get_embedding_model()
    
    if pc.has_index(index_name):
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index,
                                           embedding=embed_model)
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        print("documents loaded")
        return
    else:
        pc.create_index(name = index_name,
                        metric="cosine",
                        dimension=768,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        index = pc.Index(index_name)
        vector_store = PineconeVectorStore(index=index,
                                           embedding=embed_model)
        uuids = [str(uuid4()) for _ in range(len(docs))]
        vector_store.add_documents(documents=docs, ids=uuids)
        print("index-created and document loaded")
        return
        
def context_retriver(index_name):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    embed_model = get_embedding_model()
    vector_store = PineconeVectorStore(index=index,
                                           embedding=embed_model)
    retriever = vector_store.as_retriever(search_kwargs = {"k":5})
    return retriever

def get_session_history(session_id:str) -> BaseChatMessageHistory:
    if session_id not in history:
        history[session_id] = InMemoryChatMessageHistory()
    return history[session_id]

def trimmer(input_dict:dict):
    history = input_dict['history']
    query = input_dict['query']
    
    model = model_loader()
    
    trimmed_message = trim_messages(
        messages = history,
        max_tokens = 300,
        token_counter = model,
        strategy = "last",
        allow_partial = False,
        end_on = "ai",
        start_on = "human"
    )
    return {"history":trimmed_message,"query":query}
    
if __name__ == "__main__":
    index_name = "rag-chatbot"
    llm = model_loader()
    
    # docs = document_loader("/Users/mukulagarwal/Desktop/Projects/langchain/RAG")
    # docs = text_splitter(docs)
    # print(len(docs))
    # print(docs[5].page_content)
    # load_documents(index_name,docs)
    
    retriver = context_retriver(index_name=index_name)
    trimm_runn = RunnableLambda(trimmer)
    
    parallel_chain = RunnableParallel({
        "query": RunnableLambda(lambda x: x["query"]),  # type: ignore
        "history": RunnableLambda(lambda x: x["history"]), # type: ignore
        "context": RunnableLambda(lambda x: x["query"]) | retriver # type: ignore
    })

    template = """Based on the given context:\n{context}\n
    This is user chat_history: {history}\n
    Answer the following using query:\n{query}\n\n
    Answer based on given context and chat_history only.
    Do not assume or add anything on your own. Include your answer only in output. 
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query","context","history"]
    )
    
    parser = StrOutputParser()
    
    retriver_chain = trimm_runn | parallel_chain | prompt | llm 
    retriver_chain_with_memory = RunnableWithMessageHistory(runnable = retriver_chain,
                                                            get_session_history=get_session_history,
                                                            history_messages_key="history",
                                                            input_messages_key="query")
    
    while True:
        query = input("enter your messages: ")
        if query == "exit":
            break
        
        response = retriver_chain_with_memory.invoke({"query":query},config={"configurable":{"session_id":"firstchat"}})
        print("response: ",response.content)
        print("***********************************************************************")
        print("***********************************************************************")
print(history)