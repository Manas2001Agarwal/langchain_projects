import os
import sys
from uuid import uuid4
from typing import List
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

load_dotenv(override=True)

def model_loader():
    groq_model = ChatGroq(model = "llama-3.1-8b-instant", temperature = 0,
                          max_tokens=200)
    
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
    
if __name__ == "__main__":
    query = "What are Retrieval Augmented Generation system"
    index_name = "rag-chatbot"
    llm = model_loader()
    
    # docs = document_loader("/Users/mukulagarwal/Desktop/Projects/langchain/RAG")
    # docs = text_splitter(docs)
    # print(len(docs))
    # print(docs[5].page_content)
    # load_documents(index_name,docs)
    
    retriver = context_retriver(index_name=index_name)
    
    parallel_chain = RunnableParallel({
        "query":RunnablePassthrough(),
        "context": retriver
    })

    template = """Based on the given context:\n{context}\n
    Answer the following using query:\n{query}\n\n
    Make sure to answer based on given context/info only.
    Do not assume or add anything on your own. Cite the references after
    answer
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query","context"]
    )
    
    parser = StrOutputParser()
    
    retriver_chain = parallel_chain | prompt | llm |parser
    answer = retriver_chain.invoke(query)
    print(answer)