import os
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_groq import ChatGroq 
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from data_schemas import EvaluateContextPrecision

load_dotenv(override=True)

def model_loader():
    groq_model = ChatGroq(model = "llama-3.1-8b-instant", temperature = 0,
                          max_tokens=200)
    
    return groq_model


def get_embedding_model():

    embed_model = GoogleGenerativeAIEmbeddings(
        model = "models/text-embedding-004"
    )
    return embed_model
        
def context_retriver(index_name):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    embed_model = get_embedding_model()
    vector_store = PineconeVectorStore(index=index,
                                           embedding=embed_model)
    retriever = vector_store.as_retriever(search_kwargs = {"k":5})
    return retriever
    
def retrieval_pipeline(index_name, query):
    llm = model_loader()
    
    retriver = context_retriver(index_name=index_name)
    
    parallel_chain = RunnableParallel({
        "query":RunnablePassthrough(),
        "context": retriver
    })

    template = """Based on the given context:\n{context}\n
    Answer the following using query:\n{query}\n\n
    Make sure to answer based on given context only. Do not assume or add anything on your own. 
    No need to cite references to answer
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["query","context"]
    )
    
    parser = StrOutputParser()
    
    retrieved_context = parallel_chain.invoke(query)
    retriver_chain = prompt | llm |parser
    answer = retriver_chain.invoke(retrieved_context)
    return retrieved_context['context'],answer
    
def test_context_relevance(context,query):
    
    template = """ You are given a retrieved context and user question. You have to determine
    if the context retrieved is relevant and can be used to answer the user's query. If the user
    query and retrieved context carry the same keywords or have the similar semantic meaning then
    contrieved is relevant, otherwise not. Give your answer as 'Yes' or 'No'. Also compute the 
    consine similarity score and BM25 score between the query and context retrieved. In addition 
    to this give explanation for the your output\n
    retrived context: {context}\n
    query: {query}  
    """
    
    parser = PydanticOutputParser(pydantic_object=EvaluateContextPrecision)
    
    prompt = ChatPromptTemplate([
        ("system","You are RAG system evaluator. Make sure you follow the formatting instructions while giving output {format_instructions}"),
        ("human",template)
    ],
                                input_variables = ["context","query"],
                                partial_variables = {"format_instructions":parser.get_format_instructions()})
    llm = model_loader()
   
    context_precision = prompt | llm | parser
    return context_precision.invoke({"context":context,"query":query})
    
    
if __name__ == "__main__":
    query = "What are Retrieval Augmented Generation system"
    index_name = "rag-chatbot"
    context,answer = retrieval_pipeline(index_name,query)
    context = "\n\n".join([doc.page_content for doc in context])
    context_precision = test_context_relevance(context,query)
    print(context_precision)
    