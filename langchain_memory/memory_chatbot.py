from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(dotenv_path="/Users/mukulagarwal/Desktop/Projects/langchain/.env",override=True)

model = ChatGroq(model="llama-3.1-8b-instant",max_tokens=100)

store = {}

def get_session_id(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        
    return store[session_id]

prompt = ChatPromptTemplate([
    ("system","You are a Helpfull assistant"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{query}")
])

def trimmer(input_dict):
    history = input_dict['history']
    query = input_dict['query']
    
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

trimmer_run = RunnableLambda(trimmer)

chain = trimmer_run | prompt | model

chain_with_memory = RunnableWithMessageHistory(chain,get_session_history = get_session_id,input_messages_key="query",history_messages_key="history")

while True:
    query = input("enter your messages: ")
    if query == "exit":
        break
    
    response = chain_with_memory.invoke({"query":query},config = {"configurable" : {"session_id" : "firstchat"}})
    print("response: ",response.content)
    print("***********************************************************************")
