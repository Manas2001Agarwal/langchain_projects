from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_model = ChatGroq(model = "llama-3.1-8b-instant")

chat_template = ChatPromptTemplate(
    [
        ('system', 'You are a helpful assistant specializing in {domain_name} domain'),
        ('human','Give me brief explanation of {topic}')
    ]
)

domain_name = input("Enter domain_name: ")
topic = input("Enter topic: ")
    
chat_history = chat_template.invoke({
    'domain_name' : domain_name,
    'topic' : topic
})

chat_history = chat_history.to_messages()

result = groq_model.invoke(chat_history)

print("AI: ",result.content)

chat_history.append(result)

template = ChatPromptTemplate(
    [
        MessagesPlaceholder(variable_name='history'),
        ("human", '{query}')
    ]
)

while(True):
    query = input("You - Enter user query: ")
    if query == "exit":
        break
    input_prompt = template.invoke({
        'history' : chat_history,
        'query' : query
    })
    chat_history.append(HumanMessage(content=query))
    result = groq_model.invoke(input_prompt)
    print("AI: ",result.content)
    chat_history.append(result)

print(chat_history)