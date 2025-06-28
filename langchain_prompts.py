## Prompts are very important. The output of LLM changes depending on the prompt 
## that goes in the model

## Static Prompts vs Dynamic Prompt

from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

prompt_template = load_prompt('template.json')

model  = ChatGroq(model = "llama-3.1-8b-instant",temperature=0.5,max_retries=2)

st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

chain = prompt_template | model

prompt_template.save("template.json")

if st.button("Summarize"):
    result = result = chain.invoke({    
        'paper_input' : paper_input,
        'style_input' : style_input,
        'length_input' : length_input
    })
    st.write(result.content)