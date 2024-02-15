#Integrate our code with openai API
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
os.environ["OPENAI_API_KEY"]=openai_key
#streamlit frame work.
st.title('Celebrity Search Results')
input_text=st.text_input("Search the topic you want")
#memory buffer.
person_memory=ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory=ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory=ConversationBufferMemory(input_key='dob',memory_key='description_history')
##OPENAI LLM'S
llm_model=OpenAI(temperature=0.8)

#prompt templates
first_input_prompt=PromptTemplate(input_variables=['name'],template="Tell me about celebrity {name}")
chain1=LLMChain(llm=llm_model,prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)
second_input_prompt=PromptTemplate(input_variables=['person'],template="when was {person} born")
chain2=LLMChain(llm=llm_model,prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)
third_input_prompt=PromptTemplate(input_variables=['dob'],template="Are there any special events happended in world during {dob}")
chain3=LLMChain(llm=llm_model,prompt=third_input_prompt,verbose=True,output_key='specials',memory=descr_memory)
parent_chain=SequentialChain(chains=[chain1,chain2,chain3],input_variables=['name'],output_variables=['person','dob','specials'],verbose=True)

if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.info(person_memory.buffer)
    with st.expander('specials'):
        st.info(descr_memory.buffer)