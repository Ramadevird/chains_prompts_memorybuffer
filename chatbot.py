##conversational Q&A chatbot
import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os
st.set_page_config(page_title="Conversational Q&A chatbot")
st.header("Hey! lets chat.....")
chat=ChatOpenAI(openai_api_key=os.getenv("OPENAI_API"),temperature=0.5)
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[SystemMessage(content="you are an skincare assistant")]
def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input=st.text_input("Input:",key="input")
response=get_chatmodel_response(input)

submit=st.button("Provide the question")
if submit:
    st.subheader("The response is:")
    st.write(response)
