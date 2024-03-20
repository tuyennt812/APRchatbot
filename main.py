import streamlit as st
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
import qdrant_client
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import os
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.messages import AIMessage, HumanMessage


def get_vector_store():
    
    client = QdrantClient(
        os.getenv("QDRANT_HOST"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    embeddings = HuggingFaceInstructEmbeddings()

    vector_store = Qdrant(
        client=client, 
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
        embeddings=embeddings,
    )
    
    return vector_store



def main():
    load_dotenv()
    
    st.set_page_config(page_title="Ask APR")
    #st.markdown(footer,unsafe_allow_html=True)
    st.header("APR InsightBot")
        
    # create vector store
    vector_store = get_vector_store()
    
    # create chain 
    qa = RetrievalQA.from_chain_type(
        llm = ChatOpenAI()
        #llm= HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_length=128, temperature=0.5),
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    with st.chat_message("AI"):
        #st.write_stream(response_generator())
        st.write("Hello! I'm your friendly WFP's Annual Performance Report bot! How may I assist you today? Don't hesitate to let me know how I can be of help! 👋")  
    
    #Create chat history and make sure that streamlit won't re run chat history by using session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []    
    
    # show user input
    user_query = st.chat_input("Type your question here ...")
    
    #If user_query is not none and empty, we will write human message as user_query
    if user_query is not None and user_query != "":
    # The chat bot automatically return the Response (based on get_response funcion)
        response = qa.run(user_query)
    #Add user questions the chat_history, whenever users ask questions, it will be added to chat history
        st.session_state.chat_history.append(HumanMessage(content=user_query))
    #Add respons the chat_history, whenever AI answers question, it will be added to chat history
        st.session_state.chat_history.append(AIMessage(content=response))
    
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):  
                with st.chat_message("Human"):  
                    st.write(message.content)
    

if __name__ == '__main__':
    main()