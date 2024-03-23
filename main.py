import streamlit as st
import os
from dotenv import load_dotenv
from streamlit_option_menu import option_menu
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI,OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient


def main():
    
    selected = option_menu(
       menu_title = "",
       options = ["Home","APR Content","APR Annexes",],
       icons = ["house",'book','envelope'],
       menu_icon = 'cast',
       default_index = 0,
       orientation = "horizontal"
   )
    
##-------------------------------------Homepage-------------------------------------
    if selected == "Home":
        st.markdown("<h1 style='text-align: center; font-size: 30px;'>APRInsight Bot</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 16px;'>Hello! As an AI chat bot developed by the APR team of the World Food Programme, I'm here to guide you through the exciting world of the 2022 WFP's Annual Performance Report</p>", unsafe_allow_html=True)
        
        row1 = st.columns(2)
        row2 = st.columns(2)
            

                
        with row1[0].container(height=120):
            st.markdown("<h3 style='font-size: 14px;'>Unlock Insights with APR Conversations</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 14px;'>Engage with APR's main documents for instant information retrieval</p>", unsafe_allow_html=True)

        with row1[1].container(height=120):
            st.markdown("<h3 style='font-size: 14px;'>Explore WPF's Performance</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 14px;'>Discover insights on Program Performance, Management, Funding, and Expenditure</p>", unsafe_allow_html=True)

        with row2[0].container(height=120):
            st.markdown("<h3 style='font-size: 14px;'>Discover APR Annexes</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 14px;'>Access detailed information from APR's Annexes table effortlessly</p>", unsafe_allow_html=True)

        with row2[1].container(height=120):
            st.markdown("<h3 style='font-size: 14px;'>Top Indicators</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size: 14px;'>Uncover WFP's output, outcome, cross-cutting indicator performance</p>", unsafe_allow_html=True)
    
    
    
    
##-------------------------------------APR Annexes-------------------------------------    
    if selected == "APR Annexes":            
            with st.chat_message("AI"):
                #st.write_stream(response_generator())
                st.write("Welcome! Ready to unlock the insights of APR annexes? Let's dive in together! How can I assist you today?")  
                
            #Create chat history and make sure that streamlit won't re run chat history by using session_state
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []    
                
            user_query = st.chat_input("Type your question here ...")
            if user_query is not None and user_query != "":
            
                db = ["Crosscutting_Indicator_Annex.csv", "Output_Indicator_Annex.csv"]
                
                agent = create_csv_agent(
                    ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
                    db,
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
            )
            
                response = agent.invoke(user_query)
                
                #Add user questions the chat_history, whenever users ask questions, it will be added to chat history
                st.session_state.chat_history.append(HumanMessage(content=user_query))
                #Add respons the chat_history, whenever AI answers question, it will be added to chat history
                st.session_state.chat_history.append(AIMessage(content=response['output']))
                
                for message in st.session_state.chat_history:
                    if isinstance(message, AIMessage):
                        with st.chat_message("AI"):
                            st.write(message.content)
                    elif isinstance(message, HumanMessage):  
                        with st.chat_message("Human"):  
                            st.write(message.content)
                            
                            
                            
##-------------------------------------APR Content-------------------------------------                       
    if selected == "APR Content":
        def get_vector_store():
    
            client = QdrantClient(
                os.getenv("QDRANT_HOST"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            
            embeddings = OpenAIEmbeddings()

            vector_store = Qdrant(
                client=client, 
                collection_name=os.getenv("QDRANT_COLLECTION_NAME"), 
                embeddings=embeddings,
            )
            
            return vector_store
            
        # create vector store
        vector_store = get_vector_store()
        
        # create chain 
        qa = RetrievalQA.from_chain_type(
            llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k'),
            #llm= HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", max_length=128, temperature=0.5),
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        with st.chat_message("AI"):
            #st.write_stream(response_generator())
            st.write("Hey there! Excited to explore APR content with you! What can I do to make your discovery journey even more enjoyable?")  
        
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