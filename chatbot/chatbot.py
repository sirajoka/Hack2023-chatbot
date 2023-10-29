import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import find_dotenv, load_dotenv
import streamlit as st
import pinecone

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = "7440d145-170c-4b35-9448-249e92d4dc94"
PINECONE_ENV = "gcp-starter"
PINECONE_INDEX = "langchain-retrieval"

class DecathlonChatbot:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def get_db_decathlon(_self):
        pinecone.init(
            api_key=PINECONE_API_KEY, 
            environment=PINECONE_ENV  
        )

        index_name=PINECONE_INDEX

        db = Pinecone.from_existing_index(index_name, _self.embeddings)

        return db

    @st.cache_data(show_spinner=False)
    def get_response_from_query(_self, _db, query, k=4):
        """
        Function that generates a response to a customer question using the gpt-3.5-model and the docs provided
        """

        docs = _db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])

        chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        # Template to use for the system message prompt
        template = """
            You are an assistant designed to answer customer queries on an e-commerce platform that sells sports equipment {docs}.
            I also function as a chatbot, responding to user phrases like "Thank you", "Hello", etc.
            First, I will classify the sentiment of the customer's question or statement.
            I will use only the given information to answer the question, considering the sentiment of the customer's input.
            If I lack the necessary information or can't find a suitable answer, I will respond with I'm sorry I do not have this answer."
            If the input isn't a question, I will act as a chatbot assisting customers.
            My answers will be brief yet detailed.

            Please go ahead with your query or statement related to sport equipment or any other greetings, and I will respond accordingly!
            """

        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # Human question prompt
        human_template = "Please provide the customer's input, and I'll respond accordingly : {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        buffermemory = ConversationBufferMemory()
        chain = ConversationChain(llm=chat,memory=buffermemory, prompt=chat_prompt, verbose=True)
        #chain = LLMChain(llm=chat, memory=buffermemory, prompt=chat_prompt, verbose=True)

        try:
            response = chain.run(question=query, docs=docs_page_content)

            return response
        except:
            return None