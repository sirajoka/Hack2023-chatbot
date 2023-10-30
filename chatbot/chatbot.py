
import os
import streamlit as st
import pinecone
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Pinecone
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.chains import ConversationChain

# Set your API keys
HUGGINGFACE_API_TOKEN = st.secrets["HUGGINGFACE_API_TOKEN"]
os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = "7440d145-170c-4b35-9448-249e92d4dc94"
PINECONE_ENV = "gcp-starter"
PINECONE_INDEX = "langchain-retrieval"

class DecathlonChatbot:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def get_db_decathlon(self):
        pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        index_name = PINECONE_INDEX
        db = Pinecone.from_existing_index(index_name, self.embeddings)
        return db
    @st.cache_data(show_spinner=False)
    def get_response_from_query(db, query, k=4):
        docs = db.similarity_search(query, k=k)
        docs_page_content = " ".join([d.page_content for d in docs])

        # Create your chat model (HuggingFaceHub or other) here
        repo_id = "tiiuae/falcon-7b-instruct"
        chat = HuggingFaceHub(
            huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
            repo_id=repo_id,
            model_kwargs={"temperature": 0, "max_new_tokens": 100},
    )

        # Define system and human message prompts
        system_message_template = """
            You are a friendly E-commerce AI assistant named Kamal designed to answer customer queries on an e-commerce platform that sells products in Amazon retail store {docs}.
            Kamal functions as a chatbot and responds to user phrases like "Thank you", "Hello", etc.
            If the customer shares their name with Kamal, Kamal will remember it politely.
            First, Kamal will classify the sentiment of the customer's question or statement.
            Kamal will use only the given information to answer the question, considering the sentiment of the customer's input.
            If the customer's input contains a typo, Kamal will politely ask for a correction before proceeding to research an answer.
            If Kamal lacks the necessary information or can't find a suitable answer, Kamal will respond with "I'm sorry" and ask follow-up questions.
            If the customer's input isn't a question, Kamal will act as a chatbot assisting customers with brief answers.

            Please go ahead with your query or statement related to Amazon Retail store or any other greetings, and Kamal will respond accordingly!
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_message_template
        )

        human_question_template = "Please respond accordingly to the customer's question: {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_question_template
        )

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # Initialize the conversation buffer memory
        memory = ChatMessageHistory()
        conversation_buf = ConversationChain(llm=chat, memory=memory, prompt=chat_prompt)

        try:
            response = conversation_buf.predict(question=query, docs=docs_page_content)
            return response
        except:
            return None

if __name__ == "__main__":
    chatbot = DecathlonChatbot()
    db = chatbot.get_db_decathlon()
    query = "Your user query here"  # Replace with the actual user query
    response = get_response_from_query(db, query)
    print(response)
