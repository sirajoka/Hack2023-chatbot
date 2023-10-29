
import os
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub

from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, ConversationChain


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

from getpass import getpass

from langchain.prompts import PromptTemplate


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

        #chat = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

        repo_id = "tiiuae/falcon-7b-instruct"
        chat = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN, 
                            repo_id=repo_id, 
                            model_kwargs={"temperature":0.7, "max_new_tokens":700})


        # Template to use for the system message prompt
        #template = """
        #    You are an assistant designed to answer customer queries on an e-commerce platform that sells sports equipment {docs}.
        #    I also function as a chatbot, responding to user phrases like "Thank you", "Hello", etc.
        #    First, I will classify the sentiment of the customer's question or statement.
        #    I will use only the given information to answer the question, considering the sentiment of the customer's input.
        #    If I lack the necessary information or can't find a suitable answer, I will respond with I'm sorry I do not have this answer."
        #    If the input isn't a question, I will act as a chatbot assisting customers.
        #    My answers will be brief yet detailed.

        #    Please go ahead with your query or statement related to sport equipment or any other greetings, and I will respond accordingly!
        #    """

        template = """
            You are a friendly E-commerce chatbot named Kamal designed to answer customer queries on an e-commerce platform that sells products in amazon retail store {docs}.
            Kamal function as a chatbot, is responding to user phrases like "Thank you", "Hello", etc.
            First, Kamal will classify the sentiment of the customer's question or statement.
            Kamal should ask follow-up questions if the customer's query is too broad, in order to get necessary details to provide a useful response.
            If Kamal lacks the necessary information or can't find a suitable answer, Kamal will respond with I'm sorry I do not have this answer."
            If the input isn't a question, Kamal will act as a chatbot assisting customers.
            Kamal answers will be brief yet detailed.

            Please go ahead with your query or statement related to Amazon Retail store or any other greetings, and I will respond accordingly!
            """


        #template = """
        #You are a friendly E-commerce chatbot named Kamal. You're Kamal wherever you're mentioned.
        #Kamal is a friendly, casual e-commerce chatbot that helps customers find and learn about products in amazon retail store.
        #Kamal should demonstrate knowledge about the store's product inventory and common e-commerce processes. {docs}
        
        #Guidelines for Kamal's Responses:
        #Kamal function as a chatbot, is responding to user phrases like "Thank you", "Hello", etc.
        #Kamal should ask follow-up questions if the customer's query is too broad, in order to get necessary details to provide a useful response.
        #If Kamal does not have enough information or is asked about topics outside of its knowledge base, it should politely defer and ask the customer to rephrase or provide more details on their question or interest area
        
        #Example Dialogues:
        #- User: What laptops do you have?
        #Kamal: Hey there! We have a great selection of laptops. Do you have a specific brand or type in mind?

        #- User: How can I return a product I bought?
        #Kamal: Hi! To return a product, please provide your order number and the reason for the return so I can assist you better.

        #- User: Tell me about your store's operating hours.
        #Kamal: Sure thing! Our store is open from Monday through Friday. What else can I help you with?

        #- User: Can you recommend a good camera for beginners?
        #Kamal: Of course! We have some great options for beginners. What's your budget for the camera?

        #- User: What's the capital of France?
        #Kamal: I'm here to help with e-commerce questions. If you have any shopping-related queries, feel free to ask!
        #"""
        
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)


        # Human question prompt
        human_template = "Please provide the customer's input, and I'll respond accordingly : {question}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        #prompt = PromptTemplate(template=template, input_variables=["question"])
        
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )


        chain = LLMChain(llm=chat, prompt=chat_prompt)
     

        
        # DID NOT REMEMBER chain = LLMChain(llm=chat, memory=buffermemory, prompt=chat_prompt, verbose=True)
        
        # buffermemory = ConversationBufferMemory()      
        # FAILURES SHOWN 
        # with chain = ConversationChain(llm=chat,memory=buffermemory, prompt=chat_prompt, verbose=True)
        # or
        # with
        #chain = ConversationChain( llm=chat,
        #                           verbose=True,
        #                           memory=ConversationBufferMemory()
        #)
 
        #extracted_messages = chain.memory.chat_memory.messages
        #ingest_to_db = messages_to_dict(extracted_messages)
        #retrieve_from_db = json.loads(json.dumps(ingest_to_db))
        #retrieved_messages = messages_from_dict(retrieve_from_db)
        #retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
        #retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history)

        #reloaded_chain = ConversationChain(
        #    llm=chat,
        #    verbose=True,
        #    memory=retrieved_memory
        #)

        try:
            response = chain.run(question=query, docs=docs_page_content)

            return response
        except:
            return None