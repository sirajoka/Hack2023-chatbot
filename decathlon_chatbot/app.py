import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from chatbot import DecathlonChatbot

st.set_page_config(page_title="AI Customer Assistant", page_icon='💬')

with st.sidebar:
    st.image("https://commons.wikimedia.org/wiki/File:Hackathon-llm-2023.png")
    st.markdown('# 💬 Menu 🎿')
    st.markdown('''
    
    ### 🔮 Purpose

    This AI assisstant is designed to improve customer experience related to searching within an E-Commerce website.

    We believe this AI assisstant can help many customers worldwide.

    🏀🏓🏈🎳⚾🏒🥊⛳🤿🏏🎾🎿🏐⛸️🤖

    This application intends to respond questions from customers based in this dataset.

    ###
    ''')

    st.markdown('💻 Source code on [Github]')
    st.markdown('👨‍💻 Made by')
    
    
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["👋🏻 Hello! This is your AI assisstant, how can I help you ?"]

if 'past' not in st.session_state:
    st.session_state['past'] = ['GOOD BYE !']

# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()


def get_text():
    question = st.text_input("Vous: ", "", key="input")
    return question


with input_container:
    st.markdown("💬 Hi what is your question ?")
    user_input = get_text()


def generate_response(prompt):
    chatbot = DecathlonChatbot()
    db = chatbot.get_db_decathlon()
    response = chatbot.get_response_from_query(db, prompt)
    return response

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i), avatar_style="bottts-neutral", seed=90)
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars-neutral", seed=10)


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
