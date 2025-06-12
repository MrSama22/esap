# ======================================================================================
# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# ======================================================================================

import streamlit as st
import os
import sqlite3
from dotenv import load_dotenv

# --- LIBRERÍAS REQUERIDAS ---
from google.cloud import texttospeech
from google.oauth2 import service_account
from google.api_core import exceptions as google_exceptions
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langdetect import detect, LangDetectException

# --- NUEVOS IMPORTS PARA EL RETRIEVER DE COMPRESIÓN ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ... (El resto de tu configuración CONFIG y LANG_CONFIG se mantiene igual) ...
# --- CONFIGURACIÓN ---
CONFIG = {
    "PAGE_TITLE": "Asistente CSD",
    "PAGE_ICON": "🎓",
    "HEADER_IMAGE": "logo1.png",
    "APP_TITLE": "🎓 Asistente Virtual del Colegio Santo Domingo",
    "APP_SUBHEADER": "¡Hola! Estoy aquí para responder tus preguntas basándome en el documento oficial.",
    "WELCOME_MESSAGE": "¡Hola! Soy el asistente virtual del CSD. ¿En qué puedo ayudarte? / Hello! I'm the CSD virtual assistant. How can I help you?",
    "SPINNER_MESSAGE": "Buscando y preparando tu respuesta...",
    "PDF_DOCUMENT_PATH": "documento.pdf",
    "OFFICIAL_WEBSITE_URL": "https://colegiosantodomingo.edu.co/",
    "WEBSITE_LINK_TEXT": "Visita la página web oficial",
    "CSS_FILE_PATH": "styles.css"
}

# --- NUEVO: CONFIGURACIÓN MULTILINGÜE ---
LANG_CONFIG = {
    "es": {
        "tts_voice": {"language_code": "es-US", "name": "es-US-Standard-B"},
        "prompt_template": """
            Eres un asistente virtual amigable y servicial del Colegio Santo Domingo Bilingüe.
            Tu objetivo es responder las preguntas de los usuarios de forma natural y conversacional, basando tus respuestas estricta y únicamente en el contexto proporcionado.
            Usa un tono amable y directo. Si la información está en el contexto, preséntala claramente.
            Si la respuesta no se encuentra en el contexto, indica amablemente que no tienes esa información específica en tus documentos.
            
            Contexto:
            <context>{context}</context>
            
            Pregunta: {input}
            
            Respuesta:
        """
    },
    "en": {
        "tts_voice": {"language_code": "en-US", "name": "en-US-Wavenet-C"}, # Voz nativa en inglés
        "prompt_template": """
            You are a friendly and helpful virtual assistant for the Santo Domingo Bilingual School.
            Your goal is to answer user questions in a natural, conversational way, basing your answers strictly and solely on the provided context.
            Use a friendly and direct tone. If the information is in the context, present it clearly.
            If the answer is not in the context, kindly indicate that you do not have that specific information in your documents.
            
            Context:
            <context>{context}</context>
            
            Question: {input}
            
            Answer:
        """
    }
}
DEFAULT_LANG = "es" # Idioma por defecto si la detección falla

# --- LÓGICA DE LA APLICACIÓN (el resto del código hasta la inicialización de la IA se mantiene igual)
st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")

def load_local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_local_css(CONFIG["CSS_FILE_PATH"])

@st.cache_resource
def verify_credentials():
    try:
        creds_dict = dict(st.secrets['gcp_service_account'])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        return tts_client
    except Exception as e:
        return None

tts_client = verify_credentials()

with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists(CONFIG["HEADER_IMAGE"]):
        st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.title(CONFIG["APP_TITLE"])
st.write(CONFIG["APP_SUBHEADER"])

def text_to_speech(client, text, voice_params):
    if not client: return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_params["language_code"],
            name=voice_params["name"]
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Error al generar el audio: {e}", icon="🚨")
        return None

# --- MODIFICACIÓN CLAVE: Se implementa el ContextualCompressionRetriever para máxima precisión ---
@st.cache_resource
def initialize_rag_components():
    load_dotenv()
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key: st.stop("Error: GOOGLE_API_KEY no está configurada.")
    if not os.path.exists(CONFIG["PDF_DOCUMENT_PATH"]): st.stop(f"Error: No se encontró el PDF.")

    loader = PyPDFLoader(CONFIG["PDF_DOCUMENT_PATH"])
    docs = loader.load()
    
    # Ajuste en el chunking para mejorar la cohesión del texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    
    # 1. Creamos un retriever base que busca más documentos (ej. 10)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. Creamos un "compresor" que usará el LLM para extraer la información relevante
    document_compressor = LLMChainExtractor.from_llm(llm)
    
    # 3. Creamos el retriever de compresión contextual
    # Este retriever primero llamará al 'base_retriever' y luego pasará los resultados al 'document_compressor'
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor, 
        base_retriever=base_retriever
    )
    
    # Devuelve el retriever "inteligente" y el LLM para el paso de respuesta final
    return compression_retriever, llm

# --- INICIALIZACIÓN DE LA IA ---
try:
    retriever, llm = initialize_rag_components() 
except Exception as e:
    st.error(f"Ocurrió un error crítico al inicializar la IA: {e}", icon="🚨")
    st.stop()

# --- LÓGICA DEL CHAT (Sin cambios en esta sección, funcionará con el nuevo retriever) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": CONFIG["WELCOME_MESSAGE"]}]

chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí... / Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(CONFIG["SPINNER_MESSAGE"]):
                try:
                    lang_code = detect(prompt)
                    if lang_code not in LANG_CONFIG:
                        lang_code = DEFAULT_LANG
                except LangDetectException:
                    lang_code = DEFAULT_LANG

                selected_lang_config = LANG_CONFIG[lang_code]
                prompt_template_str = selected_lang_config["prompt_template"]
                tts_voice_params = selected_lang_config["tts_voice"]

                prompt_obj = ChatPromptTemplate.from_template(prompt_template_str)
                document_chain = create_stuff_documents_chain(llm, prompt_obj)
                
                # La cadena ahora usará nuestro retriever de compresión contextual, mucho más preciso
                rag_chain = create_retrieval_chain(retriever, document_chain)

                response = rag_chain.invoke({"input": prompt})
                respuesta_ia = response["answer"]
                st.markdown(respuesta_ia)
                
                audio_content = text_to_speech(tts_client, respuesta_ia, tts_voice_params)
                if audio_content:
                    st.audio(audio_content, autoplay=True)
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})

# --- ENLACE FINAL (Más discreto) ---
st.divider()
st.caption(f"Para más información, puedes visitar la [{CONFIG['WEBSITE_LINK_TEXT']}]({CONFIG['OFFICIAL_WEBSITE_URL']}).")
