# ======================================================================================
# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
# codigo funcional 1
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

# --- CONFIGURACIÓN ---
CONFIG = { "PAGE_TITLE": "Asistente CSD", "PAGE_ICON": "🎓", "HEADER_IMAGE": "logo1.png", "APP_TITLE": "🎓 Asistente Virtual del Colegio Santo Domingo", "APP_SUBHEADER": "¡Hola! Estoy aquí para responder tus preguntas basándome en el documento oficial.", "WELCOME_MESSAGE": "¡Hola! Soy el asistente virtual del CSD. ¿En qué puedo ayudarte?", "SPINNER_MESSAGE": "Buscando y preparando tu respuesta...", "PDF_DOCUMENT_PATH": "documento.pdf", "OFFICIAL_WEBSITE_URL": "https://colegiosantodomingo.edu.co/", "WEBSITE_LINK_TEXT": "Visita la página web oficial", "TTS_VOICE_NAME": "es-US-Standard-B", "CSS_FILE_PATH": "styles.css" }

# --- LÓGICA DE LA APLICACIÓN ---
st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")

def load_local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_local_css(CONFIG["CSS_FILE_PATH"])

# --- VERIFICADOR DE CREDENCIALES ---

@st.cache_resource
def verify_credentials():
    #st.sidebar.title("Estado del Sistema")---
   # st.sidebar.info(f"Versión de SQLite3: **{sqlite3.sqlite_version}**")
    #if st.secrets.get("GOOGLE_API_KEY"): st.sidebar.success("✔️ Secret de Gemini encontrado.")
    #else: st.sidebar.error("❌ Secret de Gemini NO encontrado.")
    #if 'gcp_service_account' in st.secrets:
        #st.sidebar.success("✔️ Secrets de TTS encontrados.")
        #try:
    creds_dict = dict(st.secrets['gcp_service_account'])
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
    st.sidebar.success("✔️ Cliente TTS creado con éxito.")
    return tts_client
        #except Exception as e:
            #st.sidebar.error(f"❌ FALLO AL CREAR CREDENCIALES TTS: {e}")
            #return None
    #else:
        #st.sidebar.error("❌ Secrets de TTS NO encontrados.")
        #return None

tts_client = verify_credentials()

# --- UI (Header, Título) ---
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists(CONFIG["HEADER_IMAGE"]):
        st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.title(CONFIG["APP_TITLE"])
st.write(CONFIG["APP_SUBHEADER"])


# --- FUNCIONES CORE (TTS y RAG) ---
def text_to_speech(client, text, voice_name):
    if not client: return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="es-US", name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Error al generar el audio: {e}", icon="🚨")
        return None

@st.cache_resource
def load_rag_chain():
    load_dotenv()
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key: st.stop("Error: GOOGLE_API_KEY no está configurada.")
    if not os.path.exists(CONFIG["PDF_DOCUMENT_PATH"]): st.stop(f"Error: No se encontró el PDF.")
    
    loader = PyPDFLoader(CONFIG["PDF_DOCUMENT_PATH"])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    # --- CAMBIO IMPORTANTE: Nuevo Prompt ---
    prompt_template = """
    Eres un asistente virtual amigable y servicial del Colegio Santo Domingo.
    Tu objetivo es responder las preguntas de los usuarios de forma natural y conversacional, basando tus respuestas estricta y únicamente en el contexto proporcionado.

    En lugar de decir "el texto dice" o "el documento relata", utiliza frases como:
    - "Según la información que tengo..."
    - "En mi base de datos encuentro que..."
    - "Consultando los documentos, veo que..."
    - "De acuerdo a los registros, ..."

    Si la respuesta no se encuentra en el contexto, indica amablemente que no tienes esa información específica en tus documentos.

    Contexto:
    <context>{context}</context>

    Pregunta: {input}

    Respuesta conversacional:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- INICIALIZACIÓN DE LA IA ---
try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Ocurrió un error crítico al inicializar la IA: {e}", icon="🚨")
    st.stop()

if not rag_chain: 
    st.error("La inicialización de la IA ha fallado.", icon="🚨")
    st.stop()

# --- LÓGICA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": CONFIG["WELCOME_MESSAGE"]}]

# --- CAMBIO IMPORTANTE: Layout del Chat y Enlace ---
# El área del chat ocupa el espacio principal
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# El input para escribir se mantiene abajo
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Añadir y mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Generar y mostrar la respuesta de la IA
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(CONFIG["SPINNER_MESSAGE"]):
                response = rag_chain.invoke({"input": prompt})
                respuesta_ia = response["answer"]
                st.markdown(respuesta_ia)
                
                audio_content = text_to_speech(tts_client, respuesta_ia, CONFIG["TTS_VOICE_NAME"])
                if audio_content: 
                    st.audio(audio_content, autoplay=True)
                
                st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})

# --- ENLACE FINAL (Más discreto) ---
st.divider()
st.caption(f"Para más información, puedes visitar la [{CONFIG['WEBSITE_LINK_TEXT']}]({CONFIG['OFFICIAL_WEBSITE_URL']}).")

