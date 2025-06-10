# ======================================================================================
# --- PARCHE PARA SQLITE3 EN STREAMLIT CLOUD ---
# Este bloque de código DEBE SER LO PRIMERO en todo el script.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass # Si pysqlite3-binary no está instalado, simplemente continúa.
# ======================================================================================

import streamlit as st
import os
import sqlite3 # Importamos para verificar la versión
from dotenv import load_dotenv

# --- LIBRERÍAS REQUERIDAS ---
from google.cloud import texttospeech
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ======================================================================================
# --- CONFIGURACIÓN FÁCIL ---
# ======================================================================================
CONFIG = {
    "PAGE_TITLE": "Asistente CSD",
    "PAGE_ICON": "🎓",
    "HEADER_IMAGE": "logo1.png",
    "APP_TITLE": "🎓 Asistente Virtual del Colegio Santo Domingo",
    "APP_SUBHEADER": "¡Hola! Estoy aquí para responder tus preguntas basándome en el documento oficial.",
    "WELCOME_MESSAGE": "¡Hola! Soy el asistente virtual del CSD. ¿En qué puedo ayudarte?",
    "SPINNER_MESSAGE": "Buscando y preparando tu respuesta...",
    "PDF_DOCUMENT_PATH": "documento.pdf",
    "OFFICIAL_WEBSITE_URL": "https://colegiosantodomingo.edu.co/",
    "WEBSITE_LINK_TEXT": "Visita la Página Web Oficial del Colegio",
    "TTS_VOICE_NAME": "es-US-Standard-B",
    "CSS_FILE_PATH": "styles.css"
}

# ======================================================================================
# --- LÓGICA DE LA APLICACIÓN ---
# ======================================================================================

# --- FUNCIÓN para Cargar CSS Externo ---
def load_local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Configuración de la Página y Carga de CSS ---
st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")
load_local_css(CONFIG["CSS_FILE_PATH"])

# --- DIAGNÓSTICO DE CREDENCIALES ---
st.sidebar.title("Estado del Sistema")
st.sidebar.info(f"Versión de SQLite3: **{sqlite3.sqlite_version}**")
if st.secrets.get("GOOGLE_API_KEY"):
    st.sidebar.success("✔️ Secret de Gemini encontrado.")
else:
    st.sidebar.error("❌ Secret de Gemini NO encontrado.")

if st.secrets.get("gcp_service_account"):
    st.sidebar.success("✔️ Sección [gcp_service_account] encontrada.")
    if st.secrets["gcp_service_account"].get("private_key"):
        st.sidebar.success("✔️ 'private_key' encontrada dentro de la sección.")
    else:
        st.sidebar.error("❌ 'private_key' NO encontrada. Revisa el formato.")
    if st.secrets["gcp_service_account"].get("client_email"):
        st.sidebar.success("✔️ 'client_email' encontrado.")
    else:
        st.sidebar.error("❌ 'client_email' NO encontrado.")
else:
    st.sidebar.error("❌ Sección [gcp_service_account] NO encontrada. Revisa el nombre.")

# --- UI (Header, Título) ---
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists(CONFIG["HEADER_IMAGE"]):
        st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.title(CONFIG["APP_TITLE"])
st.write(CONFIG["APP_SUBHEADER"])

# --- FUNCIONES CORE (TTS y RAG) ---
@st.cache_resource
def get_tts_client():
    try:
        if 'gcp_service_account' in st.secrets:
            creds_dict = dict(st.secrets['gcp_service_account'])
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            return texttospeech.TextToSpeechClient(credentials=credentials)
        else: # Para desarrollo local
            return texttospeech.TextToSpeechClient()
    except Exception as e:
        st.error(f"Error al crear el cliente TTS: {e}", icon="�")
        return None

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
    prompt_template = "Contexto: <context>{context}</context>\nPregunta: {input}\nRespuesta:"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

# --- INICIALIZACIÓN DE LA APP ---
tts_client = get_tts_client()
rag_chain = load_rag_chain()

if not rag_chain: 
    st.error("La inicialización de la cadena de IA ha fallado. Revisa los errores de arriba.", icon="🚨")
    st.stop()

# --- LÓGICA DEL CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": CONFIG["WELCOME_MESSAGE"]}]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner(CONFIG["SPINNER_MESSAGE"]):
            response = rag_chain.invoke({"input": prompt})
            respuesta_ia = response["answer"]
            st.markdown(respuesta_ia)
            audio_content = text_to_speech(tts_client, respuesta_ia, CONFIG["TTS_VOICE_NAME"])
            if audio_content: st.audio(audio_content, autoplay=True)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})

# --- ENLACE FINAL ---
st.divider()
st.markdown(f"<div class='footer-link'><a href='{CONFIG['OFFICIAL_WEBSITE_URL']}' target='_blank'>{CONFIG['WEBSITE_LINK_TEXT']}</a></div>", unsafe_allow_html=True)
�
