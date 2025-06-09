import streamlit as st
import os
from dotenv import load_dotenv

# --- LIBRERÍAS REQUERIDAS ---
# Asegúrate de haber instalado:
# pip install streamlit dotenv google-cloud-texttospeech langchain-google-genai langchain-community pypdf chromadb
from google.cloud import texttospeech
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ======================================================================================
# --- CONFIGURACIÓN FÁCIL ---
# ¡Modifica los valores en esta sección para cambiar la apariencia de tu app!
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
    
    # Cambia este valor para elegir otra voz. Ejemplos en español:
    # 'es-US-Standard-B' (Masculina), 'es-US-Wavenet-A' (Femenina, calidad premium)
    "TTS_VOICE_NAME": "es-US-Standard-B",
    
    # Ruta al archivo de estilos
    "CSS_FILE_PATH": "styles.css"
}

# ======================================================================================
# --- LÓGICA DE LA APLICACIÓN (Normalmente no necesitas tocar esto) ---
# ======================================================================================

# --- FUNCIÓN para Cargar CSS Externo ---
def load_local_css(file_name):
    """Carga un archivo CSS local y lo inyecta en la app."""
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Advertencia: No se encontró el archivo de estilos: {file_name}")

# --- Configuración de la Página y Carga de CSS ---
st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")
load_local_css(CONFIG["CSS_FILE_PATH"])

# --- Header Personalizado ---
with st.container():
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    if os.path.exists(CONFIG["HEADER_IMAGE"]):
        st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Título y Subtítulo ---
st.title(CONFIG["APP_TITLE"])
st.write(CONFIG["APP_SUBHEADER"])

# --- FUNCIÓN TEXT-TO-SPEECH ---
def text_to_speech(text, voice_name):
    """Convierte texto a audio usando Google Cloud TTS."""
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="es-US", name=voice_name)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.error(f"Error al generar el audio: {e}")
        return None

# --- Función para Cargar la Cadena RAG (con caché) ---
@st.cache_resource
def load_rag_chain():
    """Carga y configura la cadena de IA."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Error: GOOGLE_API_KEY no está configurada. Añádela a tus 'Secrets' si despliegas online.")
        st.stop()
    if not os.path.exists(CONFIG["PDF_DOCUMENT_PATH"]):
        st.error(f"Error: No se encontró el documento PDF: {CONFIG['PDF_DOCUMENT_PATH']}")
        st.stop()
        
    loader = PyPDFLoader(CONFIG["PDF_DOCUMENT_PATH"])
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt_template = "Contexto: <context>{context}</context>\nPregunta: {input}\nRespuesta:"
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
    return rag_chain

# Carga principal de la IA
try:
    rag_chain = load_rag_chain()
except Exception as e:
    st.error(f"Ocurrió un error crítico al inicializar la IA: {e}")
    st.stop()

# --- Lógica del Chat ---
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
            
            # Generar y reproducir audio
            audio_content = text_to_speech(respuesta_ia, CONFIG["TTS_VOICE_NAME"])
            if audio_content:
                st.audio(audio_content, autoplay=True)
            
            st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})

# --- Enlace al Sitio Web Oficial ---
st.divider()
st.markdown(
    f"<div class='footer-link'><a href='{CONFIG['OFFICIAL_WEBSITE_URL']}' target='_blank'>{CONFIG['WEBSITE_LINK_TEXT']}</a></div>",
    unsafe_allow_html=True
)
