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

# --- LIBRERÍAS PARA PROCESAR AUDIO ---
import io
from pydub import AudioSegment

# --- LIBRERÍAS REQUERIDAS ---
from google.cloud import texttospeech
from google.cloud import speech
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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from audio_recorder_streamlit import audio_recorder
#¡Hola! Soy el asistente virtual del CSDB pero me puedes decir Dominguito. / Hello! I'm the CSDB virtual assistant but you can call me Dominguito.
# --- CONFIGURACIÓN ---
CONFIG = {
    "PAGE_TITLE": "CSDB Assistant",
    "PAGE_ICON": "🎓",
    "HEADER_IMAGE": "logos/logo2.png",
    "APP_TITLE": "🎓 Virtual Assistant of the ESAP",
    "APP_SUBHEADER": "Hello ! I am here to answer your questions based on the information",
    "WELCOME_MESSAGE": "¿En qué puedo ayudarte? / How can I help you?",
    "SPINNER_MESSAGE": "Generating response...",
    # !!! CAMBIO AQUÍ: Ahora es la RUTA DE LA CARPETA, no una base de nombre de archivo !!!
    "PDF_DOCUMENT_BASE_PATH": "documentos", # Path to the folder containing all PDF documents
    "MAX_PDF_DOCUMENTS": 100, # This setting is now less relevant as we load all PDFs in the folder
    "OFFICIAL_WEBSITE_URL": "https://www.esap.edu.co/",
    "WEBSITE_LINK_TEXT": "official page",
    "CSS_FILE_PATH": "styles.css",
    # --- NUEVAS CONFIGURACIONES PARA ICONOS PERSONALIZADOS ---
    "ASSISTANT_AVATAR": "assistantPhoto.png",  # Tu imagen del asistente
    "USER_AVATAR": "user_avatar.png"  # Tu imagen del usuario
}

# --- CONFIGURACIÓN MULTILINGÜE ---
LANG_CONFIG = {
    "es": {
        "tts_voice": {"language_code": "es-US", "name": "es-US-Standard-B"},
        "prompt_template": """
            Eres un asistente experto del La universidad de la ESAP. Tu única función es responder preguntas basándote en el contenido de los documentos que se te proporcionan en el 'Contexto'.
            Instrucciones Críticas:
            1. Búsqueda Exhaustiva: Antes de responder, revisa CUIDADOSAMENTE y de forma COMPLETA todo el 'Contexto'. La respuesta SIEMPRE estará en ese texto.
            2. Respuesta: Si encuentras la respuesta, preséntala de manera clara y concisa, usa siempre el documento mas actualizado a la fecha actual (2025).
            3. Manejo de Incertidumbre: Solo si después de una búsqueda exhaustiva no encuentras una respuesta, indica amablemente que no tienes la información específica.
            4. Manejo de fuentes: al terminar de dar la respuessta , dile al usuario de donde que documentos sacaste la informacion y el numero de pagina.
            Contexto: <context>{context}</context>
            Pregunta: {input}
            Respuesta:
        """
    },
    "en": {
        "tts_voice": {"language_code": "en-US", "name": "en-US-Wavenet-C"},
        "prompt_template": """
            You are an expert assistant for the ESAP university. Your sole function is to answer questions based on the content of the documents provided in the 'Context'.
            Critical Instructions:
            1. Exhaustive Search: Before answering, CAREFULLY and COMPLETELY review all the 'Context'. The answer will ALWAYS be in that text.
            2. Answer: If you find the answer, present it clearly and concisely, always using the most up-to-date document as of the current date (2025).
            3. Handling Uncertainty: Only if after an exhaustive search you do not find an answer, kindly indicate that you do not have the information.
            4. Source management: after giving the answer, tell the user where you obtained the information, the document, the page number also.
            Context: <context>{context}</context>
            Question: {input}
            Answer:
        """
    }
}
DEFAULT_LANG = "es"

# --- FUNCIONES DE LÓGICA ---

@st.cache_data
def load_local_css(file_name):
    try:
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo CSS: {e}")

def load_custom_chat_css():
    """Aplica CSS personalizado para los avatares del chat"""
    css = """
    <style>
    /* Ocultar los avatares predeterminados de Streamlit */
    .stChatMessage > div:first-child {
        display: none !important;
    }
    
    /* Estilos para el contenedor del chat */
    .chat-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1rem;
        gap: 12px;
    }
    
    /* Avatar personalizado */
    .custom-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        flex-shrink: 0;
        border: 2px solid #e0e0e0;
    }
    
    /* Contenido del mensaje */
    .message-content {
        background-color: #f0f2f6;
        padding: 12px 16px;
        border-radius: 16px;
        max-width: 80%;
        word-wrap: break-word;
        color: black;
    }
    
    /* Estilos específicos para el asistente */
    .assistant-message {
        flex-direction: row;
    }
    
    .assistant-message .message-content {
        background-color: #e3f2fd;
    }
    
    /* Estilos específicos para el usuario */
    .user-message {
        flex-direction: row-reverse;
        justify-content: flex-start;
    }
    
    .user-message .message-content {
        background-color: #f3e5f5;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def get_avatar_path(role):
    """Obtiene la ruta del avatar según el rol"""
    if role == "assistant":
        return CONFIG["ASSISTANT_AVATAR"]
    else:
        return CONFIG["USER_AVATAR"]

def render_chat_message(role, content, audio=None):
    """Renderiza un mensaje de chat con avatar personalizado"""
    avatar_path = get_avatar_path(role)
    
    # Verificar si existe la imagen del avatar
    if os.path.exists(avatar_path):
        avatar_html = f'<img src="data:image/png;base64,{get_base64_image(avatar_path)}" class="custom-avatar">'
    else:
        # Fallback a emoji si no existe la imagen
        avatar_emoji = "🤖" if role == "assistant" else "👤"
        avatar_html = f'<div class="custom-avatar" style="display: flex; align-items: center; justify-content: center; background-color: #ddd; font-size: 20px;">{avatar_emoji}</div>'
    
    message_class = "assistant-message" if role == "assistant" else "user-message"
    
    chat_html = f"""
    <div class="chat-container {message_class}">
        {avatar_html}
        <div class="message-content">
            {content}
        </div>
    </div>
    """
    
    st.markdown(chat_html, unsafe_allow_html=True)
    
    # Mostrar audio si existe
    if audio:
        st.audio(audio, format='audio/mp3', autoplay=True)

@st.cache_data
def get_base64_image(image_path):
    """Convierte una imagen a base64 para embebir en HTML"""
    import base64
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.warning(f"No se pudo cargar la imagen {image_path}: {e}")
        return None

@st.cache_resource
def verify_credentials_and_get_clients():
    try:
        creds_dict = dict(st.secrets['gcp_service_account'])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        stt_client = speech.SpeechClient(credentials=credentials)
        return tts_client, stt_client
    except Exception as e:
        st.error(f"Error crítico al verificar credenciales de Google Cloud: {e}", icon="🚨")
        return None, None

@st.cache_resource
def initialize_rag_components(_llm):
    try:
        pdf_folder_path = CONFIG["PDF_DOCUMENT_BASE_PATH"]
        
        if not os.path.isdir(pdf_folder_path):
            st.error(f"Error: La carpeta de documentos PDF '{pdf_folder_path}' no existe o no es un directorio válido.", icon="🚨")
            return None

        all_docs = []
        found_pdfs = False
        
        # Iterar sobre todos los archivos en la carpeta
        for filename in os.listdir(pdf_folder_path):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(pdf_folder_path, filename)
                try:
                    loader = PyPDFLoader(file_path)
                    all_docs.extend(loader.load())
                    found_pdfs = True
                except Exception as e:
                    st.warning(f"No se pudo cargar el archivo PDF '{filename}': {e}", icon="⚠️")
        
        if not found_pdfs:
            st.error(f"Error: No se encontró ningún documento PDF en la carpeta: {pdf_folder_path}", icon="🚨")
            return None
        
        if not all_docs:
            st.error("Error: No se pudo extraer contenido de ningún documento PDF encontrado.", icon="🚨")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(all_docs)
        
        api_key = st.secrets.get("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
        document_compressor = LLMChainExtractor.from_llm(_llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=document_compressor, base_retriever=base_retriever
        )
        return compression_retriever
    except Exception as e:
        st.error(f"Ocurrió un error crítico al inicializar la IA: {e}", icon="🚨")
        return None

def text_to_speech(client, text, voice_params):
    if not client or not text or not text.strip(): return None
    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=voice_params["language_code"], name=voice_params["name"]
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        return response.audio_content
    except Exception as e:
        st.warning(f"No se pudo generar el audio para esta respuesta (API TTS Error).", icon="🔇")
        return None

def speech_to_text(client, audio_bytes):
    if not client or not audio_bytes: return None
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1)
        mono_audio_buffer = io.BytesIO()
        audio_segment.export(mono_audio_buffer, format="wav")
        
        audio_content = mono_audio_buffer.getvalue()
        audio = speech.RecognitionAudio(content=audio_content)
        
        config = speech.RecognitionConfig(
            language_code="es-CO", alternative_language_codes=["en-US"], enable_automatic_punctuation=True
        )
        response = client.recognize(config=config, audio=audio)
        return response.results[0].alternatives[0].transcript if response.results else None
    except Exception as e:
        st.error(f"Error al procesar o transcribir el audio: {e}", icon="🚨")
        return None

def main():
    st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")
    load_local_css(CONFIG["CSS_FILE_PATH"])
    load_custom_chat_css()  # Cargar CSS personalizado para avatares

    # --- INICIALIZACIÓN ---
    tts_client, stt_client = verify_credentials_and_get_clients()
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: GOOGLE_API_KEY no está configurada.", icon="🚨")
        st.stop()
    
    # Using gemini-1.5-pro for better understanding of complex documents including tables.
    # Note: This model might have different cost implications than gemini-1.5-flash.
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0)
    retriever = initialize_rag_components(llm)
    
    if not all([tts_client, stt_client, retriever, llm]):
        st.error("La aplicación no puede continuar debido a un error de inicialización.", icon="🛑")
        st.stop()

    # --- GESTIÓN DE ESTADO DE LA SESIÓN ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant", 
            "content": CONFIG["WELCOME_MESSAGE"],
            "audio": None
        }]
    if "prompt_to_process" not in st.session_state:
        st.session_state.prompt_to_process = None

    # --- INTERFAZ GRÁFICA ---
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        if os.path.exists(CONFIG["HEADER_IMAGE"]):
            st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.title(CONFIG["APP_TITLE"])
    st.write(CONFIG["APP_SUBHEADER"])

    # --- LÓGICA DE RENDERIZADO DEL CHAT ---
    # 1. Dibuja todos los mensajes que ya han sido completados.
    for message in st.session_state.messages:
        render_chat_message(message["role"], message["content"], message.get("audio"))

    # 2. Revisa si hay un nuevo prompt que necesite ser procesado.
    if prompt := st.session_state.get("prompt_to_process"):
        # Muestra el placeholder del asistente con el spinner.
        with st.spinner(CONFIG["SPINNER_MESSAGE"]):
            # Lógica de negocio: procesa el prompt para obtener la respuesta.
            try:
                lang_code = detect(prompt)
                if lang_code not in LANG_CONFIG: lang_code = DEFAULT_LANG
            except LangDetectException:
                lang_code = DEFAULT_LANG
            
            selected_lang_config = LANG_CONFIG[lang_code]
            prompt_obj = ChatPromptTemplate.from_template(selected_lang_config["prompt_template"])
            document_chain = create_stuff_documents_chain(llm, prompt_obj)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            response = rag_chain.invoke({"input": prompt})
            respuesta_ia = response.get("answer", "No pude encontrar una respuesta.")
            audio_content = text_to_speech(tts_client, respuesta_ia, selected_lang_config["tts_voice"])

            # Añade la respuesta final al historial de mensajes.
            st.session_state.messages.append({
                "role": "assistant",
                "content": respuesta_ia,
                "audio": audio_content
            })
        
        # Limpia el prompt pendiente y re-ejecuta para mostrar la respuesta final.
        st.session_state.prompt_to_process = None
        st.rerun()

    # --- MANEJO DE ENTRADAS DEL USUARIO ---
    # Entrada de texto: solo añade el prompt al historial y marca que debe ser procesado.
    if prompt_texto := st.chat_input("Escribe tu pregunta o usa el micrófono..."):
        st.session_state.messages.append({"role": "user", "content": prompt_texto})
        st.session_state.prompt_to_process = prompt_texto
        st.rerun()

    # Entrada de audio: transcribe, añade el prompt y marca para procesar.
    audio_bytes_grabados = audio_recorder(text="", icon_size="2x", recording_color="#e84242", neutral_color="#646464")
    if audio_bytes_grabados:
        with st.spinner("Transcribiendo..."):
            transcribed_prompt = speech_to_text(stt_client, audio_bytes_grabados)
        
        if transcribed_prompt:
            st.session_state.messages.append({"role": "user", "content": transcribed_prompt})
            st.session_state.prompt_to_process = transcribed_prompt
            st.rerun()
        else:
            st.toast("No pude entender lo que dijiste.", icon="🎙️")
    
    st.divider()
    st.caption(f"For more information, visit the [{CONFIG['WEBSITE_LINK_TEXT']}]({CONFIG['OFFICIAL_WEBSITE_URL']}).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Ha ocurrido un error inesperado en la aplicación: {e}", icon="💥")
        st.exception(e)
