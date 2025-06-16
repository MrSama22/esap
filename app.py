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

# --- CONFIGURACIÓN ---
CONFIG = {
    "PAGE_TITLE": "Asistente CSDB",
    "PAGE_ICON": "🎓",
    "HEADER_IMAGE": "logo1.png",
    "APP_TITLE": "🎓 Asistente Virtual del Colegio Santo Domingo BIilingüe",
    "APP_SUBHEADER": "¡Hola! Estoy aquí para responder tus preguntas basándome en el documento oficial.",
    "WELCOME_MESSAGE": "¡Hola! Soy el asistente virtual del CSD. ¿En qué puedo ayudarte? / Hello! I'm the CSDB virtual assistant. How can I help you?",
    "SPINNER_MESSAGE": "Buscando y preparando tu respuesta...",
    "PDF_DOCUMENT_PATH": "documento.pdf",
    "OFFICIAL_WEBSITE_URL": "https://colegiosantodomingo.edu.co/",
    "WEBSITE_LINK_TEXT": "Visita la página web oficial",
    "CSS_FILE_PATH": "styles.css"
}

# --- CONFIGURACIÓN MULTILINGÜE ---
LANG_CONFIG = {
    "es": {
        "tts_voice": {"language_code": "es-US", "name": "es-US-Standard-B"},
        "prompt_template": """
            Eres un asistente experto del Colegio Santo Domingo Bilingüe. Tu única función es responder preguntas basándote en el contenido de un documento institucional que se te proporciona en el 'Contexto'.

            **Instrucciones Críticas:**
            1.  **Búsqueda Exhaustiva:** Antes de responder, revisa CUIDADOSAMENTE y de forma COMPLETA todo el 'Contexto' que se te ha entregado. La respuesta que buscas SIEMPRE estará en ese texto. No asumas que no la tienes. Busca en cada rincón del contexto proporcionado.
            3.  **Respuesta Directa:** Si encuentras la respuesta, preséntala de forma clara y concisa. Por ejemplo, si te preguntan por una persona, responde directamente con su nombre y cargo y una breve descripcion.
            4.  **Manejo de Incertidumbre:** Solo si después de una búsqueda exhaustiva en el 'Contexto' no encuentras una respuesta directa, y únicamente en ese caso, indica amablemente que no tienes la información específica.

            **Contexto:**
            <context>{context}</context>
            
            **Pregunta:** {input}
            
            **Respuesta:**
        """
    },
    "en": {
        "tts_voice": {"language_code": "en-US", "name": "en-US-Wavenet-C"},
        "prompt_template": """
            You are an expert assistant for the Santo Domingo Bilingual School. Your sole function is to answer questions based on the content of an institutional document provided to you in the 'Context'.

            **Critical Instructions:**
            1.  **Exhaustive Search:** Before answering, CAREFULLY and COMPLETELY review all the 'Context' you have been given. The answer you are looking for will ALWAYS be in that text. Do not assume you don't have it. Search every corner of the provided context.
            3.  **Direct Answer:** If you find the answer, present it clearly and concisely. For example, if asked about a person, answer directly with their name and role and a short description.
            4.  **Handling Uncertainty:** Only if, after an exhaustive search of the 'Context', you do not find a direct answer, and only in that case, kindly indicate that you do not have the specific information.

            **Context:**
            <context>{context}</context>
            
            **Question:** {input}
            
            **Answer:**
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
def initialize_rag_components():
    try:
        load_dotenv()
        api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
        if not api_key:
            st.error("Error: GOOGLE_API_KEY no está configurada.", icon="🚨")
            return None, None
        
        if not os.path.exists(CONFIG["PDF_DOCUMENT_PATH"]):
            st.error(f"Error: No se encontró el documento PDF en la ruta: {CONFIG['PDF_DOCUMENT_PATH']}", icon="🚨")
            return None, None

        loader = PyPDFLoader(CONFIG["PDF_DOCUMENT_PATH"])
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
        
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 30})
        document_compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=document_compressor, 
            base_retriever=base_retriever
        )
        return compression_retriever, llm
    except Exception as e:
        st.error(f"Ocurrió un error crítico al inicializar la IA: {e}", icon="🚨")
        return None, None

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
        st.error(f"Error al generar el audio (Text-to-Speech): {e}", icon="🚨")
        return None

def speech_to_text(client, audio_bytes):
    if not client or not audio_bytes: 
        return None

    try:
        # Convierte a mono para asegurar compatibilidad
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio_segment = audio_segment.set_channels(1)
        
        mono_audio_bytes_io = io.BytesIO()
        audio_segment.export(mono_audio_bytes_io, format="wav")
        mono_audio_bytes = mono_audio_bytes_io.getvalue()
        
        audio = speech.RecognitionAudio(content=mono_audio_bytes)

        config = speech.RecognitionConfig(
            language_code="es-CO",
            alternative_language_codes=["en-US"],
            enable_automatic_punctuation=True
        )
        
        response = client.recognize(config=config, audio=audio)
        
        if response.results and response.results[0].alternatives:
            return response.results[0].alternatives[0].transcript
        else:
            return None
            
    except Exception as e:
        st.error(f"Error al procesar o transcribir el audio: {e}", icon="🚨")
        return None
        
def main():
    st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")
    
    load_local_css(CONFIG["CSS_FILE_PATH"])

    tts_client, stt_client = verify_credentials_and_get_clients()
    retriever, llm = initialize_rag_components()

    if not all([tts_client, stt_client, retriever, llm]):
        st.error("La aplicación no puede continuar debido a un error de inicialización. Revisa los mensajes anteriores.", icon="🛑")
        st.stop()
        
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        if os.path.exists(CONFIG["HEADER_IMAGE"]):
            st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.title(CONFIG["APP_TITLE"])
    st.write(CONFIG["APP_SUBHEADER"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": CONFIG["WELCOME_MESSAGE"]}]

    # --- INICIO DE LÓGICA MODIFICADA ---

    def process_and_display_response(prompt: str):
        # Añade el prompt del usuario al historial para que se muestre en la UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        
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
                rag_chain = create_retrieval_chain(retriever, document_chain)

                try:
                    response = rag_chain.invoke({"input": prompt})
                    respuesta_ia = response["answer"]
                    
                    st.markdown(respuesta_ia)
                    audio_content = text_to_speech(tts_client, respuesta_ia, tts_voice_params)
                    if audio_content:
                        st.audio(audio_content, autoplay=True)
                    
                    # Añade la respuesta del asistente al historial
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
                
                except Exception as e:
                    error_message = f"Lo siento, tuve un problema al procesar tu solicitud: {e}"
                    st.error(f"Error al invocar la cadena de IA: {e}", icon="🚨")
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    # Dibuja el historial de chat existente en cada ejecución
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- NUEVO BLOQUE: PROCESA EL PROMPT DE AUDIO SI EXISTE EN LA SESIÓN ---
    # Esto se ejecuta después de transcribir y hacer st.rerun()
    if "prompt_from_audio" in st.session_state and st.session_state.prompt_from_audio:
        prompt_de_audio = st.session_state.prompt_from_audio
        # Limpia la variable de sesión para que no se vuelva a procesar en el siguiente ciclo
        st.session_state.prompt_from_audio = None 
        # Llama a la función principal para procesar la respuesta
        process_and_display_response(prompt_de_audio)
        # Forzamos un rerun final para asegurar que el historial se redibuje correctamente con la nueva respuesta
        st.rerun()

    # --- ENTRADA DE TEXTO Y AUDIO ---

    # La entrada de texto funciona como siempre
    if prompt_texto := st.chat_input("Escribe tu pregunta o usa el micrófono..."):
        process_and_display_response(prompt_texto)
        st.rerun()

    # La entrada de audio ahora usa session_state para comunicarse
    st.markdown('<div class="mic-button-container">', unsafe_allow_html=True)
    audio_bytes = audio_recorder(
        text="",
        icon_size="2x",
        recording_color="#e84242",
        neutral_color="#646464"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if audio_bytes:
        with st.spinner("Transcribiendo tu voz..."):
            transcribed_prompt = speech_to_text(stt_client, audio_bytes)
        
        if transcribed_prompt:
            # Guarda el texto transcrito en el estado de la sesión y re-ejecuta el script
            st.session_state.prompt_from_audio = transcribed_prompt
            st.rerun()
        else:
            # Informa al usuario si la transcripción falla
            st.toast("No pude entender lo que dijiste. Por favor, intenta de nuevo.", icon="🎙️")
    
    # --- FIN DE LÓGICA MODIFICADA ---

    st.divider()
    st.caption(f"Para más información, puedes visitar la [{CONFIG['WEBSITE_LINK_TEXT']}]({CONFIG['OFFICIAL_WEBSITE_URL']}).")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Este es un último recurso para capturar cualquier error no manejado
        st.error(f"Ha ocurrido un error inesperado en la aplicación: {e}", icon="💥")
        st.exception(e)
