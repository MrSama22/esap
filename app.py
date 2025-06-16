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

# --- NUEVO: Librería para un control de grabación de audio más limpio ---
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

# --- CONFIGURACIÓN MULTILINGÜE (Sin cambios) ---
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

# --- FUNCIONES DE LÓGICA (la mayoría sin cambios) ---

@st.cache_data
def load_local_css(file_name):
    try:
        if os.path.exists(file_name):
            with open(file_name, "r", encoding="utf-8") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                return True # NUEVO: Devolver estado para comprobación
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo CSS: {e}")
    return False

@st.cache_resource
def verify_credentials_and_get_clients():
    try:
        creds_dict = dict(st.secrets['gcp_service_account'])
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
        stt_client = speech.SpeechClient(credentials=credentials)
        st.success("Credenciales de Google Cloud verificadas correctamente.", icon="✅")
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
        st.success("Componentes de IA inicializados correctamente.", icon="✅")
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
    if not client or not audio_bytes: return None
    try:
        audio = speech.RecognitionAudio(content=audio_bytes)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_codes=["es-CO", "en-US"],
            enable_automatic_punctuation=True
        )
        
        with st.spinner("Transcribiendo tu voz..."):
            response = client.recognize(config=config, audio=audio)
        
        if response.results and response.results[0].alternatives:
            return response.results[0].alternatives[0].transcript
        else:
            st.warning("No pude entender lo que dijiste. Por favor, intenta de nuevo.", icon="🤔")
            return None
            
    except Exception as e:
        st.error(f"Error al transcribir el audio (Speech-to-Text): {e}", icon="🚨")
        return None

# --- LÓGICA PRINCIPAL DE LA APP ---

def main():
    """Función principal que orquesta la aplicación Streamlit."""
    
    st.set_page_config(page_title=CONFIG["PAGE_TITLE"], page_icon=CONFIG["PAGE_ICON"], layout="wide")
    
    # --- Carga de CSS y configuración inicial ---
    load_local_css(CONFIG["CSS_FILE_PATH"])

    # --- Verificación de credenciales y componentes de IA ---
    # Se ejecuta solo una vez gracias a @st.cache_resource
    tts_client, stt_client = verify_credentials_and_get_clients()
    retriever, llm = initialize_rag_components()

    # Si los componentes críticos fallan, detener la ejecución.
    if not all([tts_client, stt_client, retriever, llm]):
        st.error("La aplicación no puede continuar debido a un error de inicialización. Revisa los mensajes de error anteriores.", icon="🛑")
        st.stop()
        
    # --- Encabezado de la página ---
    with st.container():
        st.markdown('<div class="header-container">', unsafe_allow_html=True)
        if os.path.exists(CONFIG["HEADER_IMAGE"]):
            st.image(CONFIG["HEADER_IMAGE"], use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.title(CONFIG["APP_TITLE"])
    st.write(CONFIG["APP_SUBHEADER"])
    
    # --- Inicialización del estado de la sesión ---
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": CONFIG["WELCOME_MESSAGE"]}]
    # NUEVO: Estados para controlar la grabación y el texto transcrito
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = None

    # --- Lógica del Chat (procesamiento de respuesta) ---
    def process_and_display_response(prompt: str):
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
                    
                    st.session_state.messages.append({"role": "assistant", "content": respuesta_ia})
                
                except Exception as e:
                    st.error(f"Error al invocar la cadena de RAG: {e}", icon="🚨")
                    st.session_state.messages.append({"role": "assistant", "content": f"Lo siento, tuve un problema al procesar tu solicitud: {e}"})

    # --- DIBUJAR LA INTERFAZ DEL CHAT ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- NUEVA LÓGICA DE BARRA DE ENTRADA UNIFICADA ---
    # Usamos st.chat_input para una mejor experiencia en móviles y escritorio
    prompt = st.chat_input("Escribe tu pregunta o usa el micrófono...", key="text_input")
    
    # Si el usuario envió texto, procesarlo
    if prompt:
        process_and_display_response(prompt)
        st.rerun() # Limpiar la caja de texto y mostrar la respuesta

    # Lógica del botón del micrófono
    st.markdown('<div class="mic-button-container">', unsafe_allow_html=True)
    audio_bytes = audio_recorder(
        text="", # Texto del botón, lo dejamos vacío para usar solo el ícono
        icon_size="2x", # Tamaño del ícono
        pause_threshold=3.0, # Pausa para detener automáticamente (opcional)
        recording_color="#e84242",
        neutral_color="#646464",
        key="audio_recorder"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Si audio_recorder devuelve bytes, significa que se completó una grabación
    if audio_bytes:
        transcribed_prompt = speech_to_text(stt_client, audio_bytes)
        if transcribed_prompt:
            # En lugar de procesar, llenamos el input para que el usuario envíe manualmente
            st.session_state.transcribed_text = transcribed_prompt
            # Usamos JS para actualizar el valor del input y hacer un rerun
            st.components.v1.html(
                f"""
                <script>
                // Busca el text_input de Streamlit por su etiqueta
                var input = window.parent.document.querySelector("input[aria-label='Escribe tu pregunta o usa el micrófono...']");
                input.value = `{transcribed_prompt.replace("`", "\\`")}`; // Llena el input
                input.dispatchEvent(new Event('input', {{ bubbles: true }})); // Notifica a Streamlit del cambio

                // Simula un "Enter" para enviar el formulario
                const enterKeyEvent = new KeyboardEvent('keydown', {{
                    key: 'Enter',
                    code: 'Enter',
                    keyCode: 13,
                    which: 13,
                    bubbles: true,
                    cancelable: true,
                }});
                input.dispatchEvent(enterKeyEvent);
                </script>
                """,
                height=0,
            )


    # --- ENLACE FINAL ---
    st.divider()
    st.caption(f"Para más información, puedes visitar la [{CONFIG['WEBSITE_LINK_TEXT']}]({CONFIG['OFFICIAL_WEBSITE_URL']}).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # NUEVO: Captura de errores a nivel de aplicación para depuración
        st.error(f"Ha ocurrido un error inesperado en la aplicación: {e}", icon="💥")
        st.exception(e) # Muestra el stack trace completo para facilitar la depuración
