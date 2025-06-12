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
# ... otras importaciones ...

# --- MODIFICACIÓN: CAMBIAR EL LOADER DE PDF ---
# from langchain_community.document_loaders import PyPDFLoader # Ya no usamos este
from langchain_community.document_loaders import UnstructuredPDFLoader # Usamos este nuevo loader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langdetect import detect, LangDetectException

# --- NUEVOS IMPORTS PARA EL RETRIEVER DE COMPRESIÓN ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# ... (El resto de tu código de configuración y funciones auxiliares se mantiene igual) ...


# --- MODIFICACIÓN CLAVE DENTRO DE LA FUNCIÓN DE INICIALIZACIÓN ---
@st.cache_resource
def initialize_rag_components():
    load_dotenv()
    api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
    if not api_key: st.stop("Error: GOOGLE_API_KEY no está configurada.")
    if not os.path.exists(CONFIG["PDF_DOCUMENT_PATH"]): st.stop(f"Error: No se encontró el PDF.")

    # --- CAMBIO DE LOADER ---
    # Se utiliza UnstructuredPDFLoader en lugar de PyPDFLoader para una mejor extracción de texto estructurado.
    # El modo "single" procesa el documento como una sola unidad para mantener la coherencia.
    loader = UnstructuredPDFLoader(CONFIG["PDF_DOCUMENT_PATH"], mode="single", strategy="fast")
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
    
    # El retriever base ahora busca más documentos para darle más oportunidades al compresor
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # El compresor que usa el LLM para extraer la información relevante
    document_compressor = LLMChainExtractor.from_llm(llm)
    
    # El retriever de compresión contextual que combina los dos pasos anteriores
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=document_compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever, llm


# --- (EL RESTO DEL CÓDIGO DEL CHAT SE MANTIENE EXACTAMENTE IGUAL) ---
# ... (la lógica del chat que ya tienes) ...
