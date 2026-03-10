import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Configuración de API Keys (Extraídas de st.secrets)
gemini_api = st.secrets["GOOGLE_API_KEY"]
pinecone_api = st.secrets["PINECONE_API_KEY"]

# 2. Función de procesamiento (Asegúrate de haberla ejecutado al menos una vez para llenar tu índice)
def ingest_data():
    text = "este repositorio es una libreria que ayuda a hacer operaciones aritmeticas"
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    
    # Usamos models/embedding-001 que es el estándar más estable
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api)
    
    vectorstore = PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name="my-first-index", 
        pinecone_api_key=pinecone_api
    )
    return vectorstore

# 3. Interfaz de Usuario
st.title("🤖 My RAG on GitHub")

# Botón opcional para cargar datos si el índice está vacío
if st.button("Ingestar Datos Iniciales"):
    with st.spinner("Subiendo datos a Pinecone..."):
        ingest_data()
        st.success("¡Datos cargados con éxito!")

query = st.text_input("Ask a question about the code:")

if query:
    try:
        # Definimos embeddings con el nombre de modelo correcto
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api)
        
        # Conectamos al índice REAL (my-first-index)
        vectorstore = PineconeVectorStore(
            index_name="my-first-index", 
            embedding=embeddings,
            pinecone_api_key=pinecone_api
        )
        
        # Configuración del modelo de chat
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api)
        
        # Configuración de la cadena de consulta
        qa = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=vectorstore.as_retriever()
        )
        
        # Ejecución y respuesta
        with st.spinner("Buscando respuesta..."):
            respuesta = qa.invoke(query)
            st.markdown("### Respuesta:")
            st.write(respuesta["result"])
            
    except Exception as e:
        st.error(f"Hubo un error en la consulta: {e}")
