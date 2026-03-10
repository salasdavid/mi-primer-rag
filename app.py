import streamlit as st
from langchain_community.document_loaders import GithubFileLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. API Keys configuration (Streamlit will handle them)
gemini_api = st.secrets["GOOGLE_API_KEY"]
pinecone_api = st.secrets["PINECONE_API_KEY"]

# 2. Load and Process (Example with a GitHub repo)
def ingest_data():
    # For this example, load simple text; can use GitHub loaders
    text = "Here goes the content of your repository or files..."
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])
    
    embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=gemini_api)

    vectorstore = PineconeVectorStore.from_documents(
        docs, embeddings, index_name="your-pinecone-index", pinecone_api_key=pinecone_api
    )
    return vectorstore

# 4. User Interface
st.title("🤖 My RAG on GitHub")
query = st.text_input("Ask a question about the code:")

if query:
    try:
        # 4 espacios aquí
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api)
        test_vec = embeddings.embed_query("Prueba")
    except Exception as e:
        # 8 espacios aquí (porque está dentro del if y del except)
        st.error(f"Error real de Google: {e}")
    
    # Asegúrate de que el resto del código siga alineado con el 'try'
    vectorstore = PineconeVectorStore(index_name="tu-indice", embedding=embeddings)
    # ... resto de tu código
   
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_api)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    
    respuesta = qa.invoke(query)
    st.write(respuesta["result"])
