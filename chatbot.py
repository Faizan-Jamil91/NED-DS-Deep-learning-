import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Initialize model and vector store
model = OllamaLLM(model="deepseek-r1:1.5b")
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)

# Define system prompt template
template = """
You are an advanced AI assistant designed for answering questions based on the given PDF document.  
Your task is to extract the most relevant information from the PDF to provide an accurate answer.  

Guidelines:  
- Retrieve only relevant content from the PDF.  
- If the PDF contains the exact answer, provide it as-is.  
- If the PDF does not contain the answer, state: "The answer is not found in the provided document."  

Question: {question}  
Extracted Context from PDF: {context}  
Answer:  

"""

pdfs_directory = './pdfs/'

# Function to save uploaded PDF
def upload_pdf(file):
    file_path = pdfs_directory + file.name
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path  # Return saved file path

# Function to load PDF content
def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

# Function to split text
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Corrected parameter name
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)  # Return split documents

# Function to index documents in vector store
def index_docs(documents):
    vector_store.add_documents(documents)

# Function to retrieve relevant documents based on query
def retrieve_docs(query):
    return vector_store.similarity_search(query)

# Function to answer a question using retrieved documents
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})

# Streamlit UI
st.title("📄 PDF Q&A Assistant")

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF", type="pdf", accept_multiple_files=False)

if uploaded_file:
    # Save and process PDF
    file_path = upload_pdf(uploaded_file)
    documents = load_pdf(file_path)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    # Chat-based question input
    question = st.chat_input("Ask a question about the document...")

    if question:
        # Display user question
        st.chat_message("user").write(question)

        # Retrieve related document chunks
        related_documents = retrieve_docs(question)

        # Generate an answer
        answer = answer_question(question, related_documents)

        # Display assistant's response
        st.chat_message("assistant").write(answer)
