
import os
import json
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set Groq API keys
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
def setup_llm():
    return ChatOpenAI(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )

def load_qa_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = []
    if "intents" in data:
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                response = intent["responses"][0] if intent["responses"] else "I'm here to help."
                entries.append({"question": pattern, "answer": response})
    else:
        entries = data

    docs = [Document(page_content=entry["answer"], metadata={"question": entry["question"]}) for entry in entries]
    return docs, "\n\n".join([f"- **{entry['question']}**: {entry['answer']}" for entry in entries])

def load_pdf(file):
    reader = PdfReader(file)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    doc = Document(page_content=text)
    return [doc], text

def create_retriever(docs, model_name="all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    texts = [doc.page_content for doc in docs]
    return FAISS.from_texts(texts=texts, embedding=embeddings)

def hybrid_response(user_input, llm, retriever=None, threshold=0.7):
    if retriever:
        results = retriever.similarity_search_with_score(user_input, k=1)
        if results:
            content, score = results[0]
            if score >= threshold:
                return content.page_content
    return llm.predict(user_input)

# Streamlit Chat UI
st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
st.title("🧠 Mental Health Chatbot")

llm = setup_llm()
retriever = None
doc_text = ""

with st.sidebar:
    st.subheader("📄 Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload JSON or PDF", type=["json", "pdf"])
    if uploaded_file:
        if uploaded_file.name.endswith(".json"):
            with open("temp.json", "wb") as f:
                f.write(uploaded_file.read())
            docs, doc_text = load_qa_data("temp.json")
        else:
            docs, doc_text = load_pdf(uploaded_file)
        retriever = create_retriever(docs)
        st.markdown("### Extracted Content:")
        st.markdown(doc_text[:1500] + ("..." if len(doc_text) > 1500 else ""))

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, how can I support your mental health today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = hybrid_response(prompt, llm, retriever)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
