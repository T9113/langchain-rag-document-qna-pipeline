from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def build_index(doc_path):
    loader = TextLoader(doc_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    return vectorstore

if __name__ == "__main__":
    print("LangChain RAG pipeline initialized.")
