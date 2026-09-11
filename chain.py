from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def get_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = ChatPromptTemplate.from_template("Answer question using context:\nContext: {context}\nQuestion: {question}")
    return prompt | llm
