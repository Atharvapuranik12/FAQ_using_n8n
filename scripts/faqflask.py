from flask import Flask, request, jsonify

from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = Flask(__name__)

load_dotenv()
gemini = os.getenv("Gemini-API")
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', google_api_key=gemini)

loader = CSVLoader(file_path="FAQs.csv", source_column='prompt')
data = loader.load()
texts = [doc.page_content for doc in data]

model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

vector_path = "FaissDataBase"
vector_db = FAISS.from_texts(texts, model)
vector_db.save_local(vector_path)

db = FAISS.load_local(vector_path, model, allow_dangerous_deserialization=True)
retriever = db.as_retriever()

prompt_template = PromptTemplate(
    template="""Given the following context and a question, generate an answer based on this context.
    Include as much as possible from the "response" section in the source document.
    If the answer is not defined in the context, state "I don't know".

    Context: {context}
    Question: {question}

    Answer:""",
    input_variables=['context', 'question']
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    input_key='question',
    return_source_documents=True,
    chain_type_kwargs={'prompt': prompt_template}
)

@app.route('/ask', methods=['POST'])

def ask():
    data = request.get_json()
    question = data['question']
    result = chain({"question": question})
    return jsonify({
        "answer": result['result']
    })

if __name__ == '__main__':
    app.run(port=5000)
