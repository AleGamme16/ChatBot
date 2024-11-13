from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def load_documents(pdf_path):
    """Carga y procesa el documento PDF"""
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  
        chunk_overlap=30,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

def configure_model():
    """Configura el modelo de lenguaje"""
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=110,  
        temperature=0.3,    
        top_p=0.9,
        repetition_penalty=1.2,  
        do_sample=True
    )
    return HuggingFacePipeline(pipeline=pipe)

def setup_retriever(docs):
    """Configura el sistema de recuperación"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  
    )

def create_prompt_template():
    """Crea el template del prompt"""
    template = """Instrucciones: Proporciona una respuesta clara y concisa a la pregunta usando solo la información del contexto proporcionado. 
    No agregues información adicional ni especules.

    Contexto: {context}

    Pregunta: {question}

    Respuesta concisa:"""
    
    return ChatPromptTemplate.from_template(template)

def format_docs(docs):
    """Formatea los documentos para el contexto"""
    return " ".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, llm, prompt):
    """Construye la cadena RAG"""
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def process_response(response):
    if "Respuesta concisa:" in response:
        response = response.split("Respuesta concisa:")[-1]
    
    response = response.strip()
    response = response.split("\n")[0] 
    
    return response

def main():
    pdf_path = "C:\\Users\\alega\\ProyectoPromtior\\ChatBot\\SimulacionBDDPromtior.pdf"
    print("Cargando documentos y configurando el modelo...")
    
    docs = load_documents(pdf_path)
    llm = configure_model()
    retriever = setup_retriever(docs)
    prompt = create_prompt_template()
    rag_chain = build_rag_chain(retriever, llm, prompt)
    
    print("¡Listo para responder preguntas!")
    
    while True:
        pregunta = input("\nHaz una pregunta sobre Promtior (o escribe 'salir' para terminar): ")
        if pregunta.lower() == 'salir':
            break
            
        print("Procesando tu pregunta...")
        try:
            respuesta = rag_chain.invoke(pregunta)
            respuesta_limpia = process_response(respuesta)
            print("\nRespuesta:", respuesta_limpia)
        except Exception as e:
            print(f"\nError al procesar la pregunta: {e}")

if __name__ == "__main__":
    main()