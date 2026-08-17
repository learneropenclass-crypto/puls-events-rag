import os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TOP_K           = int(os.getenv("TOP_K", 5))
INDEX_DIR       = Path(os.getenv("INDEX_DIR", str(Path(__file__).parent.parent / "vector_store" / "faiss_index")))

RAG_PROMPT_TEMPLATE = """Tu es un assistant specialise dans les evenements culturels d Ile-de-France pour Puls-Events.
Reponds UNIQUEMENT a partir des evenements fournis dans le contexte ci-dessous.
Si l information n est pas dans le contexte, dis-le clairement.
Reponds toujours en francais, de maniere conviviale. Mentionne le lieu et la date pour chaque evenement.

CONTEXTE :
{context}

QUESTION : {question}

REPONSE :"""

def load_vectorstore():
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY non definie dans .env")
    if not (INDEX_DIR / "index.faiss").exists():
        raise FileNotFoundError("Index FAISS introuvable. Lancez d abord : python scripts/vectorize.py")
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)

def format_docs(docs):
    return "\n\n".join(
        f"- {d.metadata.get('title','?')} ({d.metadata.get('city','')}, {(d.metadata.get('date_start') or '')[:10]})\n  {d.page_content}"
        for d in docs
    )

def run_chatbot():
    print("\n" + "="*60)
    print("   Puls-Events - Assistant evenements culturels IDF")
    print("="*60)
    print("Posez vos questions sur les evenements en Ile-de-France.")
    print("Tapez 'quitter' pour terminer.\n")
    print("Chargement du systeme RAG...")
    try:
        vs        = load_vectorstore()
        retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
        llm       = ChatMistralAI(model="mistral-large-latest", mistral_api_key=MISTRAL_API_KEY, temperature=0.3, max_tokens=1024)
        prompt    = PromptTemplate(template=RAG_PROMPT_TEMPLATE, input_variables=["context", "question"])
        chain     = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("Systeme pret !\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur : {e}")
        return

    while True:
        try:
            question = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break
        if not question:
            continue
        if question.lower() in ("quitter", "exit", "q"):
            print("Au revoir !")
            break
        print("\nRecherche en cours...\n")
        try:
            answer = chain.invoke(question)
            print(f"Assistant : {answer}\n")
        except Exception as e:
            print(f"Erreur : {e}\n")

if __name__ == "__main__":
    run_chatbot()


