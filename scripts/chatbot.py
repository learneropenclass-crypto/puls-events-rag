"""
chatbot.py
----------
Chatbot RAG interactif pour la recommandation d'événements culturels
en Île-de-France, basé sur LangChain, Mistral AI et FAISS.

Architecture de la chaîne RAG :
    Requête utilisateur
         │
         ▼
    [Embedding Mistral]  →  Vecteur de la requête (1024 dimensions)
         │
         ▼
    [Recherche FAISS]    →  Top-K chunks sémantiquement proches
         │
         ▼
    [Augmentation]       →  Prompt = Contexte événements + Question
         │
         ▼
    [Mistral Large LLM]  →  Réponse personnalisée en français

Choix de conception :
    - Stratégie "stuff" : tous les documents récupérés dans un seul prompt
    - Température 0.3 : réponses factuelles et cohérentes
    - Grounding strict : le LLM répond uniquement à partir du contexte fourni
    - Pas d'historique de conversation dans ce POC (chaque question est indépendante)

Usage:
    python scripts/chatbot.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv

from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
TOP_K           = int(os.getenv("TOP_K", 5))
INDEX_DIR       = Path(__file__).parent.parent / "vector_store" / "faiss_index"

# ── Prompt template ────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """Tu es un assistant spécialisé dans les événements culturels \
d'Île-de-France pour la plateforme Puls-Events.
Réponds UNIQUEMENT à partir des événements fournis dans le contexte ci-dessous.
Si l'information n'est pas dans le contexte, dis-le clairement.
Réponds toujours en français, de manière conviviale. \
Mentionne le lieu et la date pour chaque événement.

──────────────────────────────────────────────
CONTEXTE (événements disponibles) :
{context}
──────────────────────────────────────────────

QUESTION : {question}

RÉPONSE :"""


# ── Fonctions ──────────────────────────────────────────────────────────────────

def load_vectorstore() -> FAISS:
    """
    Charge l'index FAISS persisté depuis le disque.

    Initialise le modèle d'embedding Mistral nécessaire pour vectoriser
    les requêtes entrantes lors de la recherche par similarité.

    Returns:
        FAISS: Instance de la base vectorielle chargée et prête à l'emploi.

    Raises:
        ValueError:        Si MISTRAL_API_KEY n'est pas définie dans .env.
        FileNotFoundError: Si l'index FAISS n'existe pas (vectorize.py non exécuté).
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY non définie. "
            "Ajoutez-la dans votre fichier .env"
        )

    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {INDEX_DIR}\n"
            "Lancez d'abord : python scripts/vectorize.py"
        )

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY,
    )
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def format_docs(docs: list) -> str:
    """
    Formate les documents récupérés en contexte textuel pour le prompt RAG.

    Chaque document est présenté avec son titre, sa ville, sa date de début
    et son contenu textuel. Ce contexte est injecté dans le prompt template.

    Args:
        docs (list[Document]): Documents LangChain récupérés par FAISS.

    Returns:
        str: Contexte formaté, prêt à être injecté dans le prompt.
    """
    return "\n\n".join(
        f"- {d.metadata.get('title', '?')} "
        f"({d.metadata.get('city', '')}, "
        f"{(d.metadata.get('date_start') or '')[:10]})\n"
        f"  {d.page_content}"
        for d in docs
    )


def build_rag_chain(vectorstore: FAISS):
    """
    Construit la chaîne RAG LangChain (LCEL) combinant retrieval et génération.

    La chaîne est composée de :
    1. Un retriever FAISS configuré avec TOP_K documents
    2. Un formateur de contexte (format_docs)
    3. Un prompt template structuré (RAG_PROMPT_TEMPLATE)
    4. Le LLM Mistral Large (temperature=0.3, max_tokens=1024)
    5. Un parser de sortie texte (StrOutputParser)

    Args:
        vectorstore (FAISS): Base vectorielle FAISS chargée.

    Returns:
        Runnable: Chaîne RAG LangChain (LCEL) prête à invoquer.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    llm = ChatMistralAI(
        model="mistral-large-latest",
        mistral_api_key=MISTRAL_API_KEY,
        temperature=0.3,
        max_tokens=1024,
    )
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def run_chatbot() -> None:
    """
    Lance l'interface interactive du chatbot RAG en ligne de commande.

    Charge le système RAG, puis entre dans une boucle de conversation
    jusqu'à ce que l'utilisateur tape 'quitter' ou 'exit'.
    Chaque question est traitée indépendamment (pas d'historique).
    """
    print("\n" + "═" * 60)
    print("🎭  Puls-Events — Assistant événements culturels IDF")
    print("═" * 60)
    print("Posez vos questions sur les événements en Île-de-France.")
    print("Tapez 'quitter' pour terminer.\n")

    print("⏳ Chargement du système RAG…")
    try:
        vs    = load_vectorstore()
        chain = build_rag_chain(vs)
        print("✅ Système prêt !\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Erreur : {e}")
        return

    while True:
        try:
            question = input("🗣️  Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not question:
            continue
        if question.lower() in ("quitter", "exit", "q"):
            print("Au revoir ! 👋")
            break

        print("\n⏳ Recherche en cours…\n")
        try:
            answer = chain.invoke(question)
            print(f"🤖 Assistant : {answer}\n")
        except Exception as e:
            print(f"❌ Erreur lors de la génération : {e}\n")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_chatbot()
