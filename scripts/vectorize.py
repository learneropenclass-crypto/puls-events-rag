"""
vectorize.py
------------
Script de vectorisation et d'indexation des événements culturels dans FAISS.

Ce script transforme les événements JSON nettoyés en vecteurs sémantiques
et les indexe dans une base FAISS persistante, prête à être interrogée
par le chatbot RAG.

Pipeline de traitement :
    1. Chargement des événements depuis data/events_clean.json
    2. Conversion en Documents LangChain avec métadonnées
    3. Découpage en chunks (RecursiveCharacterTextSplitter)
    4. Génération des embeddings via Mistral (mistral-embed)
    5. Indexation et persistance dans vector_store/faiss_index/

Usage:
    python scripts/vectorize.py              # Création ou mise à jour de l'index
    python scripts/vectorize.py --rebuild    # Reconstruction complète de l'index
"""

import os
import json
import logging
import argparse
import shutil
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 50))

BASE_DIR  = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "events_clean.json"
# Chemin de l'index FAISS configurable via la variable d'environnement INDEX_DIR.
# Fallback Windows-safe si non définie (évite les soucis d'accents dans les chemins).
INDEX_DIR = Path(os.getenv("INDEX_DIR", "C:/rag_index"))


# ── Fonctions ──────────────────────────────────────────────────────────────────

def load_events(path: Path) -> list[dict]:
    """
    Charge les événements nettoyés depuis le fichier JSON de pré-processing.

    Args:
        path (Path): Chemin vers le fichier events_clean.json.

    Returns:
        list[dict]: Liste des événements chargés.

    Raises:
        FileNotFoundError: Si le fichier n'existe pas (fetch_events.py non exécuté).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "Lancez d'abord : python scripts/fetch_events.py"
        )
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    logger.info(f"Événements chargés : {len(events)}")
    return events


def events_to_documents(events: list[dict]) -> list[Document]:
    """
    Convertit une liste d'événements en objets Document LangChain.

    Chaque Document contient :
    - page_content : le champ `text` agrégé de l'événement
    - metadata     : uid, title, city, department, region, venue, dates, categories

    Les métadonnées permettent la traçabilité des sources dans les réponses RAG.

    Args:
        events (list[dict]): Liste des événements nettoyés issus de fetch_events.py.

    Returns:
        list[Document]: Liste de Documents LangChain prêts pour le chunking.
    """
    documents = []
    for event in events:
        doc = Document(
            page_content=event.get("text", ""),
            metadata={
                "uid":        event.get("uid", ""),
                "title":      event.get("title", ""),
                "city":       event.get("city", ""),
                "department": event.get("department", ""),
                "region":     event.get("region", ""),
                "venue":      event.get("venue", ""),
                "date_start": event.get("date_start", ""),
                "date_end":   event.get("date_end", ""),
                "categories": event.get("categories", ""),
            },
        )
        documents.append(doc)
    logger.info(f"Documents LangChain créés : {len(documents)}")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Découpe les Documents en chunks sémantiquement cohérents.

    Stratégie de découpage :
    - RecursiveCharacterTextSplitter avec séparateurs naturels (\n\n, \n, espace)
    - Taille cible : CHUNK_SIZE tokens (défaut : 500)
    - Chevauchement : CHUNK_OVERLAP tokens (défaut : 50)

    Le chevauchement préserve la continuité sémantique entre les chunks
    adjacents, ce qui améliore la précision de la recherche vectorielle.

    Args:
        documents (list[Document]): Documents LangChain à découper.

    Returns:
        list[Document]: Chunks de taille contrôlée avec métadonnées héritées.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        f"Chunking : {len(documents)} documents → {len(chunks)} chunks "
        f"(taille={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})"
    )
    return chunks


def build_faiss_index(chunks: list[Document], rebuild: bool = False) -> FAISS:
    """
    Construit ou met à jour l'index FAISS à partir des chunks vectorisés.

    Le processus :
    1. Initialise le modèle d'embedding Mistral (mistral-embed, 1024 dimensions)
    2. En mode rebuild, supprime l'index existant
    3. Génère les embeddings par lots (batch_size=50) pour éviter les timeouts API
    4. Sauvegarde l'index sur disque pour persistance

    Args:
        chunks  (list[Document]): Chunks de texte à vectoriser et indexer.
        rebuild (bool):           Si True, supprime l'index existant avant reconstruction.
                                  Si False, recrée un index depuis zéro.

    Returns:
        FAISS: Instance de la base vectorielle construite et persistée.

    Raises:
        ValueError: Si MISTRAL_API_KEY n'est pas définie dans le fichier .env.
    """
    if not MISTRAL_API_KEY:
        raise ValueError(
            "MISTRAL_API_KEY non définie. "
            "Ajoutez-la dans votre fichier .env"
        )

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY,
    )
    logger.info("Modèle d'embedding : mistral-embed (1024 dimensions)")

    if rebuild and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
        logger.info("Index existant supprimé (mode --rebuild)")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Indexation par lots pour éviter les timeouts API Mistral
    batch_size  = 50
    logger.info(f"Construction de l'index FAISS avec {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks[:batch_size], embeddings)

    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        logger.info(f"  Indexation chunks {i}–{i + len(batch)}…")
        vectorstore.add_documents(batch)

    # Persistance sur disque
    vectorstore.save_local(str(INDEX_DIR))
    logger.info(f"Index sauvegardé : {INDEX_DIR}")
    return vectorstore


def verify_index(vectorstore: FAISS) -> None:
    """
    Vérifie le bon fonctionnement de l'index avec une requête de test.

    Effectue une recherche par similarité cosinus sur une requête de test
    et affiche les 3 premiers résultats. Permet de valider que l'index
    est correctement construit et interrogeable.

    Args:
        vectorstore (FAISS): Instance de l'index FAISS à tester.
    """
    test_query = "concert de musique à Paris"
    results    = vectorstore.similarity_search(test_query, k=3)
    logger.info(f"Test de recherche ('{test_query}') : {len(results)} résultats")
    for i, doc in enumerate(results, 1):
        title = doc.metadata.get("title", "Sans titre")
        city  = doc.metadata.get("city", "")
        logger.info(f"  [{i}] {title} — {city}")


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Point d'entrée principal du script de vectorisation.

    Exécute le pipeline complet : chargement → conversion → chunking →
    embedding → indexation FAISS → vérification.
    """
    parser = argparse.ArgumentParser(
        description="Vectorisation et indexation FAISS des événements Puls-Events"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Supprime et reconstruit complètement l'index FAISS",
    )
    args = parser.parse_args()

    logger.info("=== Démarrage de la vectorisation ===")
    events    = load_events(DATA_FILE)
    documents = events_to_documents(events)
    chunks    = split_documents(documents)
    vs        = build_faiss_index(chunks, rebuild=args.rebuild)
    verify_index(vs)
    logger.info("=== Vectorisation terminée ===")


if __name__ == "__main__":
    main()
