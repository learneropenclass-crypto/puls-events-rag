import os, json, logging, argparse, shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 50))
BASE_DIR        = Path(__file__).parent.parent
DATA_FILE       = BASE_DIR / "data" / "events_clean.json"
INDEX_DIR       = Path("C:/rag_index")

def load_events(path):
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    logger.info(f"Evenements charges : {len(events)}")
    return events

def events_to_documents(events):
    docs = []
    for e in events:
        docs.append(Document(
            page_content=e.get("text", ""),
            metadata={"uid": e.get("uid",""), "title": e.get("title",""),
                      "city": e.get("city",""), "department": e.get("department",""),
                      "region": e.get("region",""), "venue": e.get("venue",""),
                      "date_start": e.get("date_start",""), "date_end": e.get("date_end",""),
                      "categories": e.get("categories","")}
        ))
    logger.info(f"Documents crees : {len(docs)}")
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(documents)
    logger.info(f"Chunking : {len(documents)} docs -> {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks, rebuild=False):
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY non definie dans .env")
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=MISTRAL_API_KEY)
    if rebuild and INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Construction index FAISS avec {len(chunks)} chunks...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    logger.info(f"Index sauvegarde : {INDEX_DIR}")
    return vectorstore

def verify_index(vectorstore):
    results = vectorstore.similarity_search("concert musique Paris", k=3)
    logger.info(f"Test recherche : {len(results)} resultats")
    for i, doc in enumerate(results, 1):
        logger.info(f"  [{i}] {doc.metadata.get('title','?')} - {doc.metadata.get('city','')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    logger.info("=== Demarrage vectorisation ===")
    events = load_events(DATA_FILE)
    docs   = events_to_documents(events)
    chunks = split_documents(docs)
    vs     = build_faiss_index(chunks, rebuild=args.rebuild)
    verify_index(vs)
    logger.info("=== Vectorisation terminee ===")

if __name__ == "__main__":
    main()
