# 🎭 Puls-Events RAG — Système de recommandation d'événements culturels

## Description

Proof of Concept (POC) d'un système **RAG (Retrieval-Augmented Generation)** pour recommander des événements culturels issus de la plateforme Open Agenda.  
Le système utilise **LangChain**, **Mistral AI** et **FAISS** pour répondre à des questions en langage naturel sur les événements culturels de la région Île-de-France.

---

## Architecture

```
Open Agenda API
      │
      ▼
[Pré-processing]  ──► Nettoyage, filtrage géo + temporel
      │
      ▼
[Vectorisation]   ──► Découpage en chunks + embeddings Mistral
      │
      ▼
[Base FAISS]      ──► Index vectoriel persistant
      │
      ▼
[Chatbot RAG]     ──► LangChain + Mistral LLM → Réponse augmentée
```

---

## Prérequis

- Python 3.10+
- Une clé API Mistral : [https://console.mistral.ai/](https://console.mistral.ai/)
- Accès Internet (pour l'API Open Agenda)

---

## Installation

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd puls_events_rag
```

### 2. Créer un environnement virtuel

```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# Éditer .env et renseigner votre clé MISTRAL_API_KEY
```

---

## Utilisation

### Étape 1 — Récupérer et nettoyer les données

```bash
python scripts/fetch_events.py
```

Télécharge les événements d'Île-de-France des 12 derniers mois depuis Open Agenda.  
Résultat : `data/events_clean.json`

### Étape 2 — Construire la base vectorielle FAISS

```bash
python scripts/vectorize.py
```

Découpe les descriptions en chunks, génère les embeddings Mistral, indexe dans FAISS.  
Résultat : `vector_store/faiss_index/`

### Étape 3 — Lancer le chatbot RAG

```bash
python scripts/chatbot.py
```

Interface interactive en ligne de commande pour interroger le système.

### Reconstruire la base vectorielle (optionnel)

```bash
python scripts/vectorize.py --rebuild
```

---

## Tests unitaires

```bash
python -m pytest tests/ -v
```

Vérifie que :
- Tous les événements ont moins d'un an
- Tous les événements appartiennent à la région Île-de-France
- La base FAISS est bien construite et interrogeable

---

## Structure du projet

```
puls_events_rag/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── events_clean.json        # Données nettoyées (généré)
├── vector_store/
│   └── faiss_index/             # Index FAISS persistant (généré)
├── scripts/
│   ├── fetch_events.py          # Récupération et nettoyage des données
│   ├── vectorize.py             # Vectorisation et indexation FAISS
│   └── chatbot.py               # Chatbot RAG interactif
└── tests/
    └── test_pipeline.py         # Tests unitaires
```

---

## Paramètres configurables (.env)

| Variable | Description | Défaut |
|---|---|---|
| `MISTRAL_API_KEY` | Clé API Mistral (obligatoire) | — |
| `GEO_REGION` | Région géographique cible | `Île-de-France` |
| `MAX_EVENTS` | Nombre max d'événements à récupérer | `500` |
| `CHUNK_SIZE` | Taille des chunks (tokens) | `500` |
| `CHUNK_OVERLAP` | Chevauchement entre chunks | `50` |
| `TOP_K` | Nombre de documents récupérés | `5` |

---

## Dépendances principales

| Package | Version | Rôle |
|---|---|---|
| `langchain` | ≥0.1.0 | Orchestration RAG |
| `langchain-mistralai` | ≥0.1.0 | Intégration Mistral |
| `faiss-cpu` | ≥1.7.4 | Base vectorielle |
| `mistralai` | ≥0.4.0 | Client API Mistral |
| `requests` | ≥2.31.0 | Appels API Open Agenda |
| `python-dotenv` | ≥1.0.0 | Gestion des variables d'env |
| `pytest` | ≥7.0.0 | Tests unitaires |

---

## Auteur

Projet réalisé dans le cadre du POC Puls-Events — Ingénieur Data Freelance
