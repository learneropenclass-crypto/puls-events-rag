# 🎭 Puls-Events RAG — Système de recommandation d'événements culturels

## Présentation

Ce projet est un **Proof of Concept (POC)** d'un système **RAG (Retrieval-Augmented Generation)** développé pour l'entreprise **Puls-Events**.

Il permet à un chatbot intelligent de répondre à des questions en langage naturel sur les événements culturels d'**Île-de-France**, en s'appuyant exclusivement sur des données réelles issues de la plateforme **Open Agenda**.

Le système combine trois technologies clés :
- **LangChain** — orchestration de la chaîne RAG (LCEL)
- **Mistral AI** — embeddings (`mistral-embed`) et génération (`mistral-large-latest`)
- **FAISS** — indexation et recherche vectorielle rapide

---

## Objectifs du projet

| Objectif | Description |
|---|---|
| **Faisabilité technique** | Démontrer qu'un système RAG peut recommander des événements culturels avec précision |
| **Qualité des données** | Garantir que seuls les événements IDF de moins d'un an sont indexés |
| **Reproductibilité** | Permettre à n'importe quel développeur de reconstruire le système en suivant ce README |
| **Évaluation** | Mesurer objectivement la qualité RAG grâce à un jeu de données Q/R annoté |

---

## Architecture du système

```
Open Agenda API
      │
      ▼
[fetch_events.py]  ──► Filtrage géo (IDF) + temporel (< 1 an) + nettoyage
      │
      ▼
[vectorize.py]     ──► Chunking (500 tokens) + Embeddings Mistral + Index FAISS
      │
      ▼
[FAISS Index]      ──► Base vectorielle persistante sur disque
      │
      ▼
[chatbot.py]       ──► LangChain LCEL + Mistral Large → Réponse augmentée
```

---

## Prérequis

- **Python 3.10 ou supérieur**
- **Clé API Mistral** : créer un compte sur [https://console.mistral.ai/](https://console.mistral.ai/) et générer une clé API
- **Accès Internet** (pour l'API Open Agenda et les appels Mistral)

---

## Installation

### 1. Cloner ou dézipper le projet

```bash
cd puls_events_rag
```

### 2. Créer un environnement virtuel

```bash
# Création
python -m venv venv

# Activation — Linux/macOS
source venv/bin/activate

# Activation — Windows (PowerShell)
venv\Scripts\activate
```

> Le prompt `(venv)` doit apparaître dans votre terminal.

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Configurer la clé API

```bash
# Linux/macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Ouvrir le fichier `.env` et renseigner vos propres clés :

```
MISTRAL_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENAGENDA_API_KEY=votre_clé_openagenda
INDEX_DIR=C:/rag_index
```

> `INDEX_DIR` est optionnel : si non défini, `C:/rag_index` est utilisé par défaut (recommandé sous Windows pour éviter les problèmes d'encodage liés aux accents dans les chemins).

---

## Utilisation

Exécuter les trois scripts **dans l'ordre** :

### Étape 1 — Récupérer et nettoyer les données

```bash
python scripts/fetch_events.py
```

- Appelle l'API Open Agenda avec filtres géographiques (IDF) et temporels (12 mois)
- En cas d'indisponibilité de l'API, bascule sur 15 événements de démonstration
- **Sortie** : `data/events_clean.json`

### Étape 2 — Construire la base vectorielle FAISS

```bash
python scripts/vectorize.py
```

- Découpe les événements en chunks de 500 tokens
- Génère les embeddings via `mistral-embed` par lots de 50
- Indexe dans FAISS et persiste sur disque
- **Sortie** : `vector_store/faiss_index/`

### Étape 3 — Lancer le chatbot RAG

```bash
# Interface ligne de commande
python scripts/chatbot.py

# Interface web (navigateur sur http://localhost:5000)
pip install flask
python scripts/app.py
```

### Reconstruire l'index FAISS

```bash
python scripts/vectorize.py --rebuild
```

---

## Tests unitaires

```bash
python -m pytest tests/ -v
```

La suite de tests valide automatiquement :

| Classe de tests | Ce qui est vérifié |
|---|---|
| `TestTemporalValidation` | Événements < 1 an, dates valides et cohérentes (5 tests) |
| `TestGeographicValidation` | Filtrage IDF sur les 8 départements (4 tests) |
| `TestDataIntegrity` | Champs requis, UIDs uniques, textes non vides (5 tests) |
| `TestChunkingPipeline` | Documents LangChain, taille des chunks (3 tests) |
| `TestFAISSIndex` | Recherche, TOP_K, métadonnées (3 tests) |
| `TestDataFileIntegration` | Validation sur données réelles si disponibles (3 tests) |

**Résultat attendu : 23/23 tests passent.**

---

## Structure du projet

```
puls_events_rag/
│
├── README.md                    # Ce fichier — documentation complète
├── requirements.txt             # Dépendances pip
├── .env.example                 # Modèle de configuration (à copier en .env)
│
├── scripts/                     # Scripts Python du pipeline
│   ├── fetch_events.py          # Récupération et nettoyage des données Open Agenda
│   ├── vectorize.py             # Vectorisation et indexation FAISS
│   ├── chatbot.py               # Chatbot RAG en ligne de commande
│   └── app.py                   # Interface web Flask (optionnel)
│
├── tests/                       # Tests unitaires pytest
│   └── test_pipeline.py         # 23 tests — validation temporelle, géo, intégrité, chunking
│
├── data/                        # Données (générées automatiquement)
│   ├── events_clean.json        # Événements nettoyés produits par fetch_events.py
│   └── qa_dataset.json          # Jeu de données Q/R annoté pour l'évaluation
│
└── vector_store/                # Index vectoriel (généré automatiquement)
    └── faiss_index/
        ├── index.faiss          # Index FAISS binaire
        └── index.pkl            # Métadonnées associées à l'index
```

> Les dossiers `data/` et `vector_store/` sont créés automatiquement à l'exécution des scripts.

---

## Paramètres configurables (.env)

| Variable | Description | Valeur par défaut |
|---|---|---|
| `MISTRAL_API_KEY` | Clé API Mistral **(obligatoire)** | — |
| `OPENAGENDA_API_KEY` | Clé API Open Agenda **(obligatoire)** | — |
| `INDEX_DIR` | Chemin de l'index FAISS (lu par `vectorize.py`, `chatbot.py` et `app.py`) | `C:/rag_index` |
| `GEO_REGION` | Région géographique cible | `Île-de-France` |
| `MAX_EVENTS` | Nombre maximum d'événements à récupérer | `500` |
| `CHUNK_SIZE` | Taille des chunks en tokens | `500` |
| `CHUNK_OVERLAP` | Chevauchement entre chunks en tokens | `50` |
| `TOP_K` | Nombre de documents récupérés par FAISS | `5` |

---

## Dépendances principales

| Package | Version | Rôle |
|---|---|---|
| `langchain-mistralai` | ≥0.1.0 | Intégration Mistral (LLM + Embeddings) |
| `langchain-text-splitters` | ≥1.1.0 | Découpage en chunks |
| `langchain-community` | ≥0.0.20 | Intégration FAISS |
| `langchain-core` | ≥1.0.0 | Composants de base LangChain (LCEL) |
| `faiss-cpu` | ≥1.7.4 | Base vectorielle (CPU) |
| `mistralai` | ≥0.4.0 | Client API Mistral |
| `requests` | ≥2.31.0 | Appels API Open Agenda |
| `python-dotenv` | ≥1.0.0 | Gestion des variables d'environnement |
| `pytest` | ≥7.0.0 | Tests unitaires |
| `flask` | ≥3.0.0 | Interface web (optionnel) |

---

## Résolution de problèmes fréquents

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | Vérifier que le venv est activé : `(venv)` doit apparaître dans le terminal |
| `401 Unauthorized` (Mistral) | Vérifier la clé API dans `.env` — pas d'espaces, pas de guillemets |
| Erreur chemin FAISS (Windows) | Créer manuellement `vector_store/faiss_index/` ou utiliser `--rebuild` |
| Encodage `.env` corrompu | Recréer avec `Set-Content -Path ".env" -Encoding UTF8` (PowerShell) |
| API Open Agenda indisponible | Le script bascule automatiquement sur les données de démonstration |
| `langchain.chains` introuvable | Utiliser `chatbot.py` avec LangChain LCEL (version fournie) |

---

## Auteur

Projet réalisé dans le cadre du POC Puls-Events — **Ingénieur Data Freelance**  
Technologies : Python 3.11 · LangChain LCEL · Mistral AI · FAISS · Open Agenda API
 

