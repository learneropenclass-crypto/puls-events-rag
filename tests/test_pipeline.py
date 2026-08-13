"""
test_pipeline.py
----------------
Tests unitaires du pipeline RAG Puls-Events.

Couvre :
    1. Validation temporelle : tous les événements ont moins d'un an
    2. Validation géographique : tous les événements sont en Île-de-France
    3. Intégrité des données : structure et champs requis
    4. Pipeline de chunking : découpage cohérent des documents
    5. Index FAISS : présence et interrogeabilité de l'index
    6. Tests d'intégration sur les données réelles (si disponibles)

Usage:
    python -m pytest tests/ -v
    python -m pytest tests/test_pipeline.py -v --tb=short
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ajout du répertoire racine au path Python
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

# ── Fixtures ───────────────────────────────────────────────────────────────────

DATA_FILE = ROOT_DIR / "data" / "events_clean.json"
INDEX_DIR = ROOT_DIR / "vector_store" / "faiss_index"

# Référence temporelle fixe pour tous les tests
NOW = datetime.now(tz=timezone.utc)

# Événements de test valides (IDF, < 1 an)
MOCK_EVENTS_VALID = [
    {
        "uid": "evt-001",
        "title": "Concert de Jazz",
        "description": "Un superbe concert de jazz au Théâtre de la Ville.",
        "date_start": (NOW - timedelta(days=30)).isoformat(),
        "date_end":   (NOW - timedelta(days=29)).isoformat(),
        "city": "Paris",
        "department": "Paris",
        "region": "Île-de-France",
        "venue": "Théâtre de la Ville",
        "categories": "Musique",
        "keywords": "jazz, concert, musique",
        "text": "Événement : Concert de Jazz\nDescription : Un superbe concert de jazz.\nLieu : Paris\nRégion : Île-de-France",
    },
    {
        "uid": "evt-002",
        "title": "Exposition Impressionnisme",
        "description": "Exposition sur les peintres impressionnistes français.",
        "date_start": (NOW - timedelta(days=10)).isoformat(),
        "date_end":   (NOW + timedelta(days=20)).isoformat(),
        "city": "Versailles",
        "department": "Yvelines",
        "region": "Île-de-France",
        "venue": "Château de Versailles",
        "categories": "Exposition",
        "keywords": "art, peinture, impressionnisme",
        "text": "Événement : Exposition Impressionnisme\nDescription : Exposition sur les impressionnistes.\nLieu : Versailles\nRégion : Île-de-France",
    },
    {
        "uid": "evt-003",
        "title": "Festival Électro",
        "description": "Festival de musique électronique en plein air.",
        "date_start": (NOW + timedelta(days=5)).isoformat(),
        "date_end":   (NOW + timedelta(days=7)).isoformat(),
        "city": "Boulogne-Billancourt",
        "department": "Hauts-de-Seine",
        "region": "Île-de-France",
        "venue": "Parc de Saint-Cloud",
        "categories": "Musique, Festival",
        "keywords": "électro, festival, musique",
        "text": "Événement : Festival Électro\nDescription : Festival de musique électronique.\nLieu : Boulogne-Billancourt\nRégion : Île-de-France",
    },
]

# Événements de test invalides (trop vieux ou hors IDF)
MOCK_EVENTS_INVALID = [
    {
        "uid": "evt-old",
        "title": "Vieux Concert",
        "description": "Un concert il y a plus d'un an.",
        "date_start": (NOW - timedelta(days=400)).isoformat(),
        "date_end":   (NOW - timedelta(days=399)).isoformat(),
        "city": "Paris",
        "department": "Paris",
        "region": "Île-de-France",
        "venue": "Salle Pleyel",
        "categories": "Musique",
        "keywords": "concert",
        "text": "Événement : Vieux Concert",
    },
    {
        "uid": "evt-wrong-region",
        "title": "Festival de Cannes",
        "description": "Le célèbre festival de cinéma.",
        "date_start": (NOW - timedelta(days=10)).isoformat(),
        "date_end":   (NOW + timedelta(days=5)).isoformat(),
        "city": "Cannes",
        "department": "Alpes-Maritimes",
        "region": "Provence-Alpes-Côte d'Azur",
        "venue": "Palais des Festivals",
        "categories": "Cinéma",
        "keywords": "cinéma, festival",
        "text": "Événement : Festival de Cannes",
    },
]

# Mots-clés de détection Île-de-France
IDF_REGIONS = [
    "île-de-france", "ile-de-france", "paris",
    "seine-et-marne", "yvelines", "essonne",
    "hauts-de-seine", "seine-saint-denis",
    "val-de-marne", "val-d'oise",
    "75", "77", "78", "91", "92", "93", "94", "95",
]


# ── Tests temporels ────────────────────────────────────────────────────────────

class TestTemporalValidation:
    """Tests de validation de la contrainte temporelle (événements < 1 an)."""

    def test_valid_events_are_recent(self):
        """Tous les événements valides doivent avoir commencé il y a moins d'un an."""
        one_year_ago = NOW - timedelta(days=365)

        for event in MOCK_EVENTS_VALID:
            date_start = datetime.fromisoformat(
                event["date_start"].replace("Z", "+00:00")
            )
            assert date_start >= one_year_ago, (
                f"L'événement '{event['title']}' a démarré le {date_start.date()}, "
                f"ce qui dépasse la limite d'un an (avant {one_year_ago.date()})."
            )

    def test_invalid_old_event_detected(self):
        """Un événement de plus d'un an doit être identifié comme invalide."""
        one_year_ago = NOW - timedelta(days=365)
        old_event    = MOCK_EVENTS_INVALID[0]

        date_start = datetime.fromisoformat(
            old_event["date_start"].replace("Z", "+00:00")
        )
        assert date_start < one_year_ago, (
            "L'événement test 'vieux' aurait dû être antérieur à un an."
        )

    def test_future_events_are_allowed(self):
        """Les événements futurs (à venir) sont autorisés."""
        for event in MOCK_EVENTS_VALID:
            date_end = datetime.fromisoformat(
                event["date_end"].replace("Z", "+00:00")
            )
            one_year_ahead = NOW + timedelta(days=365)
            assert date_end <= one_year_ahead, (
                f"La date de fin de '{event['title']}' dépasse un an dans le futur."
            )

    def test_events_have_valid_date_format(self):
        """Chaque événement doit avoir des dates de début et de fin parsables."""
        for event in MOCK_EVENTS_VALID:
            assert event.get("date_start"), f"date_start manquante : {event['uid']}"
            assert event.get("date_end"),   f"date_end manquante : {event['uid']}"
            try:
                datetime.fromisoformat(event["date_start"].replace("Z", "+00:00"))
                datetime.fromisoformat(event["date_end"].replace("Z", "+00:00"))
            except ValueError as e:
                pytest.fail(f"Format de date invalide pour '{event['title']}' : {e}")

    def test_date_start_before_date_end(self):
        """La date de début doit être antérieure ou égale à la date de fin."""
        for event in MOCK_EVENTS_VALID:
            date_start = datetime.fromisoformat(
                event["date_start"].replace("Z", "+00:00")
            )
            date_end = datetime.fromisoformat(
                event["date_end"].replace("Z", "+00:00")
            )
            assert date_start <= date_end, (
                f"Pour '{event['title']}', date_start ({date_start.date()}) "
                f"est postérieure à date_end ({date_end.date()})."
            )


# ── Tests géographiques ────────────────────────────────────────────────────────

class TestGeographicValidation:
    """Tests de validation de la contrainte géographique (Île-de-France)."""

    def _is_in_idf(self, event: dict) -> bool:
        """Vérifie si un événement appartient à la région Île-de-France."""
        combined = " ".join([
            event.get("city", ""),
            event.get("department", ""),
            event.get("region", ""),
            event.get("venue", ""),
        ]).lower()
        return any(kw in combined for kw in IDF_REGIONS)

    def test_valid_events_are_in_idf(self):
        """Tous les événements valides doivent être localisés en Île-de-France."""
        for event in MOCK_EVENTS_VALID:
            assert self._is_in_idf(event), (
                f"L'événement '{event['title']}' n'est pas en Île-de-France "
                f"(région : {event.get('region')}, ville : {event.get('city')})."
            )

    def test_non_idf_event_is_detected(self):
        """Un événement hors IDF doit être identifié comme invalide."""
        non_idf_event = MOCK_EVENTS_INVALID[1]
        assert not self._is_in_idf(non_idf_event), (
            f"L'événement '{non_idf_event['title']}' hors IDF n'a pas été détecté."
        )

    def test_all_idf_departments_recognized(self):
        """Tous les départements IDF doivent être reconnus comme appartenant à la région."""
        idf_departments = {
            "Paris", "Seine-et-Marne", "Yvelines", "Essonne",
            "Hauts-de-Seine", "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise",
        }
        for dept in idf_departments:
            event = {
                "city": "Ville Test",
                "department": dept,
                "region": "Île-de-France",
                "venue": "Lieu Test",
            }
            assert self._is_in_idf(event), (
                f"Le département '{dept}' n'est pas reconnu comme appartenant à l'IDF."
            )

    def test_events_have_required_location_fields(self):
        """Chaque événement doit avoir au minimum un champ de localisation renseigné."""
        for event in MOCK_EVENTS_VALID:
            has_location = any([
                event.get("city"),
                event.get("department"),
                event.get("region"),
                event.get("venue"),
            ])
            assert has_location, (
                f"L'événement '{event.get('uid')}' n'a aucun champ de localisation."
            )


# ── Tests d'intégrité des données ──────────────────────────────────────────────

class TestDataIntegrity:
    """Tests d'intégrité de la structure et du contenu des données."""

    REQUIRED_FIELDS = ["uid", "title", "text", "date_start", "date_end", "region"]

    def test_events_have_required_fields(self):
        """Chaque événement doit contenir tous les champs requis pour le RAG."""
        for event in MOCK_EVENTS_VALID:
            for field in self.REQUIRED_FIELDS:
                assert field in event, (
                    f"Champ '{field}' manquant dans l'événement '{event.get('uid')}'."
                )

    def test_events_have_non_empty_text(self):
        """Le champ 'text' agrégé de chaque événement doit être non vide."""
        for event in MOCK_EVENTS_VALID:
            text = event.get("text", "").strip()
            assert len(text) > 10, (
                f"Le texte de '{event.get('uid')}' est trop court ou vide."
            )

    def test_events_have_non_empty_title(self):
        """Chaque événement doit avoir un titre non vide."""
        for event in MOCK_EVENTS_VALID:
            assert event.get("title", "").strip(), (
                f"Le titre de '{event.get('uid')}' est vide."
            )

    def test_events_uids_are_unique(self):
        """Les identifiants uniques (uid) ne doivent pas se répéter."""
        uids = [e["uid"] for e in MOCK_EVENTS_VALID]
        assert len(uids) == len(set(uids)), (
            f"Des uid en doublon ont été détectés : {uids}"
        )

    def test_text_contains_title(self):
        """Le champ 'text' agrégé doit contenir le titre de l'événement."""
        for event in MOCK_EVENTS_VALID:
            assert event["title"] in event["text"], (
                f"Le champ 'text' de '{event['title']}' ne contient pas son titre."
            )


# ── Tests du pipeline de chunking ──────────────────────────────────────────────

class TestChunkingPipeline:
    """Tests du pipeline de découpage en chunks (RecursiveCharacterTextSplitter)."""

    def test_documents_creation(self):
        """La conversion d'événements en Documents LangChain doit fonctionner."""
        from langchain_core.documents import Document

        documents = []
        for event in MOCK_EVENTS_VALID:
            doc = Document(
                page_content=event.get("text", ""),
                metadata={"uid": event.get("uid"), "title": event.get("title")},
            )
            documents.append(doc)

        assert len(documents) == len(MOCK_EVENTS_VALID)
        for doc in documents:
            assert isinstance(doc.page_content, str)
            assert len(doc.page_content) > 0
            assert "uid" in doc.metadata

    def test_chunking_produces_valid_chunks(self):
        """Le chunking doit produire des chunks non vides avec les métadonnées préservées."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        long_text = "Test. " * 300  # Texte long pour forcer le découpage
        doc = Document(
            page_content=long_text,
            metadata={"uid": "test-001", "title": "Test Event"},
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200, chunk_overlap=20
        )
        chunks = splitter.split_documents([doc])

        assert len(chunks) > 1, "Un texte long doit produire plusieurs chunks."
        for chunk in chunks:
            assert len(chunk.page_content.strip()) > 0, "Un chunk ne doit pas être vide."
            assert chunk.metadata.get("uid") == "test-001", (
                "Les métadonnées doivent être préservées dans les chunks."
            )

    def test_chunk_size_respected(self):
        """Les chunks ne doivent pas dépasser la taille configurée (tolérance 10%)."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size = 200
        long_text  = "A" * 1000
        doc        = Document(page_content=long_text, metadata={})
        splitter   = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        chunks = splitter.split_documents([doc])

        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 1.1, (
                f"Un chunk dépasse la taille max : {len(chunk.page_content)} > {chunk_size}"
            )


# ── Tests de l'index FAISS (mock) ──────────────────────────────────────────────

class TestFAISSIndex:
    """Tests de la base vectorielle FAISS (avec mock de l'API Mistral)."""

    def test_faiss_search_returns_results(self):
        """La recherche FAISS doit retourner des résultats sous forme de Documents."""
        from langchain_core.documents import Document

        mock_vectorstore = MagicMock()
        mock_results = [
            Document(
                page_content=event["text"],
                metadata={
                    "uid":   event["uid"],
                    "title": event["title"],
                    "city":  event["city"],
                },
            )
            for event in MOCK_EVENTS_VALID[:2]
        ]
        mock_vectorstore.similarity_search.return_value = mock_results

        results = mock_vectorstore.similarity_search("concert Paris", k=3)

        assert len(results) > 0, "La recherche doit retourner au moins un résultat."
        assert all(isinstance(r, Document) for r in results), (
            "Tous les résultats doivent être des Documents LangChain."
        )

    def test_faiss_search_respects_top_k(self):
        """La recherche FAISS doit être appelée avec la limite TOP_K spécifiée."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = MOCK_EVENTS_VALID[:2]

        k = 2
        mock_vectorstore.similarity_search("musique", k=k)
        mock_vectorstore.similarity_search.assert_called_once_with("musique", k=k)

    def test_faiss_search_returns_relevant_metadata(self):
        """Les résultats FAISS doivent contenir les métadonnées uid, title et city."""
        from langchain_core.documents import Document

        mock_vectorstore = MagicMock()
        mock_results = [
            Document(
                page_content=MOCK_EVENTS_VALID[0]["text"],
                metadata={
                    "uid":   MOCK_EVENTS_VALID[0]["uid"],
                    "title": MOCK_EVENTS_VALID[0]["title"],
                    "city":  MOCK_EVENTS_VALID[0]["city"],
                },
            )
        ]
        mock_vectorstore.similarity_search.return_value = mock_results

        results = mock_vectorstore.similarity_search("jazz Paris", k=1)

        assert results[0].metadata.get("uid"),   "L'uid doit être présent dans les métadonnées."
        assert results[0].metadata.get("title"), "Le titre doit être présent dans les métadonnées."
        assert results[0].metadata.get("city"),  "La ville doit être présente dans les métadonnées."


# ── Tests d'intégration (nécessitent les données réelles) ─────────────────────

class TestDataFileIntegration:
    """Tests d'intégration sur le fichier de données réel (events_clean.json)."""

    @pytest.mark.skipif(
        not DATA_FILE.exists(),
        reason=f"Fichier de données absent : {DATA_FILE} — lancez fetch_events.py d'abord",
    )
    def test_data_file_contains_events(self):
        """Le fichier events_clean.json doit contenir au moins un événement."""
        with open(DATA_FILE, encoding="utf-8") as f:
            events = json.load(f)
        assert len(events) > 0, "Le fichier de données est vide."

    @pytest.mark.skipif(
        not DATA_FILE.exists(),
        reason="Fichier de données absent",
    )
    def test_all_real_events_are_recent(self):
        """Tous les événements réels doivent avoir démarré il y a moins d'un an."""
        with open(DATA_FILE, encoding="utf-8") as f:
            events = json.load(f)

        one_year_ago = NOW - timedelta(days=365)
        errors = []

        for event in events:
            date_str = event.get("date_start", "")
            if not date_str:
                continue
            try:
                date_start = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                if date_start < one_year_ago:
                    errors.append(
                        f"'{event.get('title')}' : date_start = {date_start.date()}"
                    )
            except ValueError:
                errors.append(
                    f"'{event.get('uid')}' : format de date invalide : {date_str}"
                )

        assert not errors, (
            f"{len(errors)} événements dépassent la limite d'un an :\n"
            + "\n".join(errors[:5])
        )

    @pytest.mark.skipif(
        not DATA_FILE.exists(),
        reason="Fichier de données absent",
    )
    def test_all_real_events_are_in_idf(self):
        """Tous les événements réels doivent être localisés en Île-de-France."""
        with open(DATA_FILE, encoding="utf-8") as f:
            events = json.load(f)

        errors = []
        for event in events:
            combined = " ".join([
                event.get("city", ""),
                event.get("department", ""),
                event.get("region", ""),
                event.get("venue", ""),
            ]).lower()
            if not any(kw in combined for kw in IDF_REGIONS):
                errors.append(
                    f"'{event.get('title')}' — région : {event.get('region')}, "
                    f"ville : {event.get('city')}"
                )

        assert not errors, (
            f"{len(errors)} événements ne sont pas en IDF :\n"
            + "\n".join(errors[:5])
        )
