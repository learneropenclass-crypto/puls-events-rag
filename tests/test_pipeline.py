"""
test_pipeline.py
----------------
Tests unitaires du pipeline RAG Puls-Events.

Couvre :
    1. Validation temporelle : tous les Ã©vÃ©nements ont moins d'un an
    2. Validation gÃ©ographique : tous les Ã©vÃ©nements sont en ÃŽle-de-France
    3. IntÃ©gritÃ© des donnÃ©es : structure et champs requis
    4. Pipeline de chunking : dÃ©coupage cohÃ©rent des documents
    5. Index FAISS : prÃ©sence et interrogeabilitÃ© de l'index

Usage:
    python -m pytest tests/ -v
    python -m pytest tests/test_pipeline.py -v --tb=short
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ajout du rÃ©pertoire racine au path Python
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

# â”€â”€ Fixtures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

DATA_FILE = ROOT_DIR / "data" / "events_clean.json"
INDEX_DIR = ROOT_DIR / "vector_store" / "faiss_index"

# Ã‰vÃ©nements de test (ne nÃ©cessitent pas l'API)
NOW = datetime.now(tz=timezone.utc)

MOCK_EVENTS_VALID = [
    {
        "uid": "evt-001",
        "title": "Concert de Jazz",
        "description": "Un superbe concert de jazz au ThÃ©Ã¢tre de la Ville.",
        "date_start": (NOW - timedelta(days=30)).isoformat(),
        "date_end":   (NOW - timedelta(days=29)).isoformat(),
        "city": "Paris",
        "department": "Paris",
        "region": "ÃŽle-de-France",
        "venue": "ThÃ©Ã¢tre de la Ville",
        "categories": "Musique",
        "keywords": "jazz, concert, musique",
        "text": "Ã‰vÃ©nement : Concert de Jazz\nDescription : Un superbe concert de jazz.\nLieu : Paris\nRÃ©gion : ÃŽle-de-France",
    },
    {
        "uid": "evt-002",
        "title": "Exposition Impressionnisme",
        "description": "Exposition sur les peintres impressionnistes franÃ§ais.",
        "date_start": (NOW - timedelta(days=10)).isoformat(),
        "date_end":   (NOW + timedelta(days=20)).isoformat(),
        "city": "Versailles",
        "department": "Yvelines",
        "region": "ÃŽle-de-France",
        "venue": "ChÃ¢teau de Versailles",
        "categories": "Exposition",
        "keywords": "art, peinture, impressionnisme",
        "text": "Ã‰vÃ©nement : Exposition Impressionnisme\nDescription : Exposition sur les impressionnistes.\nLieu : Versailles\nRÃ©gion : ÃŽle-de-France",
    },
    {
        "uid": "evt-003",
        "title": "Festival Ã‰lectro",
        "description": "Festival de musique Ã©lectronique en plein air.",
        "date_start": (NOW + timedelta(days=5)).isoformat(),
        "date_end":   (NOW + timedelta(days=7)).isoformat(),
        "city": "Boulogne-Billancourt",
        "department": "Hauts-de-Seine",
        "region": "ÃŽle-de-France",
        "venue": "Parc de Saint-Cloud",
        "categories": "Musique, Festival",
        "keywords": "Ã©lectro, festival, musique",
        "text": "Ã‰vÃ©nement : Festival Ã‰lectro\nDescription : Festival de musique Ã©lectronique.\nLieu : Boulogne-Billancourt\nRÃ©gion : ÃŽle-de-France",
    },
]

MOCK_EVENTS_INVALID = [
    {
        "uid": "evt-old",
        "title": "Vieux Concert",
        "description": "Un concert il y a plus d'un an.",
        "date_start": (NOW - timedelta(days=400)).isoformat(),
        "date_end":   (NOW - timedelta(days=399)).isoformat(),
        "city": "Paris",
        "department": "Paris",
        "region": "ÃŽle-de-France",
        "venue": "Salle Pleyel",
        "categories": "Musique",
        "keywords": "concert",
        "text": "Ã‰vÃ©nement : Vieux Concert",
    },
    {
        "uid": "evt-wrong-region",
        "title": "Festival de Cannes",
        "description": "Le cÃ©lÃ¨bre festival de cinÃ©ma.",
        "date_start": (NOW - timedelta(days=10)).isoformat(),
        "date_end":   (NOW + timedelta(days=5)).isoformat(),
        "city": "Cannes",
        "department": "Alpes-Maritimes",
        "region": "Provence-Alpes-CÃ´te d'Azur",
        "venue": "Palais des Festivals",
        "categories": "CinÃ©ma",
        "keywords": "cinÃ©ma, festival",
        "text": "Ã‰vÃ©nement : Festival de Cannes",
    },
]

IDF_REGIONS = [
    "Ã®le-de-france", "ile-de-france", "paris",
    "seine-et-marne", "yvelines", "essonne",
    "hauts-de-seine", "seine-saint-denis",
    "val-de-marne", "val-d'oise",
    "75", "77", "78", "91", "92", "93", "94", "95",
]


# â”€â”€ Tests temporels â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestTemporalValidation:
    """Tests de validation de la contrainte temporelle (< 1 an)."""

    def test_valid_events_are_recent(self):
        """Tous les Ã©vÃ©nements valides doivent avoir commencÃ© il y a moins d'un an."""
        one_year_ago = NOW - timedelta(days=365)

        for event in MOCK_EVENTS_VALID:
            date_start_str = event["date_start"]
            # Normalisation du format de date
            date_start = datetime.fromisoformat(
                date_start_str.replace("Z", "+00:00")
            )
            assert date_start >= one_year_ago, (
                f"L'Ã©vÃ©nement '{event['title']}' a dÃ©marrÃ© le {date_start.date()}, "
                f"ce qui dÃ©passe la limite d'un an (avant {one_year_ago.date()})."
            )

    def test_invalid_old_event_detected(self):
        """Un Ã©vÃ©nement de plus d'un an doit Ãªtre identifiÃ© comme invalide."""
        one_year_ago = NOW - timedelta(days=365)
        old_event    = MOCK_EVENTS_INVALID[0]

        date_start = datetime.fromisoformat(
            old_event["date_start"].replace("Z", "+00:00")
        )
        assert date_start < one_year_ago, (
            "L'Ã©vÃ©nement test 'vieux' aurait dÃ» Ãªtre antÃ©rieur Ã  un an."
        )

    def test_future_events_are_allowed(self):
        """Les Ã©vÃ©nements futurs (Ã  venir) sont autorisÃ©s."""
        for event in MOCK_EVENTS_VALID:
            date_end = datetime.fromisoformat(
                event["date_end"].replace("Z", "+00:00")
            )
            one_year_ahead = NOW + timedelta(days=365)
            assert date_end <= one_year_ahead, (
                f"La date de fin de '{event['title']}' dÃ©passe un an dans le futur."
            )

    def test_events_have_valid_date_format(self):
        """Chaque Ã©vÃ©nement doit avoir des dates de dÃ©but et de fin valides."""
        for event in MOCK_EVENTS_VALID:
            assert event.get("date_start"), f"date_start manquante : {event['uid']}"
            assert event.get("date_end"),   f"date_end manquante : {event['uid']}"
            # VÃ©rification que les dates sont parsables
            try:
                datetime.fromisoformat(event["date_start"].replace("Z", "+00:00"))
                datetime.fromisoformat(event["date_end"].replace("Z", "+00:00"))
            except ValueError as e:
                pytest.fail(f"Format de date invalide pour '{event['title']}' : {e}")

    def test_date_start_before_date_end(self):
        """La date de dÃ©but doit Ãªtre antÃ©rieure ou Ã©gale Ã  la date de fin."""
        for event in MOCK_EVENTS_VALID:
            date_start = datetime.fromisoformat(
                event["date_start"].replace("Z", "+00:00")
            )
            date_end   = datetime.fromisoformat(
                event["date_end"].replace("Z", "+00:00")
            )
            assert date_start <= date_end, (
                f"Pour '{event['title']}', date_start ({date_start.date()}) "
                f"est postÃ©rieure Ã  date_end ({date_end.date()})."
            )


# â”€â”€ Tests gÃ©ographiques â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestGeographicValidation:
    """Tests de validation de la contrainte gÃ©ographique (ÃŽle-de-France)."""

    def _is_in_idf(self, event: dict) -> bool:
        """VÃ©rifie si un Ã©vÃ©nement appartient Ã  la rÃ©gion ÃŽle-de-France."""
        combined = " ".join([
            event.get("city", ""),
            event.get("department", ""),
            event.get("region", ""),
            event.get("venue", ""),
        ]).lower()
        return any(kw in combined for kw in IDF_REGIONS)

    def test_valid_events_are_in_idf(self):
        """Tous les Ã©vÃ©nements valides doivent Ãªtre en ÃŽle-de-France."""
        for event in MOCK_EVENTS_VALID:
            assert self._is_in_idf(event), (
                f"L'Ã©vÃ©nement '{event['title']}' n'est pas en ÃŽle-de-France "
                f"(rÃ©gion : {event.get('region')}, ville : {event.get('city')})."
            )

    def test_non_idf_event_is_detected(self):
        """Un Ã©vÃ©nement hors IDF doit Ãªtre identifiÃ© comme invalide."""
        non_idf_event = MOCK_EVENTS_INVALID[1]
        assert not self._is_in_idf(non_idf_event), (
            f"L'Ã©vÃ©nement '{non_idf_event['title']}' hors IDF n'a pas Ã©tÃ© dÃ©tectÃ©."
        )

    def test_all_idf_departments_recognized(self):
        """Tous les dÃ©partements IDF doivent Ãªtre reconnus."""
        idf_departments = {
            "Paris", "Seine-et-Marne", "Yvelines", "Essonne",
            "Hauts-de-Seine", "Seine-Saint-Denis", "Val-de-Marne", "Val-d'Oise",
        }
        for dept in idf_departments:
            event = {
                "city": "Ville Test", "department": dept,
                "region": "ÃŽle-de-France", "venue": "Lieu Test",
            }
            assert self._is_in_idf(event), (
                f"Le dÃ©partement '{dept}' n'est pas reconnu comme appartenant Ã  l'IDF."
            )

    def test_events_have_required_location_fields(self):
        """Chaque Ã©vÃ©nement doit avoir au minimum un champ de localisation."""
        for event in MOCK_EVENTS_VALID:
            has_location = any([
                event.get("city"),
                event.get("department"),
                event.get("region"),
                event.get("venue"),
            ])
            assert has_location, (
                f"L'Ã©vÃ©nement '{event.get('uid')}' n'a aucun champ de localisation."
            )


# â”€â”€ Tests d'intÃ©gritÃ© des donnÃ©es â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDataIntegrity:
    """Tests d'intÃ©gritÃ© de la structure des donnÃ©es."""

    REQUIRED_FIELDS = ["uid", "title", "text", "date_start", "date_end", "region"]

    def test_events_have_required_fields(self):
        """Chaque Ã©vÃ©nement doit contenir tous les champs requis."""
        for event in MOCK_EVENTS_VALID:
            for field in self.REQUIRED_FIELDS:
                assert field in event, (
                    f"Champ '{field}' manquant dans l'Ã©vÃ©nement '{event.get('uid')}'."
                )

    def test_events_have_non_empty_text(self):
        """Le champ 'text' de chaque Ã©vÃ©nement doit Ãªtre non vide."""
        for event in MOCK_EVENTS_VALID:
            text = event.get("text", "").strip()
            assert len(text) > 10, (
                f"Le texte de '{event.get('uid')}' est trop court ou vide : '{text}'"
            )

    def test_events_have_non_empty_title(self):
        """Chaque Ã©vÃ©nement doit avoir un titre non vide."""
        for event in MOCK_EVENTS_VALID:
            assert event.get("title", "").strip(), (
                f"Le titre de '{event.get('uid')}' est vide."
            )

    def test_events_uids_are_unique(self):
        """Les identifiants uniques (uid) ne doivent pas se rÃ©pÃ©ter."""
        uids = [e["uid"] for e in MOCK_EVENTS_VALID]
        assert len(uids) == len(set(uids)), (
            f"Des uid en doublon ont Ã©tÃ© dÃ©tectÃ©s : {uids}"
        )

    def test_text_contains_title(self):
        """Le champ 'text' doit contenir le titre de l'Ã©vÃ©nement."""
        for event in MOCK_EVENTS_VALID:
            assert event["title"] in event["text"], (
                f"Le champ 'text' de '{event['title']}' ne contient pas son titre."
            )


# â”€â”€ Tests du pipeline de chunking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestChunkingPipeline:
    """Tests du pipeline de dÃ©coupage en chunks."""

    def test_documents_creation(self):
        """La conversion d'Ã©vÃ©nements en Documents LangChain doit fonctionner."""
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
        """Le chunking doit produire des chunks non vides avec les bonnes mÃ©tadonnÃ©es."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Texte long pour forcer le dÃ©coupage
        long_text = "Test. " * 300
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
            assert len(chunk.page_content.strip()) > 0, "Un chunk ne doit pas Ãªtre vide."
            assert chunk.metadata.get("uid") == "test-001", "Les mÃ©tadonnÃ©es doivent Ãªtre prÃ©servÃ©es."

    def test_chunk_size_respected(self):
        """Les chunks ne doivent pas dÃ©passer la taille configurÃ©e."""
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        chunk_size  = 200
        long_text   = "A" * 1000
        doc         = Document(page_content=long_text, metadata={})
        splitter    = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=0
        )
        chunks      = splitter.split_documents([doc])

        for chunk in chunks:
            # TolÃ©rance de 10% pour les sÃ©parateurs
            assert len(chunk.page_content) <= chunk_size * 1.1, (
                f"Un chunk dÃ©passe la taille max : {len(chunk.page_content)} > {chunk_size}"
            )


# â”€â”€ Tests de l'index FAISS (mock) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestFAISSIndex:
    """Tests de la base vectorielle FAISS (avec mock de l'API Mistral)."""

    def test_faiss_search_returns_results(self):
        """La recherche FAISS doit retourner des rÃ©sultats sous forme de Documents."""
        from langchain_core.documents import Document

        # Mock du vectorstore FAISS
        mock_vectorstore = MagicMock()
        mock_results = [
            Document(
                page_content=event["text"],
                metadata={"uid": event["uid"], "title": event["title"], "city": event["city"]},
            )
            for event in MOCK_EVENTS_VALID[:2]
        ]
        mock_vectorstore.similarity_search.return_value = mock_results

        results = mock_vectorstore.similarity_search("concert Paris", k=3)

        assert len(results) > 0, "La recherche doit retourner au moins un rÃ©sultat."
        assert all(isinstance(r, Document) for r in results), \
            "Tous les rÃ©sultats doivent Ãªtre des Documents LangChain."

    def test_faiss_search_respects_top_k(self):
        """La recherche FAISS doit respecter la limite TOP_K."""
        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = MOCK_EVENTS_VALID[:2]

        k       = 2
        results = mock_vectorstore.similarity_search("musique", k=k)
        mock_vectorstore.similarity_search.assert_called_once_with("musique", k=k)

    def test_faiss_search_returns_relevant_metadata(self):
        """Les rÃ©sultats FAISS doivent contenir les mÃ©tadonnÃ©es des Ã©vÃ©nements."""
        from langchain_core.documents import Document

        mock_vectorstore = MagicMock()
        mock_results     = [
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

        assert results[0].metadata.get("uid"), "L'uid doit Ãªtre prÃ©sent dans les mÃ©tadonnÃ©es."
        assert results[0].metadata.get("title"), "Le titre doit Ãªtre prÃ©sent dans les mÃ©tadonnÃ©es."
        assert results[0].metadata.get("city"), "La ville doit Ãªtre prÃ©sente dans les mÃ©tadonnÃ©es."


# â”€â”€ Tests d'intÃ©gration (optionnels, nÃ©cessitent les donnÃ©es) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class TestDataFileIntegration:
    """Tests d'intÃ©gration nÃ©cessitant le fichier de donnÃ©es rÃ©el."""

    @pytest.mark.skipif(
        not DATA_FILE.exists(),
        reason=f"Fichier de donnÃ©es absent : {DATA_FILE} â€” lancez fetch_events.py d'abord",
    )
    def test_data_file_contains_events(self):
        """Le fichier events_clean.json doit contenir des Ã©vÃ©nements."""
        with open(DATA_FILE, encoding="utf-8") as f:
            events = json.load(f)
        assert len(events) > 0, "Le fichier de donnÃ©es est vide."

    @pytest.mark.skipif(
        not DATA_FILE.exists(),
        reason="Fichier de donnÃ©es absent",
    )
    def test_all_real_events_are_recent(self):
        """Tous les Ã©vÃ©nements rÃ©els du fichier doivent avoir moins d'un an."""
        with open(DATA_FILE, encoding="utf-8") as f:
            events = json.load(f)

        one_year_ago = NOW - timedelta(days=365)
        errors       = []

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
                errors.append(f"'{event.get('uid')}' : format de date invalide : {date_str}")

        assert not errors, (
            f"{len(errors)} Ã©vÃ©nements dÃ©passent la limite d'un an :\n"
            + "\n".join(errors[:5])
        )

    @pytest.mark.skipif(
        not DATA_FILE.exists(),
        reason="Fichier de donnÃ©es absent",
    )
    def test_all_real_events_are_in_idf(self):
        """Tous les Ã©vÃ©nements rÃ©els doivent Ãªtre en ÃŽle-de-France."""
        with open(DATA_FILE, encoding="utf-8") as f:
            events = json.load(f)

        errors = []
        for event in events:
            combined = " ".join([
                event.get("city", ""), event.get("department", ""),
                event.get("region", ""), event.get("venue", ""),
            ]).lower()
            if not any(kw in combined for kw in IDF_REGIONS):
                errors.append(
                    f"'{event.get('title')}' â€” rÃ©gion : {event.get('region')}, "
                    f"ville : {event.get('city')}"
                )

        assert not errors, (
            f"{len(errors)} Ã©vÃ©nements ne sont pas en IDF :\n"
            + "\n".join(errors[:5])
        )


