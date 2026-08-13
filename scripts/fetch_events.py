"""
fetch_events.py
---------------
Script de récupération et de nettoyage des données d'événements culturels
depuis l'API Open Agenda.

P�rimètre géographique : Île-de-France (8 départements)
P�rimètre temporel     : événements de moins d'un an (passés ou à venir sur 6 mois)

Pipeline de traitement :
    1. Appel paginé à l'API Open Agenda v2
    2. Filtrage géographique (mots-clés IDF sur ville, département, région)
    3. Nettoyage et normalisation des champs
    4. Construction du champ `text` agrégé pour la vectorisation
    5. Sauvegarde dans data/events_clean.json

Usage:
    python scripts/fetch_events.py

Sortie:
    data/events_clean.json — liste d'événements nettoyés et structurés
"""

import os
import json
import logging
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# ── Configuration ──────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GEO_REGION  = os.getenv("GEO_REGION", "Île-de-France")
MAX_EVENTS  = int(os.getenv("MAX_EVENTS", 500))
DATA_DIR    = Path(__file__).parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "events_clean.json"

# API Open Agenda v2
BASE_URL = "https://api.openagenda.com/v2/events"
API_KEY  = "6a4049ca7e5047afb348db0bb3a5e58c"  # clé publique Open Agenda

# Mots-clés de détection IDF (ville, département, région)
IDF_KEYWORDS = [
    "île-de-france", "ile-de-france", "paris", "seine-et-marne",
    "yvelines", "essonne", "hauts-de-seine", "seine-saint-Denis",
    "val-de-marne", "val-d'oise",
]


# ── Fonctions ──────────────────────────────────────────────────────────────────

def get_date_boundaries() -> tuple:
    """
    Calcule les bornes temporelles du filtre de récupération.

    La fenêtre couvre :
    - Les événements passés : jusqu'à 365 jours en arrière
    - Les événements à venir : jusqu'à 180 jours en avant

    Returns:
        tuple[datetime, datetime]: (date_début, date_fin) avec timezone UTC.
    """
    now              = datetime.now(tz=timezone.utc)
    one_year_ago     = now - timedelta(days=365)
    six_months_ahead = now + timedelta(days=180)
    return one_year_ago, six_months_ahead


def fetch_page(after_uid: str | None, date_min: datetime, date_max: datetime) -> dict:
    """
    Récupère une page de résultats depuis l'API Open Agenda v2.

    Utilise la pagination par curseur (paramètre `after`) pour itérer
    sur l'ensemble des résultats sans limite de taille.

    Args:
        after_uid  (str | None): UID du dernier événement récupéré (curseur de pagination).
                                 None pour la première page.
        date_min   (datetime):   Borne temporelle inférieure (UTC).
        date_max   (datetime):   Borne temporelle supérieure (UTC).

    Returns:
        dict: Réponse JSON brute de l'API (champ "events" + métadonnées).

    Raises:
        requests.HTTPError: En cas d'erreur HTTP (4xx, 5xx).
    """
    params = {
        "key":                   API_KEY,
        "size":                  100,
        "sort":                  "updatedAt.desc",
        "locationUid":           10,  # Identifiant France dans Open Agenda
        "timings[gte]":          date_min.strftime("%Y-%m-%dT%H:%M:%S"),
        "timings[lte]":          date_max.strftime("%Y-%m-%dT%H:%M:%S"),
        "longDescriptionFormat": "markdown",
    }
    if after_uid:
        params["after"] = after_uid

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def is_idf(event: dict) -> bool:
    """
    Détermine si un événement se déroule en Île-de-France.

    Effectue une recherche par mots-clés sur les champs de localisation
    (région, département, ville) de l'événement brut.

    Args:
        event (dict): Événement brut issu de l'API Open Agenda.

    Returns:
        bool: True si l'événement est localisé en Île-de-France.
    """
    location = event.get("location", {}) or {}
    region   = (location.get("region", "")     or "").lower()
    dept     = (location.get("department", "") or "").lower()
    city     = (location.get("city", "")       or "").lower()
    combined = f"{region} {dept} {city}"
    return any(kw in combined for kw in IDF_KEYWORDS)


def clean_event(raw: dict) -> dict | None:
    """
    Nettoie et structure un événement brut issu de l'API Open Agenda.

    Règles de nettoyage appliquées :
    - Extrait les champs titre et description en français (fallback anglais)
    - Exclut les événements sans titre ET sans description
    - Normalise les champs de localisation (ville, département, région, lieu)
    - Extrait les dates de début et de fin depuis le tableau `timings`
    - Construit le champ `text` agrégé utilisé pour la vectorisation FAISS

    Args:
        raw (dict): Dictionnaire brut d'un événement Open Agenda.

    Returns:
        dict | None: Événement nettoyé et structuré, ou None si invalide.
    """
    # Extraction titre et description (préférence FR)
    title_obj   = raw.get("title", {}) or {}
    title       = (title_obj.get("fr") or title_obj.get("en") or "").strip()

    desc_obj    = raw.get("description", {}) or {}
    description = (desc_obj.get("fr") or desc_obj.get("en") or "").strip()

    # Exclusion si aucun contenu textuel
    if not title and not description:
        return None

    # Champs de localisation
    location = raw.get("location", {}) or {}
    city     = location.get("city", "")       or ""
    dept     = location.get("department", "") or ""
    region   = location.get("region", "")     or ""
    venue    = location.get("name", "")       or ""

    # Dates depuis le tableau timings
    timings    = raw.get("timings", []) or []
    date_start = timings[0].get("begin", "")  if timings else ""
    date_end   = timings[-1].get("end", "")   if timings else ""

    # Catégorie
    cat_obj  = raw.get("category", {}) or {}
    category = (cat_obj.get("fr") or "").strip()

    uid = str(raw.get("uid", ""))

    # Construction du texte agrégé pour vectorisation
    text = "\n".join(filter(None, [
        f"Événement : {title}",
        f"Description : {description}"  if description else "",
        f"Lieu : {venue}, {city}, {dept}" if (venue or city) else "",
        f"Région : {region}"            if region     else "",
        f"Catégorie : {category}"       if category   else "",
        f"Date de début : {date_start}" if date_start else "",
        f"Date de fin : {date_end}"     if date_end   else "",
    ]))

    return {
        "uid":         uid,
        "title":       title,
        "description": description,
        "date_start":  date_start,
        "date_end":    date_end,
        "city":        city,
        "department":  dept,
        "region":      region,
        "venue":       venue,
        "categories":  category,
        "keywords":    "",
        "text":        text,
    }


def fetch_all_events() -> list[dict]:
    """
    Récupère et nettoie tous les événements IDF dans la fenêtre temporelle définie.

    Effectue une pagination complète de l'API Open Agenda jusqu'à atteindre
    MAX_EVENTS événements valides ou l'épuisement des résultats.

    En cas d'erreur API (réseau, quota), le dernier état est retourné et un
    avertissement est loggé.

    Returns:
        list[dict]: Liste des événements IDF nettoyés et structurés.
    """
    date_min, date_max = get_date_boundaries()
    logger.info(f"Période : {date_min.date()} → {date_max.date()}")
    logger.info(f"Région cible : {GEO_REGION}")

    events, after_uid, total_raw = [], None, 0

    while len(events) < MAX_EVENTS:
        logger.info(f"Récupération… ({len(events)} événements IDF jusqu'ici)")
        try:
            data = fetch_page(after_uid, date_min, date_max)
        except requests.HTTPError as e:
            logger.error(f"Erreur API : {e}")
            break

        records = data.get("events", [])
        if not records:
            logger.info("Fin de pagination.")
            break

        total_raw += len(records)
        for raw in records:
            if is_idf(raw):
                cleaned = clean_event(raw)
                if cleaned:
                    events.append(cleaned)
            if len(events) >= MAX_EVENTS:
                break

        # Curseur de pagination via le dernier uid
        after_uid = records[-1].get("uid") if records else None
        if not after_uid:
            break

    logger.info(f"Événements bruts récupérés : {total_raw} | IDF valides : {len(events)}")
    return events


def generate_demo_events() -> list[dict]:
    """
    Génère un jeu de données de démonstration pour le POC.

    Utilisé comme fallback lorsque l'API Open Agenda est indisponible ou
    ne renvoie pas de résultats. Contient 15 événements culturels fictifs
    mais réalistes, tous localisés en Île-de-France et datés dans le futur
    immédiat.

    Returns:
        list[dict]: Liste de 15 événements de démonstration structurés.
    """
    now = datetime.now(tz=timezone.utc)
    return [
        {"uid": "d001", "title": "Concert Jazz au Sunset",
         "description": "Une soirée jazz intime avec le quartet de Baptiste Trotignon au Sunset Sunside.",
         "date_start": (now + timedelta(days=3)).isoformat(), "date_end": (now + timedelta(days=3)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Le Sunset Sunside",
         "categories": "Musique", "keywords": "jazz, concert, musique live",
         "text": "Événement : Concert Jazz au Sunset\nDescription : Soirée jazz intime avec Baptiste Trotignon.\nLieu : Le Sunset Sunside, Paris\nRégion : Île-de-France\nCatégorie : Musique"},
        {"uid": "d002", "title": "Exposition Klimt — L'Or et la Couleur",
         "description": "Rétrospective immersive de Klimt au Grand Palais Immersif. Projections géantes.",
         "date_start": (now - timedelta(days=10)).isoformat(), "date_end": (now + timedelta(days=60)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Grand Palais Immersif",
         "categories": "Exposition", "keywords": "art, Klimt, exposition, immersif",
         "text": "Événement : Exposition Klimt\nDescription : Rétrospective immersive au Grand Palais Immersif.\nLieu : Grand Palais Immersif, Paris\nRégion : Île-de-France\nCatégorie : Exposition"},
        {"uid": "d003", "title": "Festival Solidays",
         "description": "Festival de musique et de solidarité sur l'hippodrome de Longchamp. 3 jours, 70 artistes.",
         "date_start": (now + timedelta(days=15)).isoformat(), "date_end": (now + timedelta(days=17)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Hippodrome de Longchamp",
         "categories": "Festival, Musique", "keywords": "festival, musique, engagement",
         "text": "Événement : Festival Solidays\nDescription : 70 artistes à Longchamp.\nLieu : Hippodrome de Longchamp, Paris\nRégion : Île-de-France\nCatégorie : Festival, Musique"},
        {"uid": "d004", "title": "Corps Célestes — Danse contemporaine",
         "description": "La compagnie Käfig présente Corps Célestes, hip-hop et danse contemporaine.",
         "date_start": (now + timedelta(days=5)).isoformat(), "date_end": (now + timedelta(days=6)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Théâtre National de Chaillot",
         "categories": "Danse, Spectacle vivant", "keywords": "danse, contemporain, hip-hop",
         "text": "Événement : Corps Célestes\nDescription : Hip-hop et danse contemporaine à Chaillot.\nLieu : Théâtre National de Chaillot, Paris\nRégion : Île-de-France\nCatégorie : Danse"},
        {"uid": "d005", "title": "Nuit des Musées — Louvre gratuit",
         "description": "Ouverture nocturne gratuite du Louvre avec visites guidées thématiques et ateliers.",
         "date_start": (now + timedelta(days=8)).isoformat(), "date_end": (now + timedelta(days=8)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Musée du Louvre",
         "categories": "Musée, Gratuit", "keywords": "Louvre, gratuit, nuit musées",
         "text": "Événement : Nuit des Musées Louvre\nDescription : Ouverture nocturne gratuite avec visites guidées.\nLieu : Musée du Louvre, Paris\nRégion : Île-de-France\nCatégorie : Musée, Gratuit"},
        {"uid": "d006", "title": "Atelier poterie enfants — Cité des Sciences",
         "description": "Atelier créatif de poterie pour les enfants de 6 à 12 ans. Inscription obligatoire.",
         "date_start": (now + timedelta(days=2)).isoformat(), "date_end": (now + timedelta(days=2)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Cité des Sciences",
         "categories": "Atelier, Jeunesse", "keywords": "enfants, poterie, famille",
         "text": "Événement : Atelier poterie enfants\nDescription : Atelier créatif 6-12 ans à la Cité des Sciences.\nLieu : Cité des Sciences, Paris\nRégion : Île-de-France\nCatégorie : Atelier, Jeunesse"},
        {"uid": "d007", "title": "Concert Philharmonie — Orchestre de Paris",
         "description": "L'Orchestre de Paris interprète la Symphonie n°9 de Beethoven sous Klaus Mäkelä.",
         "date_start": (now + timedelta(days=12)).isoformat(), "date_end": (now + timedelta(days=12)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Philharmonie de Paris",
         "categories": "Musique classique", "keywords": "classique, Beethoven, Philharmonie",
         "text": "Événement : Concert Philharmonie\nDescription : Symphonie n°9 Beethoven, Klaus Mäkelä.\nLieu : Philharmonie de Paris, Paris\nRégion : Île-de-France\nCatégorie : Musique classique"},
        {"uid": "d008", "title": "Marché artisanal de Versailles",
         "description": "Grand marché artisanal dans les jardins du Château de Versailles.",
         "date_start": (now + timedelta(days=4)).isoformat(), "date_end": (now + timedelta(days=5)).isoformat(),
         "city": "Versailles", "department": "Yvelines", "region": "Île-de-France", "venue": "Château de Versailles",
         "categories": "Artisanat, Marché", "keywords": "artisanat, Versailles, famille",
         "text": "Événement : Marché artisanal Versailles\nDescription : Marché dans les jardins du Château de Versailles.\nLieu : Château de Versailles, Versailles, Yvelines\nRégion : Île-de-France\nCatégorie : Artisanat, Marché"},
        {"uid": "d009", "title": "Cinéma en plein air — Amélie Poulain",
         "description": "Projection gratuite en plein air au Parc de la Villette, Amélie Poulain version 4K restaurée.",
         "date_start": (now + timedelta(days=7)).isoformat(), "date_end": (now + timedelta(days=7)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Parc de la Villette",
         "categories": "Cinéma, Gratuit", "keywords": "cinéma, gratuit, plein air",
         "text": "Événement : Cinéma en plein air Amélie Poulain\nDescription : Projection gratuite au Parc de la Villette.\nLieu : Parc de la Villette, Paris\nRégion : Île-de-France\nCatégorie : Cinéma, Gratuit"},
        {"uid": "d010", "title": "Conférence IA & Art — Centre Pompidou",
         "description": "Table ronde sur l'intelligence artificielle et la création artistique. Entrée libre sur inscription.",
         "date_start": (now + timedelta(days=20)).isoformat(), "date_end": (now + timedelta(days=20)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Centre Pompidou",
         "categories": "Conférence", "keywords": "IA, art, conférence",
         "text": "Événement : Conférence IA et Art\nDescription : Table ronde IA et création artistique au Centre Pompidou.\nLieu : Centre Pompidou, Paris\nRégion : Île-de-France\nCatégorie : Conférence"},
        {"uid": "d011", "title": "Escape Game — Musée d'Orsay",
         "description": "Expérience immersive : résolvez les énigmes cachées dans les tableaux impressionnistes. Groupes 2-6 personnes.",
         "date_start": (now + timedelta(days=1)).isoformat(), "date_end": (now + timedelta(days=90)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Musée d'Orsay",
         "categories": "Jeu, Musée", "keywords": "escape game, Orsay, insolite",
         "text": "Événement : Escape Game Musée d'Orsay\nDescription : Énigmes dans les tableaux impressionnistes.\nLieu : Musée d'Orsay, Paris\nRégion : Île-de-France\nCatégorie : Jeu, Musée"},
        {"uid": "d012", "title": "Le Petit Prince — Théâtre du Châtelet",
         "description": "Adaptation théâtrale du Petit Prince avec marionnettes et musique live pour toute la famille.",
         "date_start": (now + timedelta(days=6)).isoformat(), "date_end": (now + timedelta(days=7)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Théâtre du Châtelet",
         "categories": "Théâtre, Jeunesse", "keywords": "théâtre, enfants, famille",
         "text": "Événement : Le Petit Prince Théâtre du Châtelet\nDescription : Marionnettes et musique live pour toute la famille.\nLieu : Théâtre du Châtelet, Paris\nRégion : Île-de-France\nCatégorie : Théâtre, Jeunesse"},
        {"uid": "d013", "title": "Visite nocturne — Catacombes de Paris",
         "description": "Visite guidée nocturne des Catacombes, histoire secrète des galeries souterraines.",
         "date_start": (now + timedelta(days=9)).isoformat(), "date_end": (now + timedelta(days=9)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "Catacombes de Paris",
         "categories": "Visite, Histoire", "keywords": "catacombes, nocturne, insolite",
         "text": "Événement : Visite nocturne Catacombes\nDescription : Visite guidée nocturne des Catacombes.\nLieu : Catacombes de Paris, Paris\nRégion : Île-de-France\nCatégorie : Visite, Histoire"},
        {"uid": "d014", "title": "Open mic poésie — La Maroquinerie",
         "description": "Soirée open mic poésie et slam à La Maroquinerie. Ouvert à tous, entrée libre.",
         "date_start": (now + timedelta(days=11)).isoformat(), "date_end": (now + timedelta(days=11)).isoformat(),
         "city": "Paris", "department": "Paris", "region": "Île-de-France", "venue": "La Maroquinerie",
         "categories": "Poésie, Slam", "keywords": "slam, poésie, gratuit",
         "text": "Événement : Open mic poésie La Maroquinerie\nDescription : Soirée slam et poésie ouverte à tous.\nLieu : La Maroquinerie, Paris\nRégion : Île-de-France\nCatégorie : Poésie, Slam"},
        {"uid": "d015", "title": "Marché de Noël de Saint-Germain-en-Laye",
         "description": "Marché de Noël traditionnel avec chalets artisanaux et animations enfants.",
         "date_start": (now + timedelta(days=30)).isoformat(), "date_end": (now + timedelta(days=45)).isoformat(),
         "city": "Saint-Germain-en-Laye", "department": "Yvelines", "region": "Île-de-France", "venue": "Centre-ville",
         "categories": "Marché, Fête", "keywords": "Noël, marché, famille",
         "text": "Événement : Marché de Noël Saint-Germain-en-Laye\nDescription : Chalets artisanaux et animations enfants.\nLieu : Centre-ville, Saint-Germain-en-Laye, Yvelines\nRégion : Île-de-France\nCatégorie : Marché, Fête"},
    ]


# ── Point d'entrée ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Point d'entrée principal du script de récupération des événements.

    Tente d'abord de récupérer les données depuis l'API Open Agenda.
    En cas d'échec ou d'absence de résultats, bascule automatiquement
    sur les données de démonstration (generate_demo_events).
    """
    logger.info("=== Démarrage de la récupération des événements ===")
    events = fetch_all_events()

    if not events:
        logger.warning("API sans résultat — utilisation des données de démonstration.")
        events = generate_demo_events()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    logger.info(f"Sauvegardé : {OUTPUT_FILE} ({len(events)} événements)")
    logger.info("=== Récupération terminée ===")


if __name__ == "__main__":
    main()
