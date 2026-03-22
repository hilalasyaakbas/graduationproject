import json
import re
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}


def fetch_imdb_movie_data(title: str, year: int | None = None, imdb_url: str | None = None):
    try:
        target_url = imdb_url or _find_imdb_url(title, year)
        if not target_url:
            return None

        response = requests.get(target_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        script = soup.find("script", type="application/ld+json")
        if not script or not script.string:
            return {"imdb_url": target_url}

        payload = json.loads(script.string.strip())
        directors = [item["name"] for item in _normalize_list(payload.get("director")) if item.get("name")]
        actors = [item["name"] for item in _normalize_list(payload.get("actor")) if item.get("name")]
        genres = payload.get("genre") if isinstance(payload.get("genre"), list) else [payload.get("genre")] if payload.get("genre") else []

        duration = None
        duration_raw = payload.get("duration")
        if isinstance(duration_raw, str):
            match = re.search(r"PT(?:(\d+)H)?(?:(\d+)M)?", duration_raw)
            if match:
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                duration = hours * 60 + minutes

        aggregate = payload.get("aggregateRating") or {}
        rating_value = aggregate.get("ratingValue")
        rating_count = aggregate.get("ratingCount")

        return {
            "title": payload.get("name") or title,
            "description": payload.get("description"),
            "image": payload.get("image"),
            "genres": [genre for genre in genres if genre],
            "directors": directors,
            "actors": actors,
            "duration": duration,
            "imdb_rating": float(rating_value) if rating_value else None,
            "imdb_votes": rating_count,
            "imdb_url": target_url,
        }
    except Exception:
        return None


def _find_imdb_url(title: str, year: int | None = None):
    query = quote_plus(f"{title} {year or ''}".strip())
    search_url = f"https://www.imdb.com/find/?q={query}&s=tt&ttype=ft"
    response = requests.get(search_url, headers=HEADERS, timeout=10)
    response.raise_for_status()

    match = re.search(r'href="(/title/tt\d+/)', response.text)
    if not match:
        return None
    return "https://www.imdb.com" + match.group(1)


def _normalize_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []
