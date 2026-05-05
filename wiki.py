import re

import httpx

ENDPOINT = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "WikipediaGolf (justinpyron@gmail.com)"}


def strip_html_tags(text: str) -> str:
    """Remove HTML tags from a string."""
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def find_articles(
    query: str,
    min_wordcount: int = 1000,
    n_results: int = 5,
) -> list[dict]:
    """Search Wikipedia for articles matching the given query."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "utf8": 1,
        "srlimit": n_results,
    }
    results = []
    try:
        response = httpx.get(ENDPOINT, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        if "query" in data and "search" in data["query"]:
            for result in data["query"]["search"]:
                if result["wordcount"] >= min_wordcount:
                    results.append(
                        {
                            "title": result["title"],
                            "page_id": str(result["pageid"]),
                            "snippet": strip_html_tags(result["snippet"]),
                        }
                    )
    except httpx.HTTPError as e:
        print(f"Error making API request: {e}")
    return results


def fetch_article(page_id: str) -> str:
    """Fetch the text of a Wikipedia article using its page ID."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "pageids": page_id,
        "explaintext": 1,  # Get plain text instead of HTML
        "exsectionformat": "plain",
    }
    try:
        response = httpx.get(ENDPOINT, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        text = data["query"]["pages"][page_id]["extract"]
        return text
    except httpx.HTTPError as e:
        print(f"Error making API request: {e}")
        return ""
