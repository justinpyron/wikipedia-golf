"""Tools for interacting with Wikipedia via the Wikimedia Core REST API.

Reference: https://www.mediawiki.org/wiki/API:REST_API/Reference
"""

import httpx
from pydantic import BaseModel

BASE_URL = "https://en.wikipedia.org/w/rest.php/v1"
HEADERS = {"User-Agent": "WikipediaGolf (justinpyron@gmail.com)"}


class ArticleSearchResult(BaseModel):
    """A single hit from a Wikipedia search.

    The `key` field is the identifier used to fetch the article's content
    via the Core REST API's `/page/{key}/...` endpoints.
    """

    id: int
    key: str
    title: str
    excerpt: str
    description: str | None = None
    thumbnail: dict | None = None


def find_articles(query: str, limit: int = 5) -> list[ArticleSearchResult]:
    """Search Wikipedia for articles matching the query.

    Use the `key` field on each result as the identifier when fetching
    article content in follow-up requests.
    """
    params = {"q": query, "limit": limit}
    try:
        response = httpx.get(
            f"{BASE_URL}/search/page",
            params=params,
            headers=HEADERS,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPError as e:
        print(f"Error making API request: {e}")
        return []

    return [
        ArticleSearchResult(
            id=page["id"],
            key=page["key"],
            title=page["title"],
            excerpt=page["excerpt"],
            description=page.get("description"),
            thumbnail=page.get("thumbnail"),
        )
        for page in data.get("pages", [])
    ]
