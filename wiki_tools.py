"""Tools for interacting with Wikipedia via the Wikimedia Core REST API.

Reference: https://www.mediawiki.org/wiki/API:REST_API/Reference
"""

from urllib.parse import quote, unquote

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel

BASE_URL = "https://en.wikipedia.org/w/rest.php/v1"
HEADERS = {"User-Agent": "WikipediaGolf (justinpyron@gmail.com)"}

EXCLUDE_SELECTORS = (
    "table.infobox",
    "table.sidebar",
    "table.vertical-navbox",
    "table.navbox",
    "div.navbox",
    "table.ambox",
    "table.metadata",
    "ol.references",
    "div.reflist",
    "sup.reference",
    "div.hatnote",
    "div[role='note']",
    "span.mw-editsection",
    "figure",
    "figcaption",
    ".gallerytext",
    ".thumbcaption",
    "div.thumb",
    "style",
    "script",
)

EXCLUDE_SECTIONS = (
    "See also",
    "Notes",
    "References",
    "Works cited",
    "Further reading",
    "External links",
    "Bibliography",
    "Citations",
    "Sources",
    "Footnotes",
    "Explanatory notes",
)

NON_ARTICLE_PREFIXES = (
    "File:",
    "Image:",
    "Category:",
    "Help:",
    "Wikipedia:",
    "Template:",
    "Portal:",
    "Special:",
    "Talk:",
)

STRIP_ATTRS = ("class", "id", "data-mw", "typeof", "about", "style", "rel")


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


class Article(BaseModel):
    """A fetched Wikipedia article. `key` reflects the post-redirect identity."""

    id: int
    key: str
    title: str
    content: str


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


def fetch_article(key: str) -> Article | None:
    """Fetch a Wikipedia article and return its cleaned HTML content.

    Wikilinks in `content` use href="./Article_Key" (URL-encoded), the
    same form accepted by this function's `key` parameter, so callers
    can chain extracted hrefs back into fetch_article directly.
    """
    data = _request_with_html(key)
    if data is None:
        return None
    soup = BeautifulSoup(data["html"], "lxml")
    body = soup.body or soup
    _strip_non_prose_elements(body)
    _strip_non_prose_sections(body)
    _strip_non_article_links(body)
    _simplify_attributes(body)
    return Article(
        id=data["id"],
        key=data["key"],
        title=data["title"],
        content=body.decode_contents(),
    )


def _request_with_html(key: str) -> dict | None:
    """GET /page/{key}/with_html and return the JSON payload."""
    url = f"{BASE_URL}/page/{quote(key, safe='')}/with_html"
    try:
        response = httpx.get(url, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"Error making API request: {e}")
        return None


def _strip_non_prose_elements(soup: Tag) -> None:
    """Remove non-prose containers (infoboxes, navboxes, references, etc.)."""
    for selector in EXCLUDE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()


def _strip_non_prose_sections(soup: Tag) -> None:
    """Remove <section> elements whose direct heading is in EXCLUDE_SECTIONS."""
    for section in soup.find_all("section"):
        heading = section.find(["h2", "h3", "h4", "h5", "h6"], recursive=False)
        if heading and heading.get_text(strip=True) in EXCLUDE_SECTIONS:
            section.decompose()


def _strip_non_article_links(soup: Tag) -> None:
    """Unwrap non-article and red wikilinks; normalize article hrefs."""
    for a in soup.find_all("a"):
        href = a.get("href", "")
        classes = a.get("class") or []
        is_red_link = "new" in classes
        if not href.startswith("./") or is_red_link:
            a.unwrap()
            continue
        target = unquote(href[2:])
        if target.startswith(NON_ARTICLE_PREFIXES):
            a.unwrap()
            continue
        a["href"] = href.split("#", 1)[0]


def _simplify_attributes(soup: Tag) -> None:
    """Strip noisy bookkeeping attributes from every tag."""
    for tag in soup.descendants:
        if not isinstance(tag, Tag):
            continue
        for attr in list(tag.attrs):
            if attr in STRIP_ATTRS or attr.startswith("data-"):
                del tag.attrs[attr]
