"""Tools for interacting with Wikipedia via the Wikimedia Core REST API.

Reference: https://www.mediawiki.org/wiki/API:REST_API/Reference
"""

import re
from urllib.parse import quote, unquote

import httpx
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from pydantic import BaseModel
from tabulate import tabulate

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


class ArticleLink(BaseModel):
    """A single wikilink extracted from an article."""

    key: str
    text: str
    title: str


class ArticleLinks(BaseModel):
    """All navigable wikilinks found in a Wikipedia article."""

    id: int
    key: str
    title: str
    links: list[ArticleLink]

    def to_markdown_table(self, omit: list[str] | None = None) -> str:
        """Return a markdown table of the links."""
        data = [link.model_dump(exclude=set(omit or [])) for link in self.links]
        if not data:
            return ""
        return tabulate(data, headers="keys", tablefmt="github")


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
    """Fetch a Wikipedia article and return its content as cleaned Markdown.

    Wikilinks in `content` use the form [label](./Article_Key), where the
    href is the same form accepted by this function's `key` parameter
    (after stripping the "./" prefix), so callers can chain extracted
    hrefs back into fetch_article directly.
    """
    data = _request_with_html(key)
    if data is None:
        return None
    soup = BeautifulSoup(data["html"], "lxml")
    body = soup.body or soup
    _strip_elements_by_selector(body)
    _strip_sections_by_heading(body)
    _unwrap_invalid_links(body)
    _normalize_article_hrefs(body)
    content = _to_markdown(body)
    return Article(
        id=data["id"],
        key=data["key"],
        title=data["title"],
        content=content,
    )


def fetch_article_links(key: str) -> ArticleLinks | None:
    """Fetch a Wikipedia article and extract all unique navigable wikilinks."""
    data = _request_with_html(key)
    if data is None:
        return None
    soup = BeautifulSoup(data["html"], "lxml")
    body = soup.body or soup

    _strip_elements_by_selector(body)
    _strip_sections_by_heading(body)
    _unwrap_invalid_links(body)
    _normalize_article_hrefs(body)

    links: list[ArticleLink] = []
    seen_keys: set[str] = set()

    for a in body.find_all("a", href=True):
        href = a["href"]
        article_key = unquote(href[2:])
        if article_key in seen_keys:
            continue
        seen_keys.add(article_key)

        text = a.get_text(strip=True)
        title = a.get("title", "")
        if not text:
            continue

        links.append(ArticleLink(key=article_key, text=text, title=title))

    return ArticleLinks(
        id=data["id"],
        key=data["key"],
        title=data["title"],
        links=links,
    )


def _request_with_html(key: str) -> dict | None:
    """GET /page/{key}/with_html and return the JSON payload."""
    # safe="" ensures characters like "/" and "?" in titles (e.g. "AC/DC")
    # are percent-encoded so they aren't parsed as URL structure.
    url = f"{BASE_URL}/page/{quote(key, safe='')}/with_html"
    try:
        response = httpx.get(url, headers=HEADERS, follow_redirects=True)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        print(f"Error making API request: {e}")
        return None


def _strip_elements_by_selector(soup: Tag) -> None:
    """Remove elements matching CSS selectors in EXCLUDE_SELECTORS."""
    for selector in EXCLUDE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()


def _strip_sections_by_heading(soup: Tag) -> None:
    """Remove entire <section> blocks whose heading text is in EXCLUDE_SECTIONS."""
    for section in soup.find_all("section"):
        heading = section.find(["h2", "h3", "h4", "h5", "h6"], recursive=False)
        if heading and heading.get_text(strip=True) in EXCLUDE_SECTIONS:
            section.decompose()


def _unwrap_invalid_links(soup: Tag) -> None:
    """Unwrap non-article and red wikilinks."""
    for a in soup.find_all("a"):
        href = a.get("href", "")

        # Red links (class="new") point to articles that don't exist yet —
        # following them would 404. External links and in-page anchors don't
        # start with "./" and aren't navigable Wikipedia articles.
        is_external_link = not href.startswith("./")
        is_red_link = "new" in (a.get("class") or [])
        if is_external_link or is_red_link:
            a.unwrap()
            continue

        # Hrefs like "./File:Example.jpg" or "./Category:Physics" point to
        # non-article namespaces. unquote is needed because Parsoid
        # percent-encodes characters like ":" (e.g. "Help%3AContents").
        target = unquote(href[2:])  # strip the "./" prefix + unquote
        if target.startswith(NON_ARTICLE_PREFIXES):
            a.unwrap()
            continue


def _normalize_article_hrefs(soup: Tag) -> None:
    """Strip section fragments from valid article links."""
    for a in soup.find_all("a"):
        href = a.get("href", "")
        # Strip section fragments (e.g. "./Einstein#Legacy" -> "./Einstein")
        # so the href is a clean article key that can be fed back into
        # fetch_article.
        a["href"] = href.split("#", 1)[0]


def _to_markdown(soup: Tag) -> str:
    """Convert cleaned HTML to Markdown and normalize whitespace."""
    md = markdownify(str(soup), heading_style="ATX", strip=["section"])
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
