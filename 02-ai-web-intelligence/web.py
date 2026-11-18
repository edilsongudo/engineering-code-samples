"""
AI Web Intelligence & External Data Extraction Tools.

Implements function-calling tools for AI agents to:
  1. `read_url` — Safely fetch and extract readable Markdown text from webpages in paginated 8,000-character chunks.
  2. `google_custom_search` — Query the Google Custom Search JSON API and optionally pre-extract page-1 content.
  3. `fetch_reddit_thread` — Retrieve public Reddit discussion threads and extract top-level user comments for
     audience research, pain-point discovery, and sentiment analysis.

Includes network security guards:
  - Strict scheme validation (`http` / `https` only).
  - Server-Side Request Forgery (SSRF) protection blocking loopback, private, link-local,
    multicast, and cloud-metadata IP ranges before outbound connections.
  - Bounded response size limit (5 MB max download) and explicit request timeouts.
"""

import ipaddress
import json
import logging
import os
import socket
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import requests
import trafilatura

try:
    from django.conf import settings as django_settings
except ImportError:
    django_settings = None  # type: ignore[assignment]

from registry import log_tool_call, register_tool

logger = logging.getLogger(__name__)

PAGE_SIZE_CHARS = 8000
MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MB response body limit
DEFAULT_TIMEOUT_SEC = 10
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)


class UnsafeUrlError(ValueError):
    """Raised when a URL uses a forbidden scheme or resolves to a private/internal network address."""


def _get_config(name: str, default: str = "") -> str:
    """Reads a setting from Django settings (when configured) or environment variables."""
    if django_settings is not None:
        try:
            if getattr(django_settings, "configured", False):
                val = getattr(django_settings, name, None)
                if val:
                    return str(val)
        except Exception:
            pass
    return os.getenv(name, default)


def _is_disallowed_ip(ip_str: str) -> bool:
    """Returns True if the IP address belongs to a private, loopback, link-local, or reserved range."""
    addr = ipaddress.ip_address(ip_str)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def validate_public_http_url(
    url: str,
    allowed_hosts: Optional[List[str]] = None,
    dns_resolver: Callable[..., Any] = socket.getaddrinfo,
) -> str:
    """
    Validates that `url` is a well-formed HTTP/HTTPS URL pointing to a public internet host.
    Resolves DNS records and blocks private, loopback, link-local (e.g. 169.254.169.254),
    and internal network addresses to prevent Server-Side Request Forgery (SSRF).
    """
    if not url or not isinstance(url, str) or not url.strip():
        raise UnsafeUrlError("URL is required.")

    cleaned_url = url.strip()
    parsed = urlparse(cleaned_url)

    if parsed.scheme not in ("http", "https"):
        raise UnsafeUrlError(f"Unsupported URL scheme '{parsed.scheme}'. Only HTTP and HTTPS are allowed.")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise UnsafeUrlError("URL is missing a valid hostname.")

    if hostname in ("localhost", "localhost.localdomain") or hostname.endswith((".local", ".internal")):
        raise UnsafeUrlError(f"Access to internal host '{hostname}' is not permitted.")

    if allowed_hosts is not None:
        if not any(hostname == h or hostname.endswith(f".{h}") for h in allowed_hosts):
            raise UnsafeUrlError(f"Host '{hostname}' is not in the allowed domain list.")

    # If hostname is a literal IP address, validate directly
    try:
        if _is_disallowed_ip(hostname):
            raise UnsafeUrlError(f"Access to non-public IP address '{hostname}' is forbidden.")
        return cleaned_url
    except ValueError as exc:
        if isinstance(exc, UnsafeUrlError):
            raise

    # Resolve hostname via DNS and inspect all returned IP addresses
    try:
        addr_info = dns_resolver(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve hostname '{hostname}': {exc}") from exc

    if not addr_info:
        raise UnsafeUrlError(f"No DNS records found for hostname '{hostname}'.")

    for entry in addr_info:
        sockaddr = entry[4]
        ip_str = sockaddr[0]
        if _is_disallowed_ip(ip_str):
            raise UnsafeUrlError(
                f"Host '{hostname}' resolved to restricted internal/private IP address '{ip_str}'."
            )

    return cleaned_url


def fetch_url_safely(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT_SEC,
    max_bytes: int = MAX_DOWNLOAD_BYTES,
    allowed_hosts: Optional[List[str]] = None,
    http_get: Callable[..., requests.Response] = requests.get,
    dns_resolver: Callable[..., Any] = socket.getaddrinfo,
) -> bytes:
    """
    Fetches a URL with SSRF validation, redirect inspection, explicit timeout,
    and a strict `max_bytes` response size limit.
    """
    validated_url = validate_public_http_url(url, allowed_hosts=allowed_hosts, dns_resolver=dns_resolver)
    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        req_headers.update(headers)

    with http_get(
        validated_url,
        headers=req_headers,
        timeout=timeout,
        stream=True,
        allow_redirects=True,
    ) as resp:
        # If the request was redirected, validate the final destination URL against SSRF rules
        if resp.url and resp.url != validated_url:
            validate_public_http_url(resp.url, allowed_hosts=allowed_hosts, dns_resolver=dns_resolver)

        resp.raise_for_status()

        content_length = resp.headers.get("Content-Length")
        if content_length and content_length.isdigit() and int(content_length) > max_bytes:
            raise ValueError(f"Response size ({content_length} bytes) exceeds maximum limit ({max_bytes} bytes).")

        chunks: List[bytes] = []
        downloaded_bytes = 0
        for chunk in resp.iter_content(chunk_size=32768):
            if not chunk:
                continue
            downloaded_bytes += len(chunk)
            if downloaded_bytes > max_bytes:
                raise ValueError(f"Response body exceeded maximum allowed size of {max_bytes} bytes.")
            chunks.append(chunk)

        return b"".join(chunks)


@register_tool(
    name="read_url",
    display_name="Extracting webpage content...",
    description=(
        "Fetches a URL and extracts clean readable plain text from the page in paginated chunks. "
        "Removes scripts, navigation, headers, and footers."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to fetch and extract text from."},
            "page": {
                "type": "integer",
                "description": (
                    "The specific page chunk to retrieve (each page contains 8000 characters). "
                    "Increment this to read further down the website. Defaults to 1."
                ),
            },
        },
        "required": ["url"],
    },
)
def read_url(
    url: str,
    page: int = 1,
    *,
    http_get: Callable[..., requests.Response] = requests.get,
    dns_resolver: Callable[..., Any] = socket.getaddrinfo,
) -> str:
    log_tool_call("read_url", url=url, page=page)

    if not url or not str(url).strip():
        return "Error extracting content: URL is required."

    try:
        page_num = max(1, int(page or 1))
    except (TypeError, ValueError):
        page_num = 1

    try:
        raw_bytes = fetch_url_safely(url, http_get=http_get, dns_resolver=dns_resolver)
        html_text = raw_bytes.decode("utf-8", errors="replace")

        full_text = trafilatura.extract(
            html_text,
            output_format="markdown",
            include_links=True,
            include_images=False,
            include_tables=True,
        )

        if not full_text:
            return "Error extracting content: No main text content could be extracted from this page."

        total_chars = len(full_text)
        start_idx = (page_num - 1) * PAGE_SIZE_CHARS
        end_idx = start_idx + PAGE_SIZE_CHARS
        total_pages = max(1, (total_chars + PAGE_SIZE_CHARS - 1) // PAGE_SIZE_CHARS)

        if start_idx >= total_chars:
            return f"[System: You requested page {page_num}, but this content only has {total_pages} pages available.]"

        sliced_text = full_text[start_idx:end_idx]
        pagination_header = (
            f"[Page {page_num} of {total_pages} | Characters {start_idx} to {min(end_idx, total_chars)} out of {total_chars}]\n"
            f"[Instruction: If you need to read the next part of this page, call 'read_url' again with 'page': {page_num + 1}]\n\n"
        )
        return pagination_header + sliced_text

    except UnsafeUrlError as exc:
        return f"Error extracting content: {exc}"
    except requests.RequestException as exc:
        return f"Error extracting content: Failed to download the URL {url} ({exc})"
    except Exception as exc:
        return f"Error extracting content: {exc}"


@register_tool(
    name="google_custom_search",
    display_name="Searching Google...",
    description=(
        "Uses the Google Custom Search API to find the top N results for a query and automatically "
        "fetches the first page of text content from each result link. "
        "Returns title, snippet, URL, and page 1 content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query string."},
            "num": {
                "type": "integer",
                "description": "Number of results to return. Defaults to 3 and is capped at 10.",
            },
            "content": {
                "type": "boolean",
                "description": "If true, fetches and includes page 1 content for each result. Defaults to true.",
            },
        },
        "required": ["query"],
    },
)
def google_custom_search(
    query: str,
    num: int = 3,
    content: bool = True,
    *,
    http_get: Callable[..., requests.Response] = requests.get,
    dns_resolver: Callable[..., Any] = socket.getaddrinfo,
) -> str:
    log_tool_call("google_custom_search", query=query, num=num, content=content)

    if not query or not str(query).strip():
        return "Error in google search: Query is required."

    api_key = _get_config("GOOGLE_API_KEY")
    search_engine_id = _get_config("GOOGLE_SEARCH_ENGINE_ID")
    if not api_key or not search_engine_id:
        return "Error in google search: GOOGLE_API_KEY and GOOGLE_SEARCH_ENGINE_ID must be configured."

    try:
        result_count = max(1, min(int(num or 3), 10))
    except (TypeError, ValueError):
        result_count = 3

    try:
        response = http_get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": search_engine_id,
                "q": query,
                "num": result_count,
            },
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("items", []):
            link = item.get("link")
            page_content = ""
            if link and content:
                page_content = read_url(link, page=1, http_get=http_get, dns_resolver=dns_resolver)
            results.append(
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "link": link,
                    "content": page_content,
                }
            )

        if not results:
            return "No Google results found for the query."

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as exc:
        return f"Error in google search: {exc}"


@register_tool(
    name="fetch_reddit_thread",
    display_name="Fetching Reddit thread...",
    description=(
        "Fetches a public Reddit thread using its URL and extracts a filtered, "
        "context-efficient JSON containing the original post text and the top N primary comments. "
        "Ideal for lead generation, listening to user pain points, or analyzing specific threads."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full Reddit thread URL (e.g., from Google search results) to analyze.",
            },
            "max_comments": {
                "type": "integer",
                "description": "Number of top-level comments to parse and return. Defaults to 15, capped at 50.",
            },
        },
        "required": ["url"],
    },
)
def fetch_reddit_thread(
    url: str,
    max_comments: int = 15,
    *,
    http_get: Callable[..., requests.Response] = requests.get,
    dns_resolver: Callable[..., Any] = socket.getaddrinfo,
) -> str:
    log_tool_call("fetch_reddit_thread", url=url, max_comments=max_comments)

    if not url or not str(url).strip():
        return "Error in reddit fetch: URL is required."

    base_url = url.strip().split("?")[0].rstrip("/")
    json_url = base_url if base_url.endswith(".json") else f"{base_url}.json"

    headers: Dict[str, str] = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    raw_cookie_string = os.getenv("REDDIT_COOKIE", "").strip()
    if raw_cookie_string:
        headers["Cookie"] = raw_cookie_string

    try:
        raw_bytes = fetch_url_safely(
            json_url,
            headers=headers,
            allowed_hosts=["reddit.com", "redd.it"],
            http_get=http_get,
            dns_resolver=dns_resolver,
        )
        data = json.loads(raw_bytes.decode("utf-8", errors="replace"))

        if not isinstance(data, list) or len(data) < 2:
            return "Error in reddit fetch: Invalid Reddit JSON structure."

        post_children = data[0].get("data", {}).get("children", [])
        post_data = post_children[0].get("data", {}) if post_children else {}
        permalink = post_data.get("permalink") or ""
        cleaned_post = {
            "title": post_data.get("title"),
            "subreddit": post_data.get("subreddit"),
            "author": post_data.get("author"),
            "content": post_data.get("selftext"),
            "ups": post_data.get("ups"),
            "permalink": f"https://reddit.com{permalink}" if permalink else base_url,
            "created_utc": post_data.get("created_utc"),
        }

        raw_comments = data[1].get("data", {}).get("children", [])
        cleaned_comments = []

        try:
            comment_limit = max(1, min(int(max_comments or 15), 50))
        except (TypeError, ValueError):
            comment_limit = 15

        for comment in raw_comments:
            if not isinstance(comment, dict) or comment.get("kind") != "t1":
                continue
            c_data = comment.get("data", {})
            author = c_data.get("author")
            body = c_data.get("body")

            if author in ("[deleted]", "[removed]", "AutoModerator") or not body:
                continue

            cleaned_comments.append(
                {
                    "author": author,
                    "body": body,
                    "ups": c_data.get("ups"),
                    "created_utc": c_data.get("created_utc"),
                }
            )

            if len(cleaned_comments) >= comment_limit:
                break

        result_payload = {
            "source_url": json_url,
            "post": cleaned_post,
            "top_comments": cleaned_comments,
        }

        return json.dumps(result_payload, ensure_ascii=False, indent=2)

    except Exception as exc:
        return f"Error in reddit fetch: {exc}"
