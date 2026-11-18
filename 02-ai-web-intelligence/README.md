# AI Web Intelligence (`registry.py` & `web.py`)

## The Problem

Museed includes an internal AI assistant that searches the web, reads articles, and inspects Reddit threads to research audience discussions.

Building the tool layer for this raised three practical issues:

1. Keeping OpenAI JSON tool schemas, permission checks (`staff_only`, `superuser_only`, and per-user tool toggles), and Python functions in sync without duplicating definitions.
2. Extracting readable text from web pages and Reddit threads without flooding the LLM context window with HTML boilerplate, deleted comments, or oversized payloads.
3. Preventing the agent from fetching internal network addresses or cloud metadata endpoints (SSRF) when following URLs.

## The Solution

We separated tool registration (`registry.py`) from the web tools (`web.py`):

- `@register_tool` attaches the JSON schema, UI status label, and role requirements directly to each Python function.
- `web.py` defines three tools (`read_url`, `google_custom_search`, and `fetch_reddit_thread`) backed by an HTTP helper that validates destination IPs, caps response sizes, strips HTML boilerplate with `trafilatura`, and paginates text output.

## How It Works

- **`ToolRegistry` (`registry.py`):** Filters available tools per user (`get_schemas_for_user` and `execute`). If a user lacks permissions or disabled a tool, it is excluded from the schemas sent to the model and blocked at execution time. Tool exceptions are caught and returned as error strings so the agent loop can recover instead of crashing.
- **`read_url` (`web.py`):** Downloads a page, extracts the main body text as Markdown using `trafilatura`, and returns 8,000-character pages with a header telling the model how to request the next page.
- **`google_custom_search`:** Calls the Google Custom Search JSON API and optionally runs `read_url(..., page=1)` on each result in the same turn.
- **`fetch_reddit_thread`:** Appends `.json` to public Reddit thread URLs, filters out `AutoModerator`, `[deleted]`, and `[removed]` entries, and returns the post body plus the top $N$ top-level comments.

## Tests & Limitations

```bash
python3 -m unittest discover -s tests -v
```

**Production vs. Sample Changes:**
- Decoupled Django ORM/settings imports so the modules run standalone using environment variables (`GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`, `REDDIT_COOKIE`) and an injectable `disabled_tools_provider` callback.
- Added `validate_public_http_url` and `fetch_url_safely` in `web.py` to block private/loopback/link-local IPs, enforce a 5 MB download cap, and restrict Reddit fetches to `reddit.com`/`redd.it`.

**Limitations:**
- `read_url` fetches static HTML over HTTP and does not execute client-side JavaScript (no headless browser/Playwright).
- The SSE streaming chat view (`ChatStreamView`) that drives the agent loop is part of the main Django app and not included here.
