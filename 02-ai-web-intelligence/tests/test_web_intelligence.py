from dataclasses import dataclass
import json
import os
import unittest
from unittest.mock import MagicMock

from registry import ToolRegistry, registry
from web import (
    UnsafeUrlError,
    fetch_reddit_thread,
    fetch_url_safely,
    google_custom_search,
    read_url,
    validate_public_http_url,
)


@dataclass
class MockUser:
    id: int
    is_authenticated: bool = True
    is_staff: bool = False
    is_superuser: bool = False


def mock_public_dns(host: str, port: int):
    """Simulates a public DNS resolution returning 93.184.216.34 (example.com)."""
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def mock_private_dns(host: str, port: int):
    """Simulates a DNS rebinding / internal resolution to AWS metadata IP 169.254.169.254."""
    return [(2, 1, 6, "", ("169.254.169.254", port))]


def make_mock_response(
    body: bytes,
    status_code: int = 200,
    url: str = "https://example.com/article",
    headers: dict = None,
    json_data=None,
):
    resp = MagicMock()
    resp.status_code = status_code
    resp.url = url
    resp.headers = headers or {"Content-Length": str(len(body))}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        import requests

        resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}")
    resp.iter_content = lambda chunk_size=32768: (
        body[i : i + chunk_size] for i in range(0, len(body), chunk_size)
    )
    resp.json = MagicMock(return_value=json_data)
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestToolRegistryAndWebIntelligence(unittest.TestCase):
    def test_tool_registration_lookup_and_permissions(self) -> None:
        disabled_for_staff = set()
        custom_reg = ToolRegistry(disabled_tools_provider=lambda u: disabled_for_staff)

        @custom_reg.register(
            name="public_echo",
            description="Accessible to any authenticated user.",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
            staff_only=False,
        )
        def public_echo(msg: str = "hi") -> str:
            return f"echo:{msg}"

        @custom_reg.register(
            name="staff_report",
            description="Accessible to staff and superusers.",
            parameters={"type": "object", "properties": {}},
            staff_only=True,
        )
        def staff_report() -> str:
            return "staff_ok"

        @custom_reg.register(
            name="root_Diagnostic",
            description="Accessible only to superusers.",
            parameters={"type": "object", "properties": {}},
            staff_only=True,
            superuser_only=True,
        )
        def root_diagnostic() -> str:
            return "root_ok"

        anon = MockUser(id=0, is_authenticated=False)
        regular = MockUser(id=1, is_authenticated=True, is_staff=False, is_superuser=False)
        staff = MockUser(id=2, is_authenticated=True, is_staff=True, is_superuser=False)
        admin = MockUser(id=3, is_authenticated=True, is_staff=True, is_superuser=True)

        self.assertEqual(custom_reg.get_schemas_for_user(anon), [])
        self.assertEqual([s["function"]["name"] for s in custom_reg.get_schemas_for_user(regular)], ["public_echo"])
        self.assertEqual(
            [s["function"]["name"] for s in custom_reg.get_schemas_for_user(staff)],
            ["public_echo", "staff_report"],
        )
        self.assertEqual(len(custom_reg.get_schemas_for_user(admin)), 3)

        # Execution permission enforcement
        self.assertEqual(custom_reg.execute("public_echo", {"msg": "ok"}, user=regular), "echo:ok")
        self.assertIn("not permitted", custom_reg.execute("staff_report", user=regular))
        self.assertEqual(custom_reg.execute("staff_report", user=staff), "staff_ok")
        self.assertIn("not permitted", custom_reg.execute("root_Diagnostic", user=staff))
        self.assertEqual(custom_reg.execute("root_Diagnostic", user=admin), "root_ok")

        # User preference disablement filtering
        disabled_for_staff.add("staff_report")
        self.assertIn("not permitted or is disabled", custom_reg.execute("staff_report", user=staff))

        # Unregistered tool handling
        self.assertIn("not registered", custom_reg.execute("non_existent_tool", user=admin))

    def test_web_tools_are_registered_in_global_registry(self) -> None:
        registered_names = set(registry.get_map().keys())
        self.assertTrue({"read_url", "google_custom_search", "fetch_reddit_thread"}.issubset(registered_names))

    def test_ssrf_protection_blocks_private_loopback_and_metadata_urls(self) -> None:
        self.assertEqual(
            validate_public_http_url("https://example.com/blog", dns_resolver=mock_public_dns),
            "https://example.com/blog",
        )

        blocked_urls = [
            "file:///etc/passwd",
            "ftp://example.com/data",
            "http://localhost:8000/admin",
            "http://127.0.0.1:8080/internal",
            "http://10.0.0.15/secret",
            "http://192.168.1.1/router",
            "http://172.16.0.5/metrics",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]:8000/",
            "http://service.internal/api",
        ]
        for bad_url in blocked_urls:
            with self.assertRaises(UnsafeUrlError, msg=f"Should have blocked {bad_url}"):
                validate_public_http_url(bad_url, dns_resolver=mock_public_dns)

        # Public hostname resolving to private/metadata IP (DNS rebinding) must also be blocked
        with self.assertRaises(UnsafeUrlError):
            validate_public_http_url("https://rebinding.attacker.example/", dns_resolver=mock_private_dns)

    def test_response_size_limit_enforcement(self) -> None:
        oversized_body = b"A" * 2048

        def fake_get(url, **kwargs):
            return make_mock_response(oversized_body, url=url)

        with self.assertRaises(ValueError):
            fetch_url_safely(
                "https://example.com/huge",
                max_bytes=1024,
                http_get=fake_get,
                dns_resolver=mock_public_dns,
            )

    def test_read_url_content_extraction_and_pagination(self) -> None:
        paragraphs = "\n\n".join(
            f"<p>Paragraph {i}: Audience intelligence platforms help marketers understand user intent and digital trends across communities.</p>"
            for i in range(1, 140)
        )
        sample_html = f"<html><head><title>Audience Report</title></head><body><article><h1>Audience Insights</h1>{paragraphs}</article></body></html>".encode(
            "utf-8"
        )

        def fake_get(url, **kwargs):
            return make_mock_response(sample_html, url=url)

        page_1 = read_url("https://example.com/report", page=1, http_get=fake_get, dns_resolver=mock_public_dns)
        self.assertIn("[Page 1 of 3", page_1)
        self.assertIn("Audience Insights", page_1)

        page_2 = read_url("https://example.com/report", page=2, http_get=fake_get, dns_resolver=mock_public_dns)
        self.assertIn("[Page 2 of 3", page_2)

        page_99 = read_url("https://example.com/report", page=99, http_get=fake_get, dns_resolver=mock_public_dns)
        self.assertIn("only has 3 pages available", page_99)

        # SSRF attempt through read_url returns clean error string
        ssrf_res = read_url("http://169.254.169.254/latest/meta-data/", page=1, http_get=fake_get, dns_resolver=mock_public_dns)
        self.assertIn("Error extracting content", ssrf_res)
        self.assertIn("169.254.169.254", ssrf_res)

    def test_google_custom_search_and_error_handling(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "dummy_test_key"
        os.environ["GOOGLE_SEARCH_ENGINE_ID"] = "dummy_cx_id"

        search_payload = {
            "items": [
                {
                    "title": "Audience Research Guide",
                    "snippet": "How agencies map audience communities.",
                    "link": "https://example.com/guide",
                }
            ]
        }

        def fake_get(url, **kwargs):
            if "googleapis.com" in url:
                return make_mock_response(b"{}", url=url, json_data=search_payload)
            html = b"<html><body><main><h1>Audience Research Guide</h1><p>Detailed methodology for community analysis.</p></main></body></html>"
            return make_mock_response(html, url=url)

        res_json = google_custom_search("audience research", num=1, content=True, http_get=fake_get, dns_resolver=mock_public_dns)
        parsed = json.loads(res_json)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["title"], "Audience Research Guide")
        self.assertIn("Detailed methodology", parsed[0]["content"])

    def test_fetch_reddit_thread_filters_deleted_and_automoderator_comments(self) -> None:
        reddit_api_payload = [
            {
                "data": {
                    "children": [
                        {
                            "data": {
                                "title": "What tools do you use for audience intelligence?",
                                "subreddit": "marketing",
                                "author": "agency_lead",
                                "selftext": "Looking for recommendations on social listening and search analytics.",
                                "ups": 42,
                                "permalink": "/r/marketing/comments/abc123/what_tools/",
                                "created_utc": 1700000000,
                            }
                        }
                    ]
                }
            },
            {
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "author": "AutoModerator",
                                "body": "Please read the subreddit rules.",
                                "ups": 1,
                                "created_utc": 1700000010,
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "author": "[deleted]",
                                "body": "[removed]",
                                "ups": 0,
                                "created_utc": 1700000020,
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "author": "growth_strategist",
                                "body": "Combining Reddit thread analysis with search intent data gives the best signal.",
                                "ups": 28,
                                "created_utc": 1700000030,
                            },
                        },
                    ]
                }
            },
        ]

        raw_json_bytes = json.dumps(reddit_api_payload).encode("utf-8")

        def fake_get(url, **kwargs):
            return make_mock_response(raw_json_bytes, url=url)

        output = fetch_reddit_thread(
            "https://www.reddit.com/r/marketing/comments/abc123/what_tools/?utm_source=share",
            max_comments=10,
            http_get=fake_get,
            dns_resolver=mock_public_dns,
        )
        parsed = json.loads(output)
        self.assertEqual(parsed["post"]["subreddit"], "marketing")
        self.assertEqual(len(parsed["top_comments"]), 1)
        self.assertEqual(parsed["top_comments"][0]["author"], "growth_strategist")

        # Verify domain restriction prevents fetch_reddit_thread from querying non-Reddit domains
        bad_domain_out = fetch_reddit_thread(
            "https://evil.example.com/r/marketing",
            http_get=fake_get,
            dns_resolver=mock_public_dns,
        )
        self.assertIn("not in the allowed domain list", bad_domain_out)


if __name__ == "__main__":
    unittest.main()
