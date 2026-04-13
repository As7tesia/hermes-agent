"""Tests for Brave Search web backend integration."""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


class TestBraveRequest:
    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BRAVE_API_KEY", None)
            from tools.web_tools import _brave_request

            with pytest.raises(ValueError, match="BRAVE_API_KEY"):
                _brave_request("web/search", {"q": "test"})

    def test_sends_subscription_header(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {"BRAVE_API_KEY": "brave-test-key"}):
            with patch("tools.web_tools.httpx.get", return_value=mock_response) as mock_get:
                from tools.web_tools import _brave_request

                result = _brave_request("web/search", {"q": "hello"})

        assert result == {"web": {"results": []}}
        mock_get.assert_called_once()
        call = mock_get.call_args
        assert call.args[0] == "https://api.search.brave.com/res/v1/web/search"
        assert call.kwargs["params"] == {"q": "hello"}
        assert call.kwargs["headers"]["X-Subscription-Token"] == "brave-test-key"


class TestNormalizeBraveSearchResults:
    def test_includes_extra_snippets(self):
        from tools.web_tools import _normalize_brave_search_results

        result = _normalize_brave_search_results(
            {
                "web": {
                    "results": [
                        {
                            "title": "Python",
                            "url": "https://python.org",
                            "description": "Official website",
                            "extra_snippets": ["Docs", "Downloads"],
                        }
                    ]
                }
            }
        )

        assert result["success"] is True
        web = result["data"]["web"]
        assert web == [
            {
                "title": "Python",
                "url": "https://python.org",
                "description": "Official website Docs Downloads",
                "position": 1,
            }
        ]


class TestWebSearchBrave:
    def test_search_dispatches_to_brave(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Result",
                        "url": "https://example.com",
                        "description": "desc",
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        with patch("tools.web_tools._get_backend", return_value="brave"), \
             patch.dict(os.environ, {"BRAVE_API_KEY": "brave-test"}), \
             patch("tools.web_tools.httpx.get", return_value=mock_response), \
             patch("tools.interrupt.is_interrupted", return_value=False):
            from tools.web_tools import web_search_tool

            result = json.loads(web_search_tool("test query", limit=3))

        assert result["success"] is True
        assert result["data"]["web"][0]["title"] == "Result"
        assert result["data"]["web"][0]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_web_extract_dispatches_to_brave_direct_fetch():
    response = httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html><head><title>Example Title</title></head><body><main>Hello <b>world</b></main></body></html>",
        request=httpx.Request("GET", "https://example.com"),
    )

    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=response)
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = False

    with patch("tools.web_tools._get_backend", return_value="brave"), \
         patch("tools.web_tools.httpx.AsyncClient", return_value=mock_client), \
         patch("tools.web_tools.check_website_access", return_value=None), \
         patch("tools.web_tools.is_safe_url", return_value=True):
        from tools.web_tools import web_extract_tool

        result = json.loads(await web_extract_tool(["https://example.com"], use_llm_processing=False))

    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.com"
    assert result["results"][0]["title"] == "Example Title"
    assert "Hello world" in result["results"][0]["content"]
