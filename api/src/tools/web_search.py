"""
Web Search Tool

Provides real-time web search capabilities for the LLM
to find current information not in the knowledge base.
"""

from typing import List, Dict, Any, Optional
import logging

from langchain_core.tools import tool

from src.config import settings

logger = logging.getLogger(__name__)


class WebSearchProvider:
    """Base class for web search providers."""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        raise NotImplementedError


class DuckDuckGoProvider(WebSearchProvider):
    """DuckDuckGo search provider (free, no API key required)."""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            from duckduckgo_search import DDGS

            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
            return results
        except ImportError:
            logger.error("duckduckgo-search package not installed")
            raise RuntimeError(
                "DuckDuckGo search requires 'duckduckgo-search' package. "
                "Install with: pip install duckduckgo-search"
            )
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            raise


class SerpAPIProvider(WebSearchProvider):
    """SerpAPI search provider (requires API key)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            from serpapi import GoogleSearch

            params = {
                "q": query,
                "api_key": self.api_key,
                "num": max_results
            }
            search = GoogleSearch(params)
            data = search.get_dict()

            results = []
            for r in data.get("organic_results", [])[:max_results]:
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "snippet": r.get("snippet", "")
                })
            return results
        except ImportError:
            logger.error("google-search-results package not installed")
            raise RuntimeError(
                "SerpAPI requires 'google-search-results' package. "
                "Install with: pip install google-search-results"
            )
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            raise


class TavilyProvider(WebSearchProvider):
    """Tavily search provider (requires API key)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        try:
            from tavily import TavilyClient

            client = TavilyClient(api_key=self.api_key)
            response = client.search(query, max_results=max_results)

            results = []
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("content", "")
                })
            return results
        except ImportError:
            logger.error("tavily-python package not installed")
            raise RuntimeError(
                "Tavily requires 'tavily-python' package. "
                "Install with: pip install tavily-python"
            )
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise


def get_search_provider() -> WebSearchProvider:
    """Get the configured web search provider."""
    provider_name = settings.tools.web_search_provider.lower()

    if provider_name == "duckduckgo":
        return DuckDuckGoProvider()
    elif provider_name == "serpapi":
        if not settings.tools.serpapi_key:
            raise ValueError("SERPAPI_KEY not configured")
        return SerpAPIProvider(settings.tools.serpapi_key)
    elif provider_name == "tavily":
        if not settings.tools.tavily_key:
            raise ValueError("TAVILY_KEY not configured")
        return TavilyProvider(settings.tools.tavily_key)
    else:
        logger.warning(f"Unknown provider '{provider_name}', using DuckDuckGo")
        return DuckDuckGoProvider()


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information.

    Use this tool when:
    - The user asks about recent events or news
    - Information might have changed since the knowledge base was created
    - The knowledge base doesn't contain relevant information
    - You need to verify facts with current sources
    - The user explicitly asks for web search

    Args:
        query: The search query
        max_results: Maximum number of results to return (1-10, default 5)

    Returns:
        Formatted search results with titles, URLs, and snippets
    """
    try:
        # Clamp max_results to valid range
        max_results = max(1, min(settings.tools.web_search_max_results, max_results))

        provider = get_search_provider()
        results = provider.search(query, max_results)

        if not results:
            return f"No web results found for query: '{query}'"

        # Format results for LLM consumption
        formatted_results = []
        for i, r in enumerate(results, 1):
            formatted_results.append(
                f"[Result {i}]\n"
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['snippet']}\n"
            )

        header = f"Web search results for '{query}':\n\n"
        return header + "\n---\n".join(formatted_results)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Error performing web search: {str(e)}"


def get_web_search_tool_info() -> Dict[str, Any]:
    """Get metadata about the web search tool."""
    return {
        "name": "web_search",
        "description": web_search.description,
        "provider": settings.tools.web_search_provider,
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query",
                "required": True
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10)",
                "default": 5
            }
        }
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add api directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import argparse
    from src.config import settings

    parser = argparse.ArgumentParser(description="Test web_search tool")
    parser.add_argument("query", help="Search query")
    parser.add_argument("-n", "--max-results", type=int, default=5, help="Max results (default: 5)")
    parser.add_argument("-p", "--provider", default="duckduckgo", help="Provider: duckduckgo, serpapi, tavily")
    args = parser.parse_args()

    print(f"Provider: {args.provider}")
    print(f"Query: {args.query}")
    print(f"Max results: {args.max_results}")
    print("-" * 40)

    result = web_search.invoke({"query": args.query, "max_results": args.max_results})
    print(result)
