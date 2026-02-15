"""
LLM Tool Calling Module

Provides tools that the LLM can invoke during conversations:
- retrieve_documents: Additional document retrieval from vector store
- web_search: Real-time web search for current information
- clarify_query: Request clarification from user

Usage:
    from src.tools import get_enabled_tools, ToolRegistry

    # Get all enabled tools for binding to LLM
    tools = get_enabled_tools()

    # Or use registry directly
    registry = ToolRegistry()
    tool = registry.get_tool("web_search")
"""

from typing import List, Dict, Callable, Optional
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry for managing available LLM tools."""

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, Callable] = {}

    def __new__(cls) -> "ToolRegistry":
        """Singleton pattern for tool registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, name: str, tool: Callable) -> None:
        """Register a tool with the registry.

        Args:
            name: Unique tool identifier
            tool: The tool function (decorated with @tool)
        """
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool by name.

        Args:
            name: Tool identifier

        Returns:
            The tool function or None if not found
        """
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Callable]:
        """Get all registered tools."""
        return self._tools.copy()

    def get_enabled_tools(self) -> List[Callable]:
        """Get only tools that are enabled in configuration.

        Returns:
            List of enabled tool functions
        """
        enabled_names = settings.tools.enabled_tools_list
        enabled = []
        for name in enabled_names:
            tool = self._tools.get(name)
            if tool:
                enabled.append(tool)
            else:
                logger.warning(f"Enabled tool '{name}' not found in registry")
        return enabled

    def list_tools(self) -> List[Dict]:
        """List all registered tools with metadata.

        Returns:
            List of tool info dictionaries
        """
        tools_info = []
        for name, tool in self._tools.items():
            info = {
                "name": name,
                "description": getattr(tool, "description", "No description"),
                "enabled": name in settings.tools.enabled_tools_list,
            }
            tools_info.append(info)
        return tools_info


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def get_enabled_tools() -> List[Callable]:
    """Get list of enabled tool functions for LLM binding.

    Returns:
        List of tool functions that are enabled in config
    """
    return _registry.get_enabled_tools()


def get_all_tools() -> Dict[str, Callable]:
    """Get all registered tools."""
    return _registry.get_all_tools()


def register_tool(name: str):
    """Decorator to register a tool with the registry.

    Usage:
        @register_tool("my_tool")
        @tool
        def my_tool(query: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        _registry.register(name, func)
        return func
    return decorator


# Import tools and register them
# These imports must be at the bottom to avoid circular imports
from src.tools.retrieval import retrieve_documents
from src.tools.web_search import web_search
from src.tools.clarifier import clarify_query

# Register tools with the registry
_registry.register("retrieve_documents", retrieve_documents)
_registry.register("web_search", web_search)
_registry.register("clarify_query", clarify_query)

__all__ = [
    "ToolRegistry",
    "get_registry",
    "get_enabled_tools",
    "get_all_tools",
    "register_tool",
    "retrieve_documents",
    "web_search",
    "clarify_query",
]
