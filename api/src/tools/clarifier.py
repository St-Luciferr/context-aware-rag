"""
Query Clarification Tool

Allows the LLM to request clarification from the user
when a query is ambiguous or lacks necessary information.
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


# Store pending clarification requests
# In production, this would be handled by the conversation state
_pending_clarifications: Dict[str, Dict] = {}


@tool
def clarify_query(
    clarification_question: str,
    options: Optional[List[str]] = None,
    reason: str = ""
) -> str:
    """Request clarification from the user when the query is ambiguous.

    Use this tool when:
    - The user's question is ambiguous or could have multiple interpretations
    - You need specific details to provide an accurate answer
    - The user's intent is unclear
    - Additional context would significantly improve the response

    Args:
        clarification_question: The specific question to ask the user
        options: Optional list of possible choices to present (max 5)
        reason: Brief explanation of why clarification is needed

    Returns:
        A formatted clarification request that will be shown to the user
    """
    try:
        # Build the clarification response
        parts = []

        if reason:
            parts.append(f"**Why I'm asking:** {reason}")

        parts.append(f"\n**Clarification needed:**\n{clarification_question}")

        if options and len(options) > 0:
            # Limit to 5 options
            options = options[:5]
            options_text = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(options)])
            parts.append(f"\n**Possible options:**\n{options_text}")

        clarification_text = "\n".join(parts)

        logger.info(f"Clarification requested: {clarification_question}")

        return f"[CLARIFICATION_NEEDED]\n{clarification_text}"

    except Exception as e:
        logger.error(f"Clarification tool error: {e}")
        return f"Error creating clarification request: {str(e)}"


def get_clarify_tool_info() -> Dict[str, Any]:
    """Get metadata about the clarification tool."""
    return {
        "name": "clarify_query",
        "description": clarify_query.description,
        "parameters": {
            "clarification_question": {
                "type": "string",
                "description": "The specific question to ask the user",
                "required": True
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of possible choices (max 5)",
                "required": False
            },
            "reason": {
                "type": "string",
                "description": "Brief explanation of why clarification is needed",
                "required": False
            }
        }
    }


def is_clarification_response(text: str) -> bool:
    """Check if a response contains a clarification request.

    Args:
        text: The response text to check

    Returns:
        True if this is a clarification request
    """
    return text.strip().startswith("[CLARIFICATION_NEEDED]")


def parse_clarification_response(text: str) -> Dict[str, Any]:
    """Parse a clarification response into structured data.

    Args:
        text: The clarification response text

    Returns:
        Dictionary with clarification details
    """
    if not is_clarification_response(text):
        return {"is_clarification": False}

    # Remove the marker
    content = text.replace("[CLARIFICATION_NEEDED]", "").strip()

    return {
        "is_clarification": True,
        "content": content
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    # Add api directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import argparse

    parser = argparse.ArgumentParser(description="Test clarify_query tool")
    parser.add_argument("question", help="Clarification question to ask")
    parser.add_argument("-r", "--reason", default="", help="Reason for asking")
    parser.add_argument("-o", "--options", nargs="+", help="List of options")
    args = parser.parse_args()

    print(f"Question: {args.question}")
    if args.reason:
        print(f"Reason: {args.reason}")
    if args.options:
        print(f"Options: {args.options}")
    print("-" * 40)

    params = {"clarification_question": args.question, "reason": args.reason}
    if args.options:
        params["options"] = args.options

    result = clarify_query.invoke(params)
    print(result)
