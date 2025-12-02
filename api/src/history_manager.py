"""
Conversation History Management Strategies

Provides multiple approaches to optimize chat history sent to LLM:
1. Sliding Window - Keep last N messages
2. Token Budget - Limit by token count
3. Summarization - Summarize older messages
4. Semantic Filtering - Keep relevant messages only
5. Hybrid - Combine multiple strategies
"""

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from dataclasses import dataclass
import tiktoken
import logging

logger = logging.getLogger(__name__)


@dataclass
class HistoryConfig:
    """Configuration for history management."""
    # Sliding window
    max_messages: int = 6

    # Token budget
    max_tokens: int = 4096
    model_name: str = "gpt-3.5-turbo"  # For token counting

    # Summarization
    summarize_after: int = 6  # Summarize messages older than this
    summary_max_tokens: int = 500


class HistoryStrategy(ABC):
    """Base class for history management strategies."""

    @abstractmethod
    def filter_history(
        self,
        messages: list[BaseMessage],
        current_query: str
    ) -> list[BaseMessage]:
        """Filter/transform conversation history."""
        pass


class SlidingWindowStrategy(HistoryStrategy):
    """
    Keep only the last N messages.

    Simple and effective for most use cases.
    Ensures consistent context size.
    """

    def __init__(self, max_messages: int = 6):
        self.max_messages = max_messages

    def filter_history(
        self,
        messages: list[BaseMessage],
        current_query: str
    ) -> list[BaseMessage]:
        if len(messages) <= self.max_messages:
            return messages

        # Keep last N messages, ensuring we don't split a user/assistant pair
        truncated = messages[-self.max_messages:]

        # If first message is from assistant, include the user message before it
        if truncated and isinstance(truncated[0], AIMessage):
            start_idx = len(messages) - self.max_messages - 1
            if start_idx >= 0:
                truncated = [messages[start_idx]] + truncated

        logger.debug(
            f"SlidingWindow: {len(messages)} -> {len(truncated)} messages")
        return truncated


class TokenBudgetStrategy(HistoryStrategy):
    """
    Keep messages within a token budget.

    More precise than message count.
    Respects actual model context limits.
    """

    def __init__(self, max_tokens: int = 4000, model: str = "gpt-3.5-turbo"):
        self.max_tokens = max_tokens
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback for models not in tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[BaseMessage]) -> int:
        """Count tokens in messages."""
        total = 0
        for msg in messages:
            # Add overhead for message structure
            total += 4  # role + formatting tokens
            total += len(self.encoding.encode(msg.content))
        return total

    def filter_history(
        self,
        messages: list[BaseMessage],
        current_query: str
    ) -> list[BaseMessage]:
        if not messages:
            return messages

        total_tokens = self.count_tokens(messages)

        if total_tokens <= self.max_tokens:
            return messages

        # Remove oldest messages until under budget
        filtered = list(messages[2:])
        while filtered and self.count_tokens(filtered) > self.max_tokens:
            # Remove oldest pair (user + assistant)
            if len(filtered) >= 2:
                filtered = filtered[2:]
            else:
                filtered = filtered[1:]

        logger.debug(
            f"TokenBudget: {len(messages)} messages ({total_tokens} tokens) "
            f"-> {len(filtered)} messages ({self.count_tokens(filtered)} tokens)"
        )
        return filtered


class SummarizationStrategy(HistoryStrategy):
    """
    Summarize older messages, keep recent ones intact.

    Preserves important context from long conversations.
    Adds some latency due to summarization call.
    """

    def __init__(
        self,
        llm: ChatOllama,
        summarize_after: int = 6,
        summary_max_tokens: int = 500
    ):
        self.llm = llm
        self.summarize_after = summarize_after
        self.summary_max_tokens = summary_max_tokens
        self._summary_cache: dict[str, str] = {}

    def _create_summary(self, messages: list[BaseMessage]) -> str:
        """Create a summary of messages."""
        if not messages:
            return ""

        # Create cache key from message contents
        cache_key = hash(tuple(m.content[:50] for m in messages))
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        # Format messages for summarization
        conversation = "\n".join([
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in messages
        ])

        summary_prompt = f"""Summarize the following conversation concisely, 
preserving key information, decisions, and context that might be relevant for future questions.
Keep the summary under {self.summary_max_tokens} tokens.

Conversation:
{conversation}
"""

        try:
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content
            self._summary_cache[cache_key] = summary
            return summary
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""

    def filter_history(
        self,
        messages: list[BaseMessage],
        current_query: str
    ) -> list[BaseMessage]:
        if len(messages) <= self.summarize_after:
            return messages

        # Split into old (to summarize) and recent (to keep)
        old_messages = messages[:-self.summarize_after]
        recent_messages = messages[-self.summarize_after:]

        # Summarize old messages
        summary = self._create_summary(old_messages)

        if summary:
            # Create a system message with the summary
            summary_message = SystemMessage(
                content=f"Summary of earlier conversation:\n{summary}"
            )
            result = [summary_message] + list(recent_messages)
        else:
            result = list(recent_messages)

        logger.debug(
            f"Summarization: {len(messages)} messages -> "
            f"summary + {len(recent_messages)} recent"
        )
        return result


strategies = {}
# Factory function


def create_history_manager(
    strategy: str = "sliding_window",
    llm: Optional[ChatOllama] = None,
    embeddings=None,
    config: Optional[HistoryConfig] = None
) -> HistoryStrategy:
    """
    Factory function to create history manager.

    Args:
        strategy: One of 'sliding_window', 'token_budget', 'summarization', 
                  'semantic', 'hybrid'
        llm: LLM instance (required for summarization)
        config: Configuration options

    Returns:
        HistoryStrategy instance
    """
    config = config or HistoryConfig()
    global strategies
    if not strategies:
        strategies = {
            "sliding_window": lambda: SlidingWindowStrategy(config.max_messages),
            "token_budget": lambda: TokenBudgetStrategy(
                config.max_tokens, config.model_name
            ),
            "summarization": lambda: SummarizationStrategy(
                llm, config.summarize_after, config.summary_max_tokens
            ) if llm else SlidingWindowStrategy(config.max_messages)
        }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    print(f"Creating History manager with strategy {strategy}")

    return strategies[strategy]()
