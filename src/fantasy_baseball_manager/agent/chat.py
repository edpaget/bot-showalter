from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessageChunk

from fantasy_baseball_manager.agent.stream import extract_text

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def run_chat(agent: CompiledStateGraph) -> None:
    """Run an interactive chat loop with the agent."""
    print("Fantasy Baseball Manager — Chat")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            text = input("You: ")
        except EOFError:
            print("\nGoodbye!")
            return
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return

        stripped = text.strip()
        if stripped.lower() in ("quit", "exit"):
            print("Goodbye!")
            return
        if not stripped:
            continue

        print("Assistant: ", end="", flush=True)
        for chunk, metadata in agent.stream(
            {"messages": [("user", stripped)]},
            stream_mode="messages",
        ):
            if (
                isinstance(chunk, AIMessageChunk)
                and isinstance(metadata, dict)
                and metadata.get("langgraph_node") == "agent"
            ):
                _print_chunk_content(chunk.content)
        print()


def _print_chunk_content(content: str | list[dict[str, Any]]) -> None:
    """Print the text portion of an AIMessageChunk's content."""
    text = extract_text(content)
    if text:
        print(text, end="", flush=True)
