"""CLI command for the interactive chat REPL.

Provides the `fantasy chat` command that launches an interactive session
with the fantasy baseball agent.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Annotated

import typer

from fantasy_baseball_manager.agent.core import DEFAULT_MODEL, create_agent, run, stream

chat_app = typer.Typer(help="Interactive chat with the fantasy baseball assistant.")


def _print_streaming(text: str) -> None:
    """Print text without newline and flush immediately for streaming effect."""
    sys.stdout.write(text)
    sys.stdout.flush()


async def _run_streaming(agent: object, message: str) -> None:
    """Run the agent with streaming output."""
    async for chunk in stream(agent, message):  # type: ignore[arg-type]
        _print_streaming(chunk)
    print()  # Final newline


@chat_app.command(name="start")
def chat_start(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use for the agent."),
    ] = DEFAULT_MODEL,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Disable streaming output."),
    ] = False,
) -> None:
    """Start an interactive chat session with the fantasy baseball assistant.

    Type your questions and press Enter to get responses. Type 'quit', 'exit',
    or 'q' to end the session. Use Ctrl+C to interrupt at any time.
    """
    typer.echo(f"Fantasy Baseball Assistant (model: {model})")
    typer.echo("Type 'quit' or 'exit' to end the session.")
    typer.echo("-" * 50)
    typer.echo()

    agent = create_agent(model_name=model)

    while True:
        try:
            # Read user input
            user_input = typer.prompt("You", prompt_suffix="> ").strip()

            if not user_input:
                continue

            # Check for exit commands
            if user_input.lower() in ("quit", "exit", "q"):
                typer.echo("Goodbye!")
                break

            # Get response
            typer.echo()
            if no_stream:
                response = run(agent, user_input)
                typer.echo(response)
            else:
                # Run async streaming
                asyncio.run(_run_streaming(agent, user_input))

            typer.echo()

        except KeyboardInterrupt:
            typer.echo("\nInterrupted. Type 'quit' to exit or continue chatting.")
            typer.echo()
        except EOFError:
            typer.echo("\nGoodbye!")
            break


@chat_app.command(name="ask")
def chat_ask(
    question: Annotated[
        str,
        typer.Argument(help="The question to ask the assistant."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use for the agent."),
    ] = DEFAULT_MODEL,
) -> None:
    """Ask a single question and get a response (non-interactive).

    Example:
        fantasy chat ask "Who are the top 5 projected batters for 2025?"
    """
    agent = create_agent(model_name=model)
    response = run(agent, question)
    typer.echo(response)


# Default command - start interactive session
@chat_app.callback(invoke_without_command=True)
def chat_default(
    ctx: typer.Context,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Model to use for the agent."),
    ] = DEFAULT_MODEL,
    no_stream: Annotated[
        bool,
        typer.Option("--no-stream", help="Disable streaming output."),
    ] = False,
) -> None:
    """Interactive chat with the fantasy baseball assistant."""
    if ctx.invoked_subcommand is None:
        chat_start(model=model, no_stream=no_stream)
