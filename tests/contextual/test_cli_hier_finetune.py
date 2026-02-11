"""Tests for the hier-finetune and build-identity CLI commands."""

from __future__ import annotations

import typer.testing

from fantasy_baseball_manager.contextual.cli import contextual_app


def _find_command(name: str) -> typer.models.CommandInfo | None:
    """Find a registered command by name in the contextual app."""
    for cmd in contextual_app.registered_commands:
        if cmd.name == name:
            return cmd
    return None


def _get_help_text() -> str:
    runner = typer.testing.CliRunner()
    result = runner.invoke(contextual_app, ["hier-finetune", "--help"])
    assert result.exit_code == 0
    return result.output


def _help_has_option(help_text: str, option_name: str) -> bool:
    """Check if an option appears in the help output, handling Rich formatting."""
    # The option name may be split across Rich box-drawing chars
    # Search for the option name with possible whitespace/box chars between chars
    return option_name in help_text


class TestHierFineTuneCommandRegistration:
    """Test that the hier-finetune command is properly registered."""

    def test_command_exists(self) -> None:
        cmd = _find_command("hier-finetune")
        assert cmd is not None, "hier-finetune command not registered"

    def test_command_callback_is_callable(self) -> None:
        cmd = _find_command("hier-finetune")
        assert cmd is not None
        assert callable(cmd.callback)

    def test_help_text_mentions_hierarchical(self) -> None:
        help_text = _get_help_text()
        assert "hierarchical" in help_text.lower()


class TestHierFineTuneCommandOptions:
    """Test that the hier-finetune command has the expected CLI options."""

    def test_has_base_model_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--base-model")

    def test_has_perspective_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--perspective")

    def test_has_identity_lr_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--identity-lr")

    def test_has_level3_lr_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--level3-lr")

    def test_has_head_lr_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--head-lr")

    def test_has_n_archetypes_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--n-archetypes")

    def test_has_archetype_model_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--archetype-model")

    def test_has_min_opportunities_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--min-opportunities")

    def test_has_profile_year_option(self) -> None:
        assert _help_has_option(_get_help_text(), "--profile-year")

    def test_has_shared_finetune_options(self) -> None:
        help_text = _get_help_text()
        for expected in [
            "--seasons",
            "--val-seasons",
            "--epochs",
            "--batch-size",
            "--context-window",
            "--target-mode",
            "--target-window",
            "--d-model",
            "--n-layers",
            "--n-heads",
            "--ff-dim",
        ]:
            assert _help_has_option(help_text, expected), f"Missing option: {expected}"


def _get_build_identity_help() -> str:
    runner = typer.testing.CliRunner()
    result = runner.invoke(contextual_app, ["build-identity", "--help"])
    assert result.exit_code == 0
    return result.output


class TestBuildIdentityCommandRegistration:
    """Test that the build-identity command is properly registered."""

    def test_command_exists(self) -> None:
        cmd = _find_command("build-identity")
        assert cmd is not None, "build-identity command not registered"

    def test_command_callback_is_callable(self) -> None:
        cmd = _find_command("build-identity")
        assert cmd is not None
        assert callable(cmd.callback)

    def test_help_text_mentions_archetype(self) -> None:
        help_text = _get_build_identity_help()
        assert "archetype" in help_text.lower()


class TestBuildIdentityCommandOptions:
    """Test that the build-identity command has the expected CLI options."""

    def test_has_perspective_option(self) -> None:
        assert _help_has_option(_get_build_identity_help(), "--perspective")

    def test_has_n_archetypes_option(self) -> None:
        assert _help_has_option(_get_build_identity_help(), "--n-archetypes")

    def test_has_min_opportunities_option(self) -> None:
        assert _help_has_option(_get_build_identity_help(), "--min-opportunities")

    def test_has_profile_year_option(self) -> None:
        assert _help_has_option(_get_build_identity_help(), "--profile-year")

    def test_has_name_option(self) -> None:
        assert _help_has_option(_get_build_identity_help(), "--name")
