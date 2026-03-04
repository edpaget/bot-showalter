from typing import Annotated

import typer

import fantasy_baseball_manager.models  # noqa: F401 — trigger model registration
from fantasy_baseball_manager.cli._logging import configure_logging
from fantasy_baseball_manager.cli.commands.compare_features import compare_features_cmd
from fantasy_baseball_manager.cli.commands.compute import compute_app
from fantasy_baseball_manager.cli.commands.datasets import datasets_app
from fantasy_baseball_manager.cli.commands.draft import draft_app
from fantasy_baseball_manager.cli.commands.experiment import experiment_app
from fantasy_baseball_manager.cli.commands.feature import feature_app
from fantasy_baseball_manager.cli.commands.ingest import ingest_app
from fantasy_baseball_manager.cli.commands.keeper import keeper_app
from fantasy_baseball_manager.cli.commands.marginal_value import marginal_value_cmd
from fantasy_baseball_manager.cli.commands.mock_draft import mock_app
from fantasy_baseball_manager.cli.commands.model import (
    ablate,
    coverage,
    evaluate,
    finetune,
    gate,
    predict,
    prepare,
    sweep,
    train,
    tune,
)
from fantasy_baseball_manager.cli.commands.profile import profile_app
from fantasy_baseball_manager.cli.commands.projections import projections_app
from fantasy_baseball_manager.cli.commands.quick_eval import quick_eval_cmd
from fantasy_baseball_manager.cli.commands.report import report_app
from fantasy_baseball_manager.cli.commands.residuals import residuals_app
from fantasy_baseball_manager.cli.commands.runs import runs_app
from fantasy_baseball_manager.cli.commands.standalone import (
    chat_cmd,
    compare_cmd,
    discord_cmd,
    features,
    import_cmd,
    info,
    list_cmd,
)
from fantasy_baseball_manager.cli.commands.validate import validate_app
from fantasy_baseball_manager.cli.commands.valuations import valuations_app
from fantasy_baseball_manager.cli.commands.yahoo import yahoo_app

app = typer.Typer(name="fbm", help="Fantasy Baseball Manager — projection model CLI")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging")] = False,
) -> None:
    """Fantasy Baseball Manager — projection model CLI."""
    configure_logging(verbose=verbose)
    if ctx.invoked_subcommand is None:
        raise typer.Exit()


# Top-level commands
app.command()(prepare)
app.command()(train)
app.command()(evaluate)
app.command()(predict)
app.command()(finetune)
app.command()(ablate)
app.command()(tune)
app.command()(sweep)
app.command()(gate)
app.command()(coverage)
app.command("quick-eval")(quick_eval_cmd)
app.command("marginal-value")(marginal_value_cmd)
app.command("compare-features")(compare_features_cmd)
app.command("list")(list_cmd)
app.command()(info)
app.command()(features)
app.command("import")(import_cmd)
app.command("compare")(compare_cmd)
app.command("chat")(chat_cmd)
app.command("discord")(discord_cmd)

# Sub-app groups
app.add_typer(runs_app, name="runs")
app.add_typer(datasets_app, name="datasets")
app.add_typer(projections_app, name="projections")
app.add_typer(valuations_app, name="valuations")
app.add_typer(ingest_app, name="ingest")
app.add_typer(compute_app, name="compute")
app.add_typer(report_app, name="report")
app.add_typer(residuals_app, name="residuals")
draft_app.add_typer(mock_app, name="mock")
app.add_typer(draft_app, name="draft")
app.add_typer(experiment_app, name="experiment")
app.add_typer(feature_app, name="feature")
app.add_typer(keeper_app, name="keeper")
app.add_typer(profile_app, name="profile")
app.add_typer(validate_app, name="validate")
app.add_typer(yahoo_app, name="yahoo")
