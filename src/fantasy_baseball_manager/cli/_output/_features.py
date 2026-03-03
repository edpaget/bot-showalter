from rich.table import Table

from fantasy_baseball_manager.cli._output._common import console
from fantasy_baseball_manager.features import (
    AnyFeature,
    DeltaFeature,
    DerivedTransformFeature,
    TransformFeature,
)


def print_features(model_name: str, features: tuple[AnyFeature, ...]) -> None:
    console.print(f"Features for model [bold]'{model_name}'[/bold] ({len(features)} features):")
    table = Table(show_header=True, show_edge=False, pad_edge=False)
    table.add_column("Name")
    table.add_column("Details")
    for f in features:
        if isinstance(f, DeltaFeature):
            table.add_row(f.name, f"delta({f.left.name} - {f.right.name})")
        elif isinstance(f, TransformFeature):
            outputs = ", ".join(f.outputs)
            table.add_row(f.name, f"{f.source.value} transform → {outputs}")
        elif isinstance(f, DerivedTransformFeature):
            outputs = ", ".join(f.outputs)
            table.add_row(f.name, f"derived transform → {outputs}")
        elif f.computed:
            table.add_row(f.name, f"{f.source.value} computed={f.computed}")
        else:
            detail = f"{f.source.value}.{f.column}"
            if f.lag:
                detail += f" lag={f.lag}"
            if f.system:
                detail += f" system={f.system}"
            table.add_row(f.name, detail)
    console.print(table)
