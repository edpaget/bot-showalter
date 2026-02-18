from dataclasses import dataclass


@dataclass(frozen=True)
class FbmError:
    message: str


@dataclass(frozen=True)
class DispatchError(FbmError):
    model_name: str
    operation: str


@dataclass(frozen=True)
class IngestError(FbmError):
    source_type: str
    source_detail: str
    target_table: str


@dataclass(frozen=True)
class ConfigError(FbmError):
    unrecognized_keys: tuple[str, ...] = ()
