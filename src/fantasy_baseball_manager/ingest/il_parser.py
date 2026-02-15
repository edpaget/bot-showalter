import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ILParseResult:
    transaction_type: str
    il_type: str
    injury_description: str | None


_PLACEMENT_RE = re.compile(r"placed .+ on the (\d+)-day injured list")
_ACTIVATION_RE = re.compile(r"activated .+ from the (\d+)-day injured list")
_TRANSFER_RE = re.compile(r"transferred .+ from the \d+-day injured list to the (\d+)-day injured list")
_INJURY_RE = re.compile(r"injured list.*?\.\s+(.+?)\.?\s*$")


def parse_il_transaction(description: str) -> ILParseResult | None:
    """Parse an MLB transaction description into structured IL data.

    Returns None if the description is not an IL-related transaction.
    """
    transfer_match = _TRANSFER_RE.search(description)
    if transfer_match:
        il_type = transfer_match.group(1)
        injury = _extract_injury(description)
        return ILParseResult(transaction_type="transfer", il_type=il_type, injury_description=injury)

    placement_match = _PLACEMENT_RE.search(description)
    if placement_match:
        il_type = placement_match.group(1)
        injury = _extract_injury(description)
        return ILParseResult(transaction_type="placement", il_type=il_type, injury_description=injury)

    activation_match = _ACTIVATION_RE.search(description)
    if activation_match:
        il_type = activation_match.group(1)
        injury = _extract_injury(description)
        return ILParseResult(transaction_type="activation", il_type=il_type, injury_description=injury)

    return None


def _extract_injury(description: str) -> str | None:
    injury_match = _INJURY_RE.search(description)
    if injury_match:
        return injury_match.group(1)
    return None
