"""
Helpers for categorizing data sections and formatting section-specific
data-address tokens such as `daddr_rodata(...)`.
"""

from __future__ import annotations

SECTION_KIND_RODATA = 1
SECTION_KIND_BSS = 2
SECTION_KIND_DATA = 3
SECTION_KIND_OTHER = 4

SECTION_KIND_NAMES = {
    SECTION_KIND_RODATA: ".rodata",
    SECTION_KIND_BSS: ".bss",
    SECTION_KIND_DATA: ".data",
    SECTION_KIND_OTHER: "other-data",
}

SECTION_KIND_TOKENS = {
    SECTION_KIND_RODATA: "daddr_rodata",
    SECTION_KIND_BSS: "daddr_bss",
    SECTION_KIND_DATA: "daddr_data",
    SECTION_KIND_OTHER: "daddr_other",
}

SECTION_TOKEN_TO_KIND = {token: kind for kind, token in SECTION_KIND_TOKENS.items()}


def parse_section_kind(raw_value) -> int:
    try:
        parsed = int(float(raw_value))
    except (TypeError, ValueError):
        return SECTION_KIND_OTHER
    return parsed if parsed in SECTION_KIND_NAMES else SECTION_KIND_OTHER


def daddr_token_for_kind(section_kind_id: int) -> str:
    return SECTION_KIND_TOKENS.get(parse_section_kind(section_kind_id), SECTION_KIND_TOKENS[SECTION_KIND_OTHER])


def parse_section_token(token_name: str) -> int:
    return SECTION_TOKEN_TO_KIND.get((token_name or "").strip(), SECTION_KIND_OTHER)


def classify_data_section(section_name: str) -> tuple[bool, int]:
    """
    Returns:
      (is_data_like, section_kind_id)

    `.rodata`, `.bss`, and exact `.data` get dedicated IDs.
    All remaining data-like sections collapse into `other-data`.
    """
    sname = (section_name or "").strip().lower()
    if not sname:
        return False, SECTION_KIND_OTHER

    if ".rodata" in sname or sname in {"rodata", ".rdata", "rdata"}:
        return True, SECTION_KIND_RODATA
    if ".bss" in sname or sname == "bss":
        return True, SECTION_KIND_BSS
    if sname in {".data", "data"}:
        return True, SECTION_KIND_DATA

    other_markers = (
        ".data.",
        "data.",
        ".got",
        "got",
        ".got.plt",
        ".data.rel",
        ".sdata",
        ".sbss",
        ".tdata",
        ".tbss",
        ".idata",
        ".init_array",
        ".fini_array",
        ".ctors",
        ".dtors",
        ".jcr",
        ".data1",
        ".bss.",
        ".rodata.",
        ".rdata.",
    )
    if any(marker in sname or sname == marker.lstrip(".") for marker in other_markers):
        return True, SECTION_KIND_OTHER

    if "data" in sname or "bss" in sname or "rodata" in sname:
        return True, SECTION_KIND_OTHER

    return False, SECTION_KIND_OTHER


def format_daddr_payload(section_pos: float, offset_in_section: float, section_kind_id: int) -> str:
    return (
        f"{max(0.0, min(1.0, float(section_pos))):.8f}:"
        f"{max(0.0, min(1.0, float(offset_in_section))):.8f}:"
        f"{parse_section_kind(section_kind_id)}"
    )


def format_daddr_token_payload(section_pos: float, offset_in_section: float) -> str:
    return (
        f"{max(0.0, min(1.0, float(section_pos))):.8f}:"
        f"{max(0.0, min(1.0, float(offset_in_section))):.8f}"
    )


def format_daddr_operand(address_hex: str, section_pos: float, offset_in_section: float, section_kind_id: int) -> str:
    token_name = daddr_token_for_kind(section_kind_id)
    payload = format_daddr_token_payload(section_pos, offset_in_section)
    return f"{token_name}({address_hex}:{payload})"


def default_daddr_operand(address_hex: str) -> str:
    return format_daddr_operand(address_hex, 0.0, 0.0, SECTION_KIND_OTHER)


def default_daddr_payload() -> str:
    return format_daddr_payload(0.0, 0.0, SECTION_KIND_OTHER)
