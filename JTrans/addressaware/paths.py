import os


DEFAULT_DATA_ROOT = "/path/to/numAnchor_artifact/numAnchor"


def data_root() -> str:
    return os.path.abspath(os.environ.get("JTRANS_ADDRESSAWARE_DATA_ROOT", DEFAULT_DATA_ROOT))


def _normalize(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def ensure_under_data_root(path: str, label: str, must_exist: bool = False) -> str:
    normalized = _normalize(path)
    root = data_root()
    try:
        common = os.path.commonpath([normalized, root])
    except ValueError as exc:
        raise ValueError(f"{label} must be under {root}: {path}") from exc
    if common != root:
        raise ValueError(f"{label} must be under {root}: {path}")
    if must_exist and not os.path.exists(normalized):
        raise FileNotFoundError(f"{label} does not exist: {normalized}")
    return normalized


def ensure_parent_under_data_root(path: str, label: str) -> str:
    normalized = _normalize(path)
    parent = os.path.dirname(normalized) or normalized
    ensure_under_data_root(parent, f"{label} parent", must_exist=False)
    return normalized
