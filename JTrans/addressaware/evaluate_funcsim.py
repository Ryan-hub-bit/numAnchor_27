import os
import runpy
import sys

from paths import ensure_parent_under_data_root, ensure_under_data_root


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PALMTREE_FUNCSIM_EVAL = os.path.join(ROOT, "PalmTree", "addressaware", "funcsim", "evaluate.py")


def _normalize_cli_paths(argv):
    must_exist = {
        "--function_blocks",
        "--funcsim_pairs",
        "--vocab",
        "--eval_pool",
        "--checkpoint",
    }
    writable = {"--output", "--log_dir"}
    normalized = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        normalized.append(arg)
        if arg in must_exist and i + 1 < len(argv):
            normalized.append(ensure_under_data_root(argv[i + 1], arg, must_exist=True))
            i += 2
            continue
        if arg in writable and i + 1 < len(argv):
            normalized.append(ensure_parent_under_data_root(argv[i + 1], arg))
            i += 2
            continue
        i += 1
    return normalized


if __name__ == "__main__":
    sys.argv = _normalize_cli_paths(sys.argv)
    sys.argv[0] = PALMTREE_FUNCSIM_EVAL
    runpy.run_path(PALMTREE_FUNCSIM_EVAL, run_name="__main__")
