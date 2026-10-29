import argparse
import re
import subprocess
from pathlib import Path


FUNC_RE = re.compile(r"^([0-9a-fA-F]+) <[^>]*>:$")
INSN_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F][0-9a-fA-F](?:\s+[0-9a-fA-F][0-9a-fA-F])*)\s+(.*)$")
SYMBOL_ADDR_RE = re.compile(r"\b([0-9a-fA-F]+)\s+<[^>]+>")
COMMENT_RE = re.compile(r"\s+#\s+.*$")
PTR_RE = re.compile(r"\b(?:BYTE|WORD|DWORD|QWORD|XMMWORD|YMMWORD|ZMMWORD)\s+PTR\s+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")


def normalize_instruction(text: str) -> str:
    text = COMMENT_RE.sub("", text)
    text = SYMBOL_ADDR_RE.sub(lambda m: f"0x{m.group(1)}", text)
    text = PTR_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def export(binary: str, output: str) -> None:
    result = subprocess.run(
        ["objdump", "-d", "-Mintel", binary],
        check=True,
        capture_output=True,
        text=True,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        first_func = True
        for line in result.stdout.splitlines():
            func_match = FUNC_RE.match(line)
            if func_match:
                if not first_func:
                    fout.write("\n")
                first_func = False
                fout.write("=" * 80 + "\n")
                fout.write(f"Function @ 0x{func_match.group(1)}\n")
                fout.write("=" * 80 + "\n")
                continue

            insn_match = INSN_RE.match(line)
            if insn_match:
                ea = insn_match.group(1)
                insn_text = normalize_instruction(insn_match.group(3))
                if insn_text:
                    fout.write(f"0x{ea}: {insn_text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export numeric Intel disassembly grouped by function")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    export(args.binary, args.output)


if __name__ == "__main__":
    main()
