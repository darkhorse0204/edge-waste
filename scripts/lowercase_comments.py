# lowercase_comments.py - turns every code comment (the text after a hash sign) in python files into lowercase, leaving code and docstrings alone
"""usage: python scripts/lowercase_comments.py <folder> [<folder> ...]"""
import io
import re
import sys
import tokenize
from pathlib import Path

# comments that tools read must keep their exact spelling
KEEP = re.compile(r"#\s*(noqa|type:|pragma|pylint|fmt:|isort|ruff|!|-\*-|%%)", re.I)


def fix(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    changed = False
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and not KEEP.match(tok.string):
            low = tok.string.lower()
            if low != tok.string:
                row, col = tok.start
                line = lines[row - 1]
                lines[row - 1] = line[:col] + low + line[col + len(tok.string):]
                changed = True
    if changed:
        path.write_text("".join(lines), encoding="utf-8", newline="")
    return changed


if __name__ == "__main__":
    n = 0
    for folder in sys.argv[1:]:
        for p in Path(folder).rglob("*.py"):
            n += fix(p)
    print("files changed:", n)
