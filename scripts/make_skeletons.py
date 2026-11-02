#!/usr/bin/env python3
"""Derive the student skeleton files in finetune/ from the completed versions
in extra/.

Each (find, replace) pair in scripts/lab_gaps.json carves out exactly one
teaching moment. Everything else is byte-identical, so `code -d` shows students
only the gaps.

    usage:  python scripts/make_skeletons.py [--check]

  (no flag)  rewrite finetune/*.py from extra/*_complete.txt
  --check    verify the skeletons on disk match what would be generated,
             and exit non-zero if they do not. Use this in CI.

Edit a completed file, then rerun this. Never hand-edit finetune/*.py --
it is generated, and your edit will be overwritten.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRA = os.path.join(ROOT, "extra")
DEST = os.path.join(ROOT, "finetune")
SPEC = os.path.join(ROOT, "scripts", "lab_gaps.json")


def build(stem, gaps):
    src = os.path.join(EXTRA, f"{stem}_complete.txt")
    text = open(src).read()
    for find, repl in gaps:
        if find not in text:
            raise SystemExit(
                f"ERROR {stem}: gap text no longer present in {src}.\n"
                f"  Looked for:\n    {find[:200]!r}\n"
                f"  The completed file changed around a gap -- update "
                f"scripts/lab_gaps.json to match.")
        if text.count(find) != 1:
            raise SystemExit(
                f"ERROR {stem}: gap text appears {text.count(find)} times "
                f"(must be exactly 1):\n    {find[:200]!r}")
        text = text.replace(find, repl)
    # A skeleton must still be valid Python even though it cannot run.
    compile(text, os.path.join(DEST, f"{stem}.py"), "exec")
    return text


def main():
    check = "--check" in sys.argv
    spec = json.load(open(SPEC))
    failed = False
    for stem, gaps in spec.items():
        text = build(stem, gaps)
        dst = os.path.join(DEST, f"{stem}.py")
        if check:
            current = open(dst).read() if os.path.exists(dst) else None
            if current != text:
                print(f"STALE  {dst}  (regenerate with "
                      f"python scripts/make_skeletons.py)")
                failed = True
            else:
                print(f"ok     {dst}  ({len(gaps)} gaps)")
        else:
            with open(dst, "w") as f:
                f.write(text)
            print(f"wrote  {dst}  ({len(gaps)} gaps, parses clean)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
