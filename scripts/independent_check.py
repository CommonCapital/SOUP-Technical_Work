"""H9(c): an independent re-implementation of the Countdown checker, used to
re-score saved answers and measure agreement with the main checker
(src/soupgrpo/reward.py). Shares no code with it:

  - extraction by string search (last "</answer>", then the last "<answer>"
    before it), not a regular expression;
  - a hand-written tokenizer + recursive-descent parser over + - * / ( ),
    not Python's ast;
  - exact arithmetic with Fraction; integer literals only; no unary minus.

Same rules (as specified in the task, not copied from the code): every given
number used exactly once, no other numbers, value equals the target; an
optional "= <target>" on either side; unicode × ÷ − accepted.

  python scripts/independent_check.py results/eval/main.jsonl [more.jsonl ...]
Prints agreement and every disagreement; writes <file>.indep.json.
"""

import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


class Bad(Exception):
    pass


def extract(text: str):
    end = text.rfind("</answer>")
    if end < 0:
        return None
    start = text.rfind("<answer>", 0, end)
    if start < 0:
        return None
    return text[start + len("<answer>"):end]


def tokenize(s: str):
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-").replace("·", "*")
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c in " \t\n":
            i += 1
        elif c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            out.append(("num", int(s[i:j])))
            i = j
        elif c in "+-*/()=":
            out.append(("op", c))
            i += 1
        else:
            raise Bad(f"bad character {c!r}")
    return out


class Parser:
    def __init__(self, toks):
        self.t, self.i, self.used = toks, 0, []

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self):
        v = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            w = self.term()
            v = v + w if op == "+" else v - w
        return v

    def term(self):
        v = self.atom()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            w = self.atom()
            if op == "/" and w == 0:
                raise Bad("division by zero")
            v = v * w if op == "*" else v / w
        return v

    def atom(self):
        tok = self.take()
        if tok is None:
            raise Bad("unexpected end")
        if tok[0] == "num":
            self.used.append(tok[1])
            return Fraction(tok[1])
        if tok == ("op", "("):
            v = self.expr()
            if self.take() != ("op", ")"):
                raise Bad("missing )")
            return v
        raise Bad(f"unexpected {tok}")       # includes unary minus/plus


def check(text: str, numbers, target) -> float:
    body = extract(text)
    if body is None:
        return 0.0
    try:
        toks = tokenize(body)
        if sum(1 for t in toks if t == ("op", "=")) > 1:
            return 0.0
        if ("op", "=") in toks:
            k = toks.index(("op", "="))
            left, right = toks[:k], toks[k + 1:]
            if len(left) == 1 and left[0][0] == "num" and not (len(right) == 1 and right[0][0] == "num"):
                left, right = right, left           # "<target> = <expression>"
            if not (len(right) == 1 and right[0][0] == "num" and right[0][1] == target):
                return 0.0
            toks = left
        if not toks or "**" in body or "//" in body:
            return 0.0
        p = Parser(toks)
        v = p.expr()
        if p.i != len(toks):
            return 0.0
    except Bad:
        return 0.0
    if Counter(p.used) != Counter(numbers):
        return 0.0
    return 1.0 if v == target else 0.0


def main():
    for f in map(Path, sys.argv[1:]):
        rows = [json.loads(line) for line in open(f)]
        dis = []
        for r in rows:
            mine = check(r["completion"], r["numbers"], r["target"])
            if mine != r["reward"]:
                dis.append({"id": r["id"], "main": r["reward"], "independent": mine,
                            "expr": r.get("expr"), "error": r.get("error"), "tail": r["completion"][-160:]})
        res = {"file": str(f), "n": len(rows), "agree": 1 - len(dis) / len(rows), "disagreements": dis}
        f.with_suffix(".indep.json").write_text(json.dumps(res, indent=1))
        print(f"{f.name}: agreement {res['agree']:.2%} ({len(dis)} of {len(rows)} differ)")
        for d in dis[:5]:
            print("   ", d)


if __name__ == "__main__":
    main()
