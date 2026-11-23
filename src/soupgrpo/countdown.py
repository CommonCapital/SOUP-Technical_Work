"""Countdown puzzle generator.

A puzzle is a list of numbers plus a target. A valid answer is an expression
using + - * / and parentheses that uses every number exactly once and equals
the target exactly.

Every generated puzzle is solvable by construction: we build a random
expression tree over the numbers first, and the target is its value. The
reference solution is stored with the puzzle (silent-failure row
"unsolvable puzzles in the set").

Difficulty knob (no per-prompt filtering, see plan "Why 1,500 prompts"):
  n_numbers, number range, target range, allowed operators.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from fractions import Fraction


@dataclass(frozen=True)
class Level:
    name: str
    n_numbers: int
    num_min: int = 1
    num_max: int = 99
    target_min: int = 1
    target_max: int = 100
    ops: tuple[str, ...] = ("+", "-", "*", "/")


# Starting levels for the pilot. These are knobs, not results: the pilot
# measures base-model success per level and picks one in 20-60%.
LEVELS: dict[str, Level] = {
    "n3": Level("n3", 3),
    "n4": Level("n4", 4),
    "n5": Level("n5", 5),
    "n6": Level("n6", 6, target_min=100, target_max=1000),
    # easier fallback named in the plan's search loop: only + and -
    "n3_addsub": Level("n3_addsub", 3, num_max=50, target_max=100, ops=("+", "-")),
}


@dataclass
class Puzzle:
    id: str
    level: str
    numbers: list[int]
    target: int
    solution: str
    meta: dict = field(default_factory=dict)

    def key(self) -> str:
        """Identity used for overlap checks: order of numbers does not matter."""
        return puzzle_key(self.numbers, self.target)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def puzzle_key(numbers: list[int], target: int) -> str:
    s = ",".join(str(n) for n in sorted(numbers)) + "=" + str(target)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def stable_seed(*parts: object) -> int:
    """Seed derived from names, stable across Python runs (unlike hash())."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def _combine(rng: random.Random, items: list[tuple[Fraction, str]], ops: tuple[str, ...]):
    """Randomly reduce items to one (value, expr) using every item once.

    Intermediate values are kept as non-negative integers so the reference
    solution looks like a normal Countdown solution. The reward checker does
    NOT impose this; it accepts any exact expression.
    """
    items = list(items)
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        (a, ea), (b, eb) = items[i], items[j]
        candidates = []
        for op in ops:
            if op == "+":
                candidates.append((a + b, f"({ea} + {eb})"))
            elif op == "-":
                if a - b >= 0:
                    candidates.append((a - b, f"({ea} - {eb})"))
                if b - a >= 0:
                    candidates.append((b - a, f"({eb} - {ea})"))
            elif op == "*":
                candidates.append((a * b, f"({ea} * {eb})"))
            elif op == "/":
                if b != 0 and a % b == 0:
                    candidates.append((a / b, f"({ea} / {eb})"))
                if a != 0 and b % a == 0:
                    candidates.append((b / a, f"({eb} / {ea})"))
        if not candidates:
            return None
        new = rng.choice(candidates)
        items = [x for k, x in enumerate(items) if k not in (i, j)] + [new]
    return items[0]


def _strip_outer(expr: str) -> str:
    if expr.startswith("(") and expr.endswith(")"):
        depth = 0
        for k, ch in enumerate(expr):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and k < len(expr) - 1:
                return expr
        return expr[1:-1]
    return expr


def generate_one(rng: random.Random, level: Level, max_tries: int = 1000) -> tuple[list[int], int, str]:
    for _ in range(max_tries):
        nums = [rng.randint(level.num_min, level.num_max) for _ in range(level.n_numbers)]
        res = _combine(rng, [(Fraction(n), str(n)) for n in nums], level.ops)
        if res is None:
            continue
        value, expr = res
        if value.denominator != 1:
            continue
        t = int(value)
        if level.target_min <= t <= level.target_max:
            return nums, t, _strip_outer(expr)
    raise RuntimeError(f"could not generate a puzzle for {level}")


def generate_split(level: Level, n: int, split: str, exclude_keys: set[str] | None = None) -> list[Puzzle]:
    """Generate n unique puzzles. Seed comes from (split, level), so train and
    held-out never share a random stream. Keys in exclude_keys are skipped."""
    rng = random.Random(stable_seed("countdown", split, level.name))
    exclude = set(exclude_keys or ())
    out: list[Puzzle] = []
    seen: set[str] = set()
    while len(out) < n:
        nums, target, sol = generate_one(rng, level)
        k = puzzle_key(nums, target)
        if k in seen or k in exclude:
            continue
        seen.add(k)
        out.append(Puzzle(id=f"{split}-{level.name}-{len(out):05d}", level=level.name,
                          numbers=nums, target=target, solution=sol))
    return out


def write_jsonl(path, puzzles: list[Puzzle]) -> None:
    with open(path, "w") as f:
        for p in puzzles:
            f.write(p.to_json() + "\n")


def read_jsonl(path) -> list[Puzzle]:
    with open(path) as f:
        return [Puzzle(**json.loads(line)) for line in f if line.strip()]
