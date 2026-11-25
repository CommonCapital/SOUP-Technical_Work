"""Countdown reward: 1.0 if the answer is a valid equation that uses every
given number exactly once and equals the target, else 0.0.

Safety: the model's text is parsed with `ast` and evaluated by walking a
whitelist of node types with exact Fractions. `eval` is never called.

Four sub-checks are returned separately so format learning can be told apart
from skill learning:
  tag_present     an <answer>...</answer> block exists
  parsed          its content parses as an allowed arithmetic expression
  numbers_correct the integers used are exactly the given multiset
  value_correct   the expression's exact value equals the target
reward = 1.0 only if all four hold.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction

# A block's content may not contain another "<answer>": prose such as
# "write it inside the <answer> tags: <answer>1+2</answer>" must yield "1+2"
# (found in the first Qwen3-4B pilot, E11; the old pattern read "tags: <answer>1+2").
ANSWER_RE = re.compile(r"<answer>((?:(?!<answer>).)*?)</answer>", re.DOTALL | re.IGNORECASE)
MAX_EXPR_CHARS = 200          # guards the parser against huge inputs
MAX_ABS_VALUE = 10**12        # guards against exponent-like blowups via *

_UNICODE_OPS = {"×": "*", "·": "*", "÷": "/", "−": "-", "–": "-", "—": "-"}


@dataclass
class RewardResult:
    reward: float
    tag_present: bool = False
    parsed: bool = False
    numbers_correct: bool = False
    value_correct: bool = False
    n_answer_tags: int = 0
    expr: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class _Reject(Exception):
    pass


def extract_answer(text: str) -> tuple[str | None, int]:
    """Return the content of the LAST <answer> block and how many there were.
    Using the last one means writing many answers does not help the model."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None, 0
    return matches[-1], len(matches)


def normalize_expr(s: str) -> str:
    s = s.strip()
    for k, v in _UNICODE_OPS.items():
        s = s.replace(k, v)
    return s


def _split_equals(s: str) -> tuple[str, str | None]:
    """Accept "= <number>" on either side (e.g. "19 + 36 + 10 = 65"): a common,
    correct format we must not reject (silent-failure row "checker rejects a
    correct answer in another format"). Only the left side is evaluated; the
    right side must be a plain integer equal to the target."""
    if s.count("=") == 1:
        left, right = (x.strip() for x in s.split("="))
        # "<target> = <expression>" is the same claim written the other way round
        # (seen in the 4B pilot, E11). Swap so the expression is evaluated; the
        # plain-integer side must still equal the target.
        if re.fullmatch(r"\d+", left) and not re.fullmatch(r"\d+", right):
            left, right = right, left
        return left, right
    if "=" in s:
        raise _Reject("more than one '='")
    return s, None


def _eval(node: ast.AST, used: list[int]) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval(node.body, used)
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, used)
        right = _eval(node.right, used)
        if isinstance(node.op, ast.Add):
            v = left + right
        elif isinstance(node.op, ast.Sub):
            v = left - right
        elif isinstance(node.op, ast.Mult):
            v = left * right
        elif isinstance(node.op, ast.Div):
            if right == 0:
                raise _Reject("division by zero")
            v = left / right
        else:
            raise _Reject(f"operator not allowed: {type(node.op).__name__}")
        if abs(v) > MAX_ABS_VALUE:
            raise _Reject("value too large")
        return v
    if isinstance(node, ast.Constant):
        # bool is a subclass of int; reject it. Floats are rejected: the given
        # numbers are integers, and floats like 64.9999 must never pass.
        if type(node.value) is not int:
            raise _Reject(f"literal not an integer: {node.value!r}")
        used.append(node.value)
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp):
        # Unary minus/plus would let the model negate without using a number.
        raise _Reject("unary operator not allowed")
    raise _Reject(f"syntax not allowed: {type(node).__name__}")


def parse_and_eval(expr: str) -> tuple[Fraction, list[int]]:
    if len(expr) > MAX_EXPR_CHARS:
        raise _Reject("expression too long")
    if not re.fullmatch(r"[0-9+\-*/() \t\n]*", expr):
        raise _Reject("characters outside 0-9 + - * / ( )")
    if "**" in expr or "//" in expr:
        raise _Reject("** and // not allowed")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise _Reject(f"syntax error: {e.msg}") from None
    used: list[int] = []
    value = _eval(tree, used)
    return value, used


def countdown_reward(text: str, numbers: list[int], target: int) -> RewardResult:
    raw, n_tags = extract_answer(text)
    r = RewardResult(reward=0.0, n_answer_tags=n_tags)
    if raw is None:
        r.error = "no <answer> tag"
        return r
    r.tag_present = True
    try:
        expr, rhs = _split_equals(normalize_expr(raw))
        r.expr = expr
        if rhs is not None and not re.fullmatch(r"\d+", rhs):
            raise _Reject("right side of '=' is not an integer")
        value, used = parse_and_eval(expr)
    except _Reject as e:
        r.error = str(e)
        return r
    r.parsed = True
    r.numbers_correct = Counter(used) == Counter(numbers)
    r.value_correct = value == target and (rhs is None or int(rhs) == target)
    if not r.numbers_correct:
        r.error = f"numbers used {sorted(used)} != given {sorted(numbers)}"
    elif not r.value_correct:
        r.error = f"value {value} != target {target}"
    r.reward = 1.0 if (r.numbers_correct and r.value_correct) else 0.0
    return r
