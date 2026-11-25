"""Cheat tests: every known way to fool the checker must score 0, and correct
answers in common formats must score 1."""

import pytest

from soupgrpo.reward import countdown_reward

NUMS = [19, 36, 55, 7]
TARGET = 65  # 55 + 36 - 19 - 7 = 65


def R(text, nums=NUMS, target=TARGET):
    return countdown_reward(text, nums, target)


# ---------- correct answers must pass ----------

@pytest.mark.parametrize("ans", [
    "55 + 36 - 19 - 7",
    "(55 + 36) - (19 + 7)",
    "  55+36-19-7  ",
    "55 + 36 - 19 - 7 = 65",          # trailing "= target" is a common format
    "55 + 36 − 19 − 7",                 # unicode minus sign
    "(55 − 19) + (36 − 7)",
    "((55 + 36) - 19) - 7",
])
def test_correct_passes(ans):
    r = R(f"Some reasoning.\n<answer>{ans}</answer>")
    assert r.reward == 1.0, r
    assert r.tag_present and r.parsed and r.numbers_correct and r.value_correct


def test_fraction_intermediate_is_exact():
    # 7 / 2 * 4 = 14 exactly; must pass with Fraction arithmetic
    r = countdown_reward("<answer>7 / 2 * 4</answer>", [7, 2, 4], 14)
    assert r.reward == 1.0, r


def test_last_answer_tag_counts():
    r = R("<answer>1+1</answer> ... <answer>55 + 36 - 19 - 7</answer>")
    assert r.reward == 1.0 and r.n_answer_tags == 2
    r = R("<answer>55 + 36 - 19 - 7</answer> ... <answer>1+1</answer>")
    assert r.reward == 0.0, "spraying answers must not help: only the last counts"


# ---------- cheats must fail ----------

@pytest.mark.parametrize("ans,why", [
    ("65", "just the target"),
    ("65 = 65", "target equals target"),
    ("55 + 36 - 19 - 7 + 0", "extra number 0"),
    ("55 + 36 - 19 - 7 + 7 - 7", "extra numbers"),
    ("55 + 55 - 19 - 36 + 7 - 7", "reused number"),
    ("55 + 36 - 26", "missing numbers"),
    ("55 + 36 - 19 - 7.0", "float literal"),
    ("64.9999", "float near target"),
    ("55 + 36 - 19 - 7 = 66", "wrong right-hand side"),
    ("55 + 36 - 19 - 7 = 65 = 65", "two equals signs"),
    ("55 + 36 - 19 - 7 / 0", "division by zero"),
    ("-(19 - 55) + 36 - 7", "unary minus"),
    ("+55 + 36 - 19 - 7", "unary plus"),
    ("55 + 36 - 19 - 7**1", "power operator"),
    ("55 + 36 - 19 - 7//1", "floor division"),
    ("55 + 36 - 19 - 7 % 8", "modulo"),
    ("__import__('os').system('echo hi')", "code injection"),
    ("sum([55, 36]) - 19 - 7", "function call"),
    ("True + 55 + 36 - 19 - 7 - 1", "bool literal"),
    ("0x37 + 36 - 19 - 7", "hex literal"),
    ("55_0 + 36", "underscore literal"),
    ("1e2", "scientific notation"),
    ("(55 + 36 - 19 - 7", "unbalanced parens"),
    ("x = 65", "variable"),
    ("9" * 300, "too long"),
    ("99999999 * 99999999 * 99999999", "huge value"),
])
def test_cheats_fail(ans, why):
    r = R(f"<answer>{ans}</answer>")
    assert r.reward == 0.0, f"{why}: {r}"


def test_no_tag_fails():
    r = R("The answer is 55 + 36 - 19 - 7")
    assert r.reward == 0.0 and not r.tag_present


def test_unclosed_tag_fails():
    r = R("<answer>55 + 36 - 19 - 7")
    assert r.reward == 0.0 and not r.tag_present


def test_subchecks_separate_format_from_skill():
    r = R("<answer>55 + 36 - 19 - 8</answer>")  # right format, wrong numbers
    assert r.tag_present and r.parsed and not r.numbers_correct and r.reward == 0.0
    r = R("<answer>55 + 36 - 7 - 19 + 0</answer>")
    assert r.tag_present and r.parsed and not r.numbers_correct
    r = R("<answer>55 + 36 + 19 - 7</answer>")  # right numbers, wrong value
    assert r.parsed and r.numbers_correct and not r.value_correct


def test_duplicate_given_numbers():
    # Puzzles may contain the same number twice; it must be used twice.
    assert countdown_reward("<answer>5 + 5 + 3</answer>", [5, 5, 3], 13).reward == 1.0
    assert countdown_reward("<answer>5 + 3 + 5 - 0</answer>", [5, 5, 3], 13).reward == 0.0
    assert countdown_reward("<answer>10 + 3</answer>", [5, 5, 3], 13).reward == 0.0


def test_tag_named_in_prose_before_answer():
    # E11: the 4B model writes "inside the <answer> tags:" before the real block
    nums, target = [62, 47, 2], 17
    for text in ["Now, write only the final equation inside the <answer> tags:\n\n<answer>(62 - 47) + 2</answer>",
                 "Put it in <answer></answer> tags.\n<answer>(62 - 47) + 2</answer>",
                 "<ANSWER> tags: <answer>(62 - 47) + 2</answer>"]:
        r = countdown_reward(text, nums, target)
        assert r.reward == 1.0, (text, r)


def test_prose_tag_does_not_create_an_answer():
    r = countdown_reward("I will use <answer> tags later. (62 - 47) + 2", [62, 47, 2], 17)
    assert r.reward == 0.0 and not r.tag_present


def test_target_on_the_left():
    # E11: "<answer>59 = 39 + 23 - (62 - 59)</answer>"
    assert countdown_reward("<answer>59 = 39 + 23 - (62 - 59)</answer>", [39, 62, 23, 59], 59).reward == 1.0
    assert countdown_reward("<answer>58 = 39 + 23 - (62 - 59)</answer>", [39, 62, 23, 59], 59).reward == 0.0
    assert countdown_reward("<answer>59 = 59</answer>", [39, 62, 23, 59], 59).reward == 0.0
