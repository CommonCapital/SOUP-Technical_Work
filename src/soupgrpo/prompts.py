"""Prompt text and Qwen3 chat-template rendering.

Thinking mode must be OFF (enable_thinking=False). With it off, the Qwen3
template appends an EMPTY "<think>\\n\\n</think>\\n\\n" block to the assistant
prefix; that is expected. What must never happen is the model producing its
own <think> content in the completion (silent-failure row "thinking mode left
on, or wrong chat template").
"""

from __future__ import annotations

from functools import lru_cache

MODEL_ID = "Qwen/Qwen3-4B"

# v1: first version. Local smoke test on Qwen3-0.6B (E09) showed the model
#     copying the numeric example "(1 + 2) * 3" and answering in ~20 tokens
#     without working anything out.
# v2: no numeric example (nothing to copy); asks for short visible work first;
#     restates the rules the 0.6B model broke (only the given numbers, "* 1" is
#     an extra number).
PROMPTS = {
    "v1": (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You may use +, -, * and / and parentheses, and each number must be used exactly once. "
        "Think step by step briefly, then give only the final equation inside <answer></answer> tags, "
        "for example <answer>(1 + 2) * 3</answer>."
    ),
    "v2": (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "Use each of these numbers exactly once and no other numbers. "
        "You may use +, -, * and / and parentheses. "
        "First work it out briefly: try a few combinations and check the arithmetic. "
        "Then write only the final equation (without '= {target}') inside <answer></answer> tags "
        "at the end of your reply."
    ),
}
DEFAULT_PROMPT = "v2"
USER_TEMPLATE = PROMPTS[DEFAULT_PROMPT]

EMPTY_THINK = "<think>\n\n</think>\n\n"


def user_message(numbers: list[int], target: int, version: str = DEFAULT_PROMPT) -> str:
    return PROMPTS[version].format(numbers=", ".join(str(n) for n in numbers), target=target)


@lru_cache(maxsize=4)
def get_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    # Generation pads on the left (so all prompts end at the same position);
    # training sequences are padded on the right. Set explicitly by the caller
    # per phase; never rely on the default.
    return tok


def render_prompt(numbers: list[int], target: int, tokenizer=None, version: str = DEFAULT_PROMPT) -> str:
    tok = tokenizer or get_tokenizer()
    msgs = [{"role": "user", "content": user_message(numbers, target, version)}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return text


def check_rendered(text: str) -> None:
    """Assertions used by tests and before every run."""
    assert text.startswith("<|im_start|>user\n"), text[:50]
    assert text.endswith("<|im_start|>assistant\n" + EMPTY_THINK), repr(text[-80:])
    assert text.count("<think>") == 1, "exactly one (empty) think block expected"


def completion_has_thinking(completion: str) -> bool:
    return "<think>" in completion or "</think>" in completion
