import pytest

from soupgrpo.prompts import check_rendered, completion_has_thinking, render_prompt, user_message


@pytest.fixture(scope="module")
def tok():
    pytest.importorskip("transformers")
    from soupgrpo.prompts import get_tokenizer
    try:
        return get_tokenizer()
    except Exception as e:  # offline and not cached
        pytest.skip(f"tokenizer unavailable: {e}")


def test_render_thinking_off(tok):
    t = render_prompt([19, 36, 55, 7], 65, tok)
    check_rendered(t)
    assert "19, 36, 55, 7" in t and "equals 65" in t


def test_special_tokens(tok):
    # Generation stops on <|im_end|>; padding uses <|endoftext|>, never EOS.
    assert tok.eos_token == "<|im_end|>"
    assert tok.pad_token == "<|endoftext|>"
    assert tok.pad_token_id != tok.eos_token_id


def test_prompt_is_short(tok):
    n = len(tok(render_prompt([88, 13, 85, 92, 7], 68, tok)).input_ids)
    assert n < 128, n


def test_thinking_detector():
    assert completion_has_thinking("<think>hmm</think> <answer>1</answer>")
    assert not completion_has_thinking("Step 1 ... <answer>1+2</answer>")


def test_user_message_lists_all_numbers():
    m = user_message([5, 5, 3], 13)
    assert "5, 5, 3" in m and "13" in m
