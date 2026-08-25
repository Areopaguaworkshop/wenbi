from types import SimpleNamespace

from wenbi import model
from wenbi.bilingual import rewrite_english


def test_rewrite_english_uses_the_edited_transcript_signature(monkeypatch):
    captured = {}

    class FakeSignature:
        pass

    class FakePredict:
        def __init__(self, signature):
            captured["instructions"] = signature.__doc__

        def __call__(self, **kwargs):
            captured["input"] = kwargs
            return SimpleNamespace(written_transcript="I think the point is clear.")

    fake_dspy = SimpleNamespace(
        Signature=FakeSignature,
        InputField=lambda **kwargs: kwargs,
        OutputField=lambda **kwargs: kwargs,
        Predict=FakePredict,
    )
    monkeypatch.setattr(model, "configure_lm", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "_import_dspy", lambda: fake_dspy)

    rewritten = rewrite_english(["Um, I think the point is clear."], llm="test/model")

    assert rewritten == ["I think the point is clear."]
    assert captured["input"] == {
        "spoken_english_transcript": "Um, I think the point is clear."
    }
    assert "Do not summarize" in captured["instructions"]
    assert "internal segment IDs" in captured["instructions"]
    assert "Keep hedges" in captured["instructions"]
    assert "discourse-only openers" in captured["instructions"]
