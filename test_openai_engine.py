"""Focused checks for the selectable OpenAI translation engine."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from wenbi.bilingual import process_speaker
from wenbi.llm.openai import DEFAULT_MODEL, get_openai_lm
from wenbi.model import resolve_translation_engine


class TestOpenAIEngine(unittest.TestCase):
    def test_openai_engine_uses_gpt_and_skips_deepl(self):
        self.assertEqual(
            resolve_translation_engine("openai"),
            (DEFAULT_MODEL, False),
        )

    def test_ollama_engine_skips_deepl(self):
        self.assertEqual(
            resolve_translation_engine("ollama"),
            ("ollama/glm-5.2:cloud", False),
        )

    def test_speaker_pipeline_forwards_openai_without_deepl(self):
        with TemporaryDirectory() as output_dir:
            input_path = Path(output_dir, "lecture.vtt")
            input_path.write_text(
                "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nSt. Maximos teaches.\n",
                encoding="utf-8",
            )
            with (
                patch("wenbi.bilingual.group_into_topics", return_value=["St. Maximos teaches."]),
                patch("wenbi.bilingual.rewrite_english", return_value=["St. Maximos teaches."]),
                patch("wenbi.bilingual.translate_chunks", return_value=["圣马克西姆教导。"]) as translate,
            ):
                process_speaker(
                    str(input_path),
                    output_dir=output_dir,
                    llm=DEFAULT_MODEL,
                    use_deepl=False,
                )
            self.assertFalse(translate.call_args.kwargs["use_deepl"])

    @patch("wenbi.llm.openai.dspy.LM")
    def test_openai_lm_reads_explicit_key(self, lm):
        get_openai_lm(api_key="test-key", max_tokens=123, timeout=45)
        self.assertEqual(lm.call_args.kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(lm.call_args.kwargs["api_key"], "test-key")
        self.assertEqual(lm.call_args.kwargs["max_tokens"], 123)
        self.assertEqual(lm.call_args.kwargs["timeout"], 45)
        self.assertEqual(lm.call_args.kwargs["temperature"], 1.0)


if __name__ == "__main__":
    unittest.main()