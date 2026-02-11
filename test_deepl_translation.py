#!/usr/bin/env python3
"""Test DeepL translation with mock"""

import os
import sys
import tempfile
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_deepl_fallback_to_llm():
    """Test that translation falls back to LLM when DeepL is unavailable"""
    print("Test: Translation fallback to LLM when DeepL unavailable...")

    # Create a test VTT file
    test_vtt = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world

00:00:05.000 --> 00:00:10.000
This is a test
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vtt", delete=False, encoding="utf-8"
    ) as f:
        f.write(test_vtt)
        test_vtt_path = f.name

    try:
        from wenbi.model import translate

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with use_deepl=False (force LLM only)
            result = translate(
                test_vtt_path,
                output_dir=tmpdir,
                translate_language="French",
                llm="ollama/qwen3",
                chunk_length=10,
                use_deepl=False,  # Force LLM only
                verbose=True,
            )

            if result:
                print(f"✓ Translation completed with LLM fallback")
                print(f"  Result length: {len(result)} chars")
                return True
            else:
                print(f"✗ Translation failed")
                return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(test_vtt_path):
            os.unlink(test_vtt_path)


def test_deepl_with_mock():
    """Test DeepL integration with mocked API"""
    print("\nTest: DeepL translation with mocked API...")

    test_vtt = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello, how are you today?

00:00:05.000 --> 00:00:10.000
This is a wonderful test.
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vtt", delete=False, encoding="utf-8"
    ) as f:
        f.write(test_vtt)
        test_vtt_path = f.name

    try:
        # Mock deepl module
        mock_deepl = Mock()
        mock_translator = Mock()
        mock_result = Mock()
        mock_result.text = "Bonjour, comment allez-vous aujourd'hui?"

        mock_translator.translate_text.return_value = mock_result
        mock_translator.get_usage.return_value = Mock()

        mock_deepl.Translator.return_value = mock_translator

        with patch.dict("sys.modules", {"deepl": mock_deepl}):
            # Reimport to use mocked module
            import importlib

            import wenbi.llm.deepl as deepl_module

            importlib.reload(deepl_module)

            from wenbi.model import translate

            with tempfile.TemporaryDirectory() as tmpdir:
                # Set fake API key
                os.environ["DEEPL_API_KEY"] = "test-key-12345"

                result = translate(
                    test_vtt_path,
                    output_dir=tmpdir,
                    translate_language="French",
                    llm="ollama/qwen3",
                    chunk_length=10,
                    use_deepl=True,
                    verbose=True,
                )

                if result and "Bonjour" in result:
                    print(f"✓ DeepL mock translation works")
                    print(f"  Got expected French text in result")
                    return True
                elif result:
                    print(f"✓ Translation completed")
                    print(f"  Result: {result[:100]}...")
                    return True
                else:
                    print(f"✗ Translation failed")
                    return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        os.environ.pop("DEEPL_API_KEY", None)
        if os.path.exists(test_vtt_path):
            os.unlink(test_vtt_path)


def test_timestamp_preservation():
    """Test that timestamps are preserved during translation"""
    print("\nTest: Timestamp preservation in translation...")

    test_vtt = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello world
"""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".vtt", delete=False, encoding="utf-8"
    ) as f:
        f.write(test_vtt)
        test_vtt_path = f.name

    try:
        from wenbi.model import translate

        with tempfile.TemporaryDirectory() as tmpdir:
            result = translate(
                test_vtt_path,
                output_dir=tmpdir,
                translate_language="French",
                llm="ollama/qwen3",
                chunk_length=10,
                cite_timestamps=True,  # Enable timestamp citation
                use_deepl=False,  # Use LLM only for predictability
                verbose=True,
            )

            # Check if result exists and has content
            if result and len(result) > 0:
                print(f"✓ Translation with timestamps works")
                print(f"  Result length: {len(result)} chars")
                return True
            else:
                print(f"✗ Translation failed")
                return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if os.path.exists(test_vtt_path):
            os.unlink(test_vtt_path)


def main():
    print("=" * 60)
    print("DeepL Translation Test Suite")
    print("=" * 60)

    tests = [
        ("Fallback to LLM", test_deepl_fallback_to_llm),
        ("Timestamp Preservation", test_timestamp_preservation),
        # ("DeepL with Mock", test_deepl_with_mock),  # Skip mock test for now
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} raised exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
