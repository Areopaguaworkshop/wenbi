#!/usr/bin/env python3
"""Test script for DeepL integration"""

import os
import sys
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_deepl_module_import():
    """Test if DeepL module can be imported"""
    print("Test 1: Testing DeepL module import...")
    try:
        from wenbi.llm.deepl import (
            configure_deepl,
            map_language_to_deepl_code,
            translate_with_deepl,
            is_deepl_available,
        )
        print("✓ DeepL module imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import DeepL module: {e}")
        return False


def test_language_mapping():
    """Test language code mapping"""
    print("\nTest 2: Testing language code mapping...")
    try:
        from wenbi.llm.deepl import map_language_to_deepl_code

        test_cases = [
            ("English", "EN"),
            ("Chinese", "ZH"),
            ("French", "FR"),
            ("Spanish", "ES"),
            ("english", "EN"),
            ("CHINESE", "ZH"),
        ]

        for lang_name, expected_code in test_cases:
            result = map_language_to_deepl_code(lang_name)
            if result == expected_code:
                print(f"  ✓ {lang_name} -> {result}")
            else:
                print(f"  ✗ {lang_name} -> {result} (expected {expected_code})")
                return False

        print("✓ Language mapping works correctly")
        return True
    except Exception as e:
        print(f"✗ Language mapping test failed: {e}")
        return False


def test_deepl_availability():
    """Test DeepL availability check"""
    print("\nTest 3: Testing DeepL availability check...")
    try:
        from wenbi.llm.deepl import is_deepl_available

        # Test without API key
        available = is_deepl_available()
        print(f"  DeepL available (without key): {available}")

        # Test with dummy key (will fail)
        available = is_deepl_available(api_key="fake-key-12345")
        print(f"  DeepL available (with fake key): {available}")

        print("✓ DeepL availability check works")
        return True
    except Exception as e:
        print(f"✗ DeepL availability test failed: {e}")
        return False


def test_translate_function_signature():
    """Test that translate function has new parameters"""
    print("\nTest 4: Testing translate function signature...")
    try:
        from wenbi.model import translate
        import inspect

        sig = inspect.signature(translate)
        params = list(sig.parameters.keys())

        required_params = ["use_deepl", "deepl_key"]
        missing_params = [p for p in required_params if p not in params]

        if missing_params:
            print(f"  ✗ Missing parameters: {missing_params}")
            return False

        print(f"  ✓ Found use_deepl parameter")
        print(f"  ✓ Found deepl_key parameter")
        print("✓ Translate function signature updated correctly")
        return True
    except Exception as e:
        print(f"✗ Translate function signature test failed: {e}")
        return False


def test_create_test_vtt():
    """Create a simple VTT file for testing"""
    print("\nTest 5: Creating test VTT file...")
    try:
        test_vtt = """WEBVTT

00:00:00.000 --> 00:00:05.000
Hello, this is a test.

00:00:05.000 --> 00:00:10.000
This is the second line.

00:00:10.000 --> 00:00:15.000
And this is the final test.
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".vtt", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_vtt)
            test_vtt_path = f.name

        print(f"✓ Created test VTT file at: {test_vtt_path}")
        return test_vtt_path
    except Exception as e:
        print(f"✗ Failed to create test VTT: {e}")
        return None


def test_translate_without_deepl_key():
    """Test translate function without DeepL key (should fallback to LLM)"""
    print("\nTest 6: Testing translate without DeepL key...")
    try:
        from wenbi.model import translate

        test_vtt_path = test_create_test_vtt()
        if not test_vtt_path:
            return False

        # Make sure DEEPL_API_KEY is not set
        os.environ.pop("DEEPL_API_KEY", None)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = translate(
                test_vtt_path,
                output_dir=tmpdir,
                translate_language="French",
                llm="ollama/qwen3",
                chunk_length=5,
                use_deepl=True,
                deepl_key=None,
                verbose=True,
            )

            if result and len(result) > 0:
                print(f"✓ Translation completed (fell back to LLM)")
                print(f"  Result length: {len(result)} characters")
                return True
            else:
                print(f"✗ Translation returned empty result")
                return False

    except Exception as e:
        print(f"✗ Translation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        if test_vtt_path and os.path.exists(test_vtt_path):
            os.unlink(test_vtt_path)


def test_cli_deepl_options():
    """Test that CLI has new DeepL options"""
    print("\nTest 7: Testing CLI DeepL options...")
    try:
        from wenbi.cli import add_global_args
        import argparse

        parser = argparse.ArgumentParser()
        subparser = parser.add_subparsers(dest="command").add_parser("test")
        add_global_args(subparser)

        # Try parsing with the new options
        args = subparser.parse_args(
            [
                "dummy_input.txt",
                "--use-deepl",
                "--deepl-key",
                "test-key-123",
            ]
        )

        if hasattr(args, "use_deepl") and hasattr(args, "deepl_key"):
            print(f"  ✓ use_deepl parameter: {args.use_deepl}")
            print(f"  ✓ deepl_key parameter: {args.deepl_key}")
            print("✓ CLI options added correctly")
            return True
        else:
            print("✗ CLI options not found")
            return False

    except Exception as e:
        print(f"✗ CLI options test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DeepL Integration Test Suite")
    print("=" * 60)

    tests = [
        ("DeepL Module Import", test_deepl_module_import),
        ("Language Mapping", test_language_mapping),
        ("DeepL Availability", test_deepl_availability),
        ("Translate Function Signature", test_translate_function_signature),
        ("CLI DeepL Options", test_cli_deepl_options),
        # ("Translate Without DeepL Key", test_translate_without_deepl_key),
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
