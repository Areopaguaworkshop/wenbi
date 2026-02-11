#!/usr/bin/env python3
"""Test the keep_original_lang feature for translate subcommand"""

import os
import tempfile
from wenbi.model import translate

def test_keep_original_lang():
    """Test that keep_original_lang option works correctly"""
    
    # Create a temporary test file with English text
    test_content = """This is the first paragraph.
It has multiple sentences.
This tests the translation feature.

This is the second paragraph.
We want to keep the original text.
And see it displayed above the translation."""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as output_dir:
            print("Testing translate with keep_original_lang=True...")
            print("=" * 60)
            
            # Test with keep_original_lang=True
            result = translate(
                test_file,
                output_dir=output_dir,
                translate_language="Chinese",
                chunk_length=5,
                keep_original_lang=True,
                verbose=True,
            )
            
            print("\n" + "=" * 60)
            print("RESULT WITH keep_original_lang=True:")
            print("=" * 60)
            print(result)
            
            # Verify the output contains both original and translation markers
            assert "**[Original]**" in result, "Missing [Original] marker"
            assert "**[Chinese]**" in result, "Missing [Chinese] marker"
            print("\n✓ Successfully kept original language with translation markers")
            
            # Test with keep_original_lang=False (default)
            print("\n" + "=" * 60)
            print("Testing translate with keep_original_lang=False...")
            print("=" * 60)
            
            result2 = translate(
                test_file,
                output_dir=output_dir,
                translate_language="Spanish",
                chunk_length=5,
                keep_original_lang=False,
                verbose=False,
            )
            
            print("\nRESULT WITH keep_original_lang=False:")
            print("=" * 60)
            print(result2)
            
            # Verify that translation-only output doesn't have markers
            assert "**[Original]**" not in result2, "Should not have [Original] marker when keep_original_lang=False"
            assert "**[Spanish]**" not in result2, "Should not have [Spanish] marker when keep_original_lang=False"
            print("\n✓ Successfully translated without original language markers")
            
            print("\n" + "=" * 60)
            print("All tests passed!")
            print("=" * 60)
    
    finally:
        # Clean up
        os.unlink(test_file)

if __name__ == "__main__":
    test_keep_original_lang()
