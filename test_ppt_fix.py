#!/usr/bin/env python3
"""
Test script to verify PPT content preservation fix
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from wenbi.model import combine_speech_and_slides

# Test speech content with various newline patterns
test_speech = """各位观众朋友们，大家好！欢迎来到"光从东方来"平台，很高兴大家能够参与本次讲座。

本平台以学术讲座形式介绍东方教会传统，近期进行了简单更新：我们的通识课程将通过YouTube视频形式呈现，而非实时讲座形式。

今天是第一期以视频形式呈现的教会历史通识课，属于我开设的教会历史课程系列之一。

我们已完成了教会历史第一季，涵盖主耶稣时代至使徒时期（约公元20世纪中期）。第二季将分为三个部分，分别介绍三个教会传统。

第一个传统为希腊传统（已于去年完成），第二个传统即拉丁传统（今年开讲主题）。目前已进行至第九、十课，今天是拉丁传统第十课，介绍人物为卡西安。

卡西安与上一讲的哲禄为同时代人物。现开始介绍卡西安生平。其生平具有以下特点：首先介绍其主要著作。

他留下两部重要著作：《要则》与《会谈录》。《要则》是写给所在修院修士的，相当于当时圣巴赦尔的长会规或短会规。

因圣巴赦尔的长会规在西方广为流传，西方修士希望拥有自己的会规，故《要则》实质为修士会规内容。《会谈录》（英文为conference）是本期讲座重点，我们将展示其教刊本与音译本。该著作将在后续详细介绍。"""

# Test slides content
test_slides = """# **卡西安与《会谈录》**

![](_page_1_Picture_0.jpeg)

# **校勘本译本**

Cassian, 《要则 The Institutes》(Ins.)

- Edited by Petschenig, Michael. Iohannis Cassiani De institutis coenobiorum et de octo principalium vitiorum remediis libri XII : De incarnatione Domini contra Nestorium libri VII. Vindobonae [Vienna, Austria]: F. Tempsky, 1888. •
- Translated by Ramsey, Boniface. John Cassian: The Institutes. New York: Newman Press, 2000. •

# **生平**

360年出生于Dobrudja,现罗马尼亚

380-385: Bethlehem,与Germanus一道在伯利恒成为修士

385-400:Scetis, Kellia, Nitria; 奥利金主义 (400) VS 上帝人形论"""

def test_content_preservation():
    print("Testing PPT content preservation...")
    
    # Get original speech word count
    original_words = len(test_speech.split())
    print(f"Original speech word count: {original_words}")
    
    # Test the combination
    try:
        combined = combine_speech_and_slides(
            test_speech,
            test_slides,
            llm="dummy",  # Use dummy to avoid LLM calls for this test
            verbose=True
        )
        
        combined_words = len(combined.split())
        print(f"Combined content word count: {combined_words}")
        
        preservation_ratio = combined_words / original_words if original_words > 0 else 0
        print(f"Content preservation ratio: {preservation_ratio:.2%}")
        
        if preservation_ratio >= 0.90:
            print("✅ SUCCESS: Content preservation is >= 90%")
        else:
            print("❌ FAILURE: Content preservation is < 90%")
            
        # Save test output
        with open("test_combined_output.md", "w", encoding="utf-8") as f:
            f.write(combined)
        print("Test output saved to: test_combined_output.md")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_content_preservation()