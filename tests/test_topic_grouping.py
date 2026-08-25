from wenbi.bilingual import _strip_topic_segment_markers, group_into_topics


def test_large_transcripts_are_chunked_without_an_llm_round_trip():
    segments = [{"text": f"segment-{number} " + "x" * 700} for number in range(100)]

    paragraphs = group_into_topics(segments)

    assert len(paragraphs) > 1
    assert " ".join(paragraphs) == " ".join(segment["text"] for segment in segments)


def test_topic_grouping_markers_are_not_emitted_in_output():
    assert _strip_topic_segment_markers("[307] First line\n[308] Second line") == "First line\nSecond line"