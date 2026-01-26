# Enhanced combination functions for wenbi
import re
import math
from collections import Counter, defaultdict
import logging

def extract_mixed_keywords(text, verbose=False):
    """
    Extract mixed keywords: technical terms, proper nouns, and key concepts
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("Extracting mixed keywords from text")
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Remove common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'a', 'an'}
    
    # Filter stop words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Find proper nouns (capitalized words in original text)
    proper_nouns = set()
    for word in re.findall(r'\b[A-Z][a-zA-Z]+\b', text):
        proper_nouns.add(word.lower())
    
    # Find technical terms (words with numbers, underscores, or specific patterns)
    technical_terms = set()
    # Words with numbers, underscores, or mixed case
    technical_patterns = [r'\b\w*\d\w*\b', r'\b[a-zA-Z]+_[a-zA-Z_]+\b', r'\b[A-Z]{2,}\b']
    for pattern in technical_patterns:
        for match in re.findall(pattern, text):
            technical_terms.add(match.lower())
    
    # Get word frequencies
    word_freq = Counter(filtered_words)
    
    # Combine all keyword types
    keywords = set()
    keywords.update(proper_nouns)
    keywords.update(technical_terms)
    
    # Add frequent words (top 20% by frequency)
    if word_freq:
        freq_threshold = len(word_freq) // 5 or 1
        frequent_words = {word for word, freq in word_freq.most_common(freq_threshold)}
        keywords.update(frequent_words)
    
    if verbose:
        logger.debug(f"Extracted {len(keywords)} keywords: {list(keywords)[:10]}...")
    
    return list(keywords)


def calculate_ngram_similarity(text1, text2, n=2, verbose=False):
    """Calculate n-gram Jaccard similarity"""
    logger = logging.getLogger(__name__)
    
    # Generate n-grams
    def get_ngrams(text, n):
        words = text.lower().split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.append(ngram)
        return set(ngrams)
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    
    similarity = intersection / union if union > 0 else 0.0
    
    if verbose:
        logger.debug(f"N-gram similarity ({n}-gram): {similarity:.3f}")
    
    return similarity


def calculate_keyword_similarity(text1, text2, verbose=False):
    """Calculate keyword overlap similarity"""
    keywords1 = set(extract_mixed_keywords(text1, verbose))
    keywords2 = set(extract_mixed_keywords(text2, verbose))
    
    if not keywords1 and not keywords2:
        return 1.0
    if not keywords1 or not keywords2:
        return 0.0
    
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    similarity = intersection / union if union > 0 else 0.0
    
    if verbose:
        logger = logging.getLogger(__name__)
        logger.debug(f"Keyword similarity: {similarity:.3f}")
    
    return similarity


def calculate_tfidf_similarity(text1, text2, verbose=False):
    """Calculate TF-IDF cosine similarity"""
    logger = logging.getLogger(__name__)
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        if verbose:
            logger.debug(f"TF-IDF similarity: {similarity:.3f}")
        
        return similarity
        
    except ImportError:
        if verbose:
            logger.debug("sklearn not available, using simple word overlap")
        return calculate_keyword_similarity(text1, text2, verbose)
    except Exception as e:
        if verbose:
            logger.debug(f"TF-IDF calculation failed: {e}, using keyword similarity")
        return calculate_keyword_similarity(text1, text2, verbose)


def calculate_embedding_similarity(text1, text2, verbose=False):
    """Calculate word embedding similarity using pre-trained models"""
    logger = logging.getLogger(__name__)
    
    try:
        # Try to use spaCy for embeddings
        import spacy
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            if verbose:
                logger.debug("spaCy medium model not found, using simple similarity")
            return calculate_keyword_similarity(text1, text2, verbose)
        
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        
        similarity = doc1.similarity(doc2)
        
        if verbose:
            logger.debug(f"Embedding similarity: {similarity:.3f}")
        
        return similarity
        
    except ImportError:
        if verbose:
            logger.debug("spaCy not available, using keyword similarity")
        return calculate_keyword_similarity(text1, text2, verbose)
    except Exception as e:
        if verbose:
            logger.debug(f"Embedding calculation failed: {e}, falling back to keyword similarity")
        return calculate_keyword_similarity(text1, text2, verbose)


def calculate_combined_similarity(slide_text, speech_text, verbose=False):
    """
    Calculate combined similarity score using multiple methods
    Weights: TF-IDF (0.4), Keywords (0.3), N-gram (0.2), Embeddings (0.1)
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("Calculating combined similarity score")
    
    # Calculate individual similarities
    tfidf_sim = calculate_tfidf_similarity(slide_text, speech_text, verbose)
    keyword_sim = calculate_keyword_similarity(slide_text, speech_text, verbose)
    ngram_sim = calculate_ngram_similarity(slide_text, speech_text, 2, verbose)
    embedding_sim = calculate_embedding_similarity(slide_text, speech_text, verbose)
    
    # Combined weighted score
    combined_score = (0.4 * tfidf_sim + 0.3 * keyword_sim + 0.2 * ngram_sim + 0.1 * embedding_sim)
    
    if verbose:
        logger.debug(f"Combined similarity: {combined_score:.3f} (TF-IDF: {tfidf_sim:.3f}, Keywords: {keyword_sim:.3f}, N-gram: {ngram_sim:.3f}, Embeddings: {embedding_sim:.3f})")
    
    return combined_score


def create_similarity_matrix(slides, speech_paragraphs, verbose=False):
    """Create similarity matrix for all slide-speech pairs"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("Creating similarity matrix for all slide-speech pairs")
    
    matrix = {}
    for slide_idx, slide in enumerate(slides):
        for speech_idx, paragraph in enumerate(speech_paragraphs):
            similarity = calculate_combined_similarity(slide, paragraph, verbose)
            matrix[(slide_idx, speech_idx)] = similarity
    
    if verbose:
        logger.debug(f"Created similarity matrix with {len(matrix)} slide-speech pairs")
    
    return matrix


def is_temporally_consistent(slide_idx, speech_idx, aligned_slides, tolerance=2):
    """
    Check if alignment is temporally consistent with previous alignments
    Allows some flexibility with tolerance parameter
    """
    if not aligned_slides:
        return True
    
    # Get the most recent slide alignment
    max_slide_idx = max(aligned_slides.keys())
    corresponding_speech_indices = aligned_slides[max_slide_idx]
    
    # Handle both single speech_idx and list of speech_indices
    if isinstance(corresponding_speech_indices, list):
        if not corresponding_speech_indices:
            return True
        corresponding_speech_idx = max(corresponding_speech_indices)  # Use the latest one
    else:
        corresponding_speech_idx = corresponding_speech_indices
    
    # Check if current alignment maintains reasonable order
    # Allow some flexibility but prevent major temporal violations
    if slide_idx > max_slide_idx:
        # This slide comes after the latest aligned slide
        # Its speech should also come after, within tolerance
        return speech_idx >= corresponding_speech_idx - tolerance
    else:
        # This slide comes before or between existing alignments
        # Check against the nearest slide alignment
        for aligned_slide_idx, aligned_speech_idx_list in sorted(aligned_slides.items()):
            if isinstance(aligned_speech_idx_list, list):
                if not aligned_speech_idx_list:
                    continue
                aligned_speech_idx = max(aligned_speech_idx_list)
            else:
                aligned_speech_idx = aligned_speech_idx_list
                
            if abs(aligned_slide_idx - slide_idx) <= tolerance:
                return abs(speech_idx - aligned_speech_idx) <= tolerance * 2
        
        return True
    
    # Get the most recent slide alignment
    max_slide_idx = max(aligned_slides.keys())
    corresponding_speech_idx = aligned_slides[max_slide_idx]
    
    # Check if the current alignment maintains reasonable order
    # Allow some flexibility but prevent major temporal violations
    if slide_idx > max_slide_idx:
        # This slide comes after the latest aligned slide
        # Its speech should also come after, within tolerance
        return speech_idx >= corresponding_speech_idx - tolerance
    else:
        # This slide comes before or between existing alignments
        # Check against the nearest slide alignment
        for aligned_slide_idx, aligned_speech_idx in sorted(aligned_slides.items()):
            if abs(aligned_slide_idx - slide_idx) <= tolerance:
                return abs(speech_idx - aligned_speech_idx) <= tolerance * 2
        
        return True


def find_optimal_alignment(similarity_matrix, num_slides, num_paragraphs, verbose=False):
    """
    Find optimal alignment allowing multiple matches with hybrid temporal constraints
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("Finding optimal alignment with multiple matches allowed")
    
    # Sort all possible matches by similarity score (descending)
    sorted_matches = sorted(similarity_matrix.items(), key=lambda x: x[1], reverse=True)
    
    # Threshold for considering a match
    SIMILARITY_THRESHOLD = 0.25
    
    # Track alignments (allow multiple slides per speech paragraph)
    aligned_slides = {}  # slide_idx -> list of speech_idx
    speech_usage_count = defaultdict(int)  # speech_idx -> number of slides aligned
    
    # Process matches in order of similarity
    for (slide_idx, speech_idx), score in sorted_matches:
        if score >= SIMILARITY_THRESHOLD:
            # Check temporal consistency with existing alignments
            if is_temporally_consistent(slide_idx, speech_idx, aligned_slides):
                
                # Initialize alignment list for this slide if not exists
                if slide_idx not in aligned_slides:
                    aligned_slides[slide_idx] = []
                
                # Allow multiple matches but prefer high-quality ones
                # Don't let any speech paragraph get overloaded (>3 slides)
                if speech_usage_count[speech_idx] < 3:
                    aligned_slides[slide_idx].append(speech_idx)
                    speech_usage_count[speech_idx] += 1
                    
                    if verbose:
                        logger.debug(f"Aligned slide {slide_idx} with speech paragraph {speech_idx} (similarity: {score:.3f})")
    
    if verbose:
        total_aligned_slides = len([slide for slide, speeches in aligned_slides.items() if speeches])
        logger.debug(f"Aligned {total_aligned_slides}/{num_slides} slides with speech content")
    
    return aligned_slides


def distribute_unaligned_slides(slides, speech_paragraphs, aligned_slides, verbose=False):
    """
    Distribute unaligned slides evenly throughout the content
    """
    logger = logging.getLogger(__name__)
    
    # Find unaligned slides
    unaligned_slides = [i for i in range(len(slides)) if i not in aligned_slides or not aligned_slides[i]]
    
    if not unaligned_slides:
        return aligned_slides
    
    if verbose:
        logger.debug(f"Distributing {len(unaligned_slides)} unaligned slides")
    
    # Create distribution points throughout the speech
    num_distribution_points = len(speech_paragraphs)
    distribution_interval = max(1, num_distribution_points // (len(unaligned_slides) + 1))
    
    for i, slide_idx in enumerate(unaligned_slides):
        # Find distribution point
        target_paragraph = min((i + 1) * distribution_interval, len(speech_paragraphs) - 1)
        
        if slide_idx not in aligned_slides:
            aligned_slides[slide_idx] = []
        
        aligned_slides[slide_idx].append(target_paragraph)
        
        if verbose:
            logger.debug(f"Distributed unaligned slide {slide_idx} to speech paragraph {target_paragraph}")
    
    return aligned_slides


def _extract_slides(slides_markdown, verbose=False):
    """Extract individual slides from markdown (by splitting on major headings)"""
    logger = logging.getLogger(__name__)

    # Split by ## or # headings (common slide markers)
    slides = []
    current_slide = []

    lines = slides_markdown.split("\n")
    for line in lines:
        # Check if this is a slide heading (## or #)
        if re.match(r"^#+\s", line) and current_slide:
            # Save current slide and start new one
            slides.append("\n".join(current_slide).strip())
            current_slide = [line]
        else:
            current_slide.append(line)

    # Don't forget last slide
    if current_slide:
        slides.append("\n".join(current_slide).strip())

    if verbose:
        logger.debug(f"Extracted {len(slides)} slides from markdown")

    return [s for s in slides if s.strip()]  # Filter empty slides


def build_enhanced_combined_markdown(speech_paragraphs, slides, aligned_slides, cite_timestamps=False, verbose=False):
    """
    Build combined markdown with enhanced alignment and content preservation
    """
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.debug("Building enhanced combined markdown")
    
    # Create mapping of speech indices to slides that should be inserted before them
    slides_before_speech = defaultdict(list)
    all_used_speech = set()
    
    # Organize slides by their target speech positions
    for slide_idx, speech_indices in aligned_slides.items():
        for speech_idx in speech_indices:
            slides_before_speech[speech_idx].append((slide_idx, slides[slide_idx]))
            all_used_speech.add(speech_idx)
    
    # Build combined content
    combined = []
    
    for speech_idx, paragraph in enumerate(speech_paragraphs):
        # Insert slides before this paragraph
        if speech_idx in slides_before_speech:
            for slide_idx, slide_content in sorted(slides_before_speech[speech_idx]):
                combined.append(f"<!-- Slide {slide_idx + 1} -->")
                combined.append(slide_content)
                combined.append("")  # Blank line separator
        
        # Add the speech paragraph
        combined.append(paragraph)
        combined.append("")  # Blank line separator
    
    # Add any unused speech paragraphs at the end
    unused_speech = [i for i in range(len(speech_paragraphs)) if i not in all_used_speech]
    if unused_speech:
        if verbose:
            logger.debug(f"Adding {len(unused_speech)} unused speech paragraphs at the end")
        
        combined.append("---\n\n## Additional Speech Content\n")
        for speech_idx in unused_speech:
            combined.append(speech_paragraphs[speech_idx])
            combined.append("")
    
    result = "\n".join(combined).strip()
    
    if verbose:
        logger.debug(f"Enhanced combined content built with {len(speech_paragraphs)} speech paragraphs")
        logger.debug(f"Final enhanced combined content length: {len(result)} characters")
    
    return result