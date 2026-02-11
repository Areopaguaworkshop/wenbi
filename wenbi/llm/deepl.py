import logging
import os

try:
    import deepl
except ImportError:
    deepl = None


def configure_deepl(api_key=None, verbose=False):
    """Configure and validate DeepL client"""
    logger = logging.getLogger(__name__)

    if deepl is None:
        raise ImportError(
            "deepl-python library not installed. Install with: pip install deepl"
        )

    if not api_key:
        api_key = os.getenv("DEEPL_API_KEY")

    if not api_key:
        raise ValueError(
            "DEEPL_API_KEY not provided. Set environment variable or pass --deepl-key"
        )

    if verbose:
        logger.debug(f"Configuring DeepL with API key (length: {len(api_key)})")

    try:
        translator = deepl.Translator(api_key)
        # Test the connection
        translator.get_usage()
        if verbose:
            logger.debug("DeepL API connection successful")
        return translator
    except Exception as e:
        error_msg = f"Failed to configure DeepL: {e}"
        if verbose:
            logger.debug(error_msg)
        raise Exception(error_msg)


def map_language_to_deepl_code(lang_name, verbose=False):
    """Map language name to DeepL language code"""
    logger = logging.getLogger(__name__)

    language_map = {
        "english": "EN",
        "chinese": "ZH",
        "spanish": "ES",
        "french": "FR",
        "german": "DE",
        "italian": "IT",
        "portuguese": "PT",
        "russian": "RU",
        "japanese": "JA",
        "korean": "KO",
        "turkish": "TR",
        "polish": "PL",
        "dutch": "NL",
        "swedish": "SV",
        "norwegian": "NO",
        "danish": "DA",
        "finnish": "FI",
        "czech": "CS",
        "greek": "EL",
        "hungarian": "HU",
        "romanian": "RO",
        "slovak": "SK",
        "slovenian": "SL",
        "bulgarian": "BG",
        "estonian": "ET",
        "latvian": "LV",
        "lithuanian": "LT",
        "ukrainian": "UK",
        "arabic": "AR",
        "hebrew": "HE",
        "persian": "FA",
        "thai": "TH",
        "vietnamese": "VI",
        "indonesian": "ID",
        "malay": "MS",
        "burmese": "MY",
        "tagalog": "TL",
    }

    lang_lower = lang_name.lower().strip()

    # Handle regional variants like "EN-US" or "english-us"
    if "-" in lang_lower:
        base_lang = lang_lower.split("-")[0]
    else:
        base_lang = lang_lower

    deepl_code = language_map.get(base_lang)

    if not deepl_code:
        error_msg = f"Language '{lang_name}' not supported by DeepL"
        if verbose:
            logger.debug(error_msg)
        raise ValueError(error_msg)

    if verbose:
        logger.debug(f"Mapped '{lang_name}' to DeepL code '{deepl_code}'")

    return deepl_code


def translate_with_deepl(
    translator, chunk, target_language, verbose=False, max_retries=2
):
    """Translate a single chunk using DeepL"""
    logger = logging.getLogger(__name__)

    if not chunk or not chunk.strip():
        return chunk

    deepl_code = map_language_to_deepl_code(target_language, verbose=verbose)

    if verbose:
        logger.debug(
            f"Translating chunk ({len(chunk)} chars) to {target_language} ({deepl_code})"
        )

    for attempt in range(max_retries):
        try:
            result = translator.translate_text(chunk, target_lang=deepl_code)
            translated = result.text

            if verbose:
                logger.debug(
                    f"DeepL translation successful (attempt {attempt + 1}/{max_retries})"
                )

            return translated

        except deepl.DocumentTranslationException as e:
            error_msg = f"DeepL document translation error: {e}"
            if verbose:
                logger.debug(error_msg)
            raise Exception(error_msg)

        except deepl.QuotaExceededException as e:
            error_msg = f"DeepL quota exceeded: {e}"
            if verbose:
                logger.debug(error_msg)
            raise Exception(error_msg)

        except deepl.TooManyRequestsException as e:
            error_msg = f"DeepL rate limited (attempt {attempt + 1}/{max_retries}): {e}"
            if verbose:
                logger.debug(error_msg)
            if attempt < max_retries - 1:
                import time

                wait_time = 2**attempt  # Exponential backoff
                if verbose:
                    logger.debug(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                raise Exception(error_msg)

        except Exception as e:
            error_msg = f"DeepL translation error: {e}"
            if verbose:
                logger.debug(error_msg)
            raise Exception(error_msg)

    raise Exception("DeepL translation failed after all retries")


def is_deepl_available(api_key=None, verbose=False):
    """Check if DeepL is available and properly configured"""
    logger = logging.getLogger(__name__)

    if deepl is None:
        if verbose:
            logger.debug("deepl-python not installed")
        return False

    if not api_key:
        api_key = os.getenv("DEEPL_API_KEY")

    if not api_key:
        if verbose:
            logger.debug("DEEPL_API_KEY not set")
        return False

    try:
        translator = deepl.Translator(api_key)
        translator.get_usage()
        if verbose:
            logger.debug("DeepL is available and functional")
        return True
    except Exception as e:
        if verbose:
            logger.debug(f"DeepL check failed: {e}")
        return False
