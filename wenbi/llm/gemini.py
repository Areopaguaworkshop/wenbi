import dspy
import os

def get_gemini_lm(model_name="gemini/gemini-2.5-flash", api_key=None, base_url=None, max_tokens=50000, timeout=3600, temperature=0.1, **kwargs):
    """
    Configures and returns a dspy.LM instance for a Gemini model.

    Args:
        model_name (str): The name of the Gemini model to use.
        api_key (str, optional): The Gemini API key. If not provided,
                                 it will be read from the GOOGLE_API_KEY
                                 environment variable.
        base_url (str, optional): The base URL for the Gemini API. If not provided,
                                  a default will be used.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 50000.
        timeout (int, optional): The timeout in seconds for the API request. Defaults to 3600.
        temperature (float, optional): The temperature for sampling. Defaults to 0.1.
        **kwargs: Additional keyword arguments to pass to the dspy.LM constructor.

    Returns:
        dspy.LM: An instance of the configured Gemini language model.
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")

    # The model string for dspy should not include the provider prefix
    model_id = model_name.replace("gemini/", "")

    # Set the base URL if not provided
    url = base_url or "https://generativelanguage.googleapis.com/v1beta"
    if not url.endswith('/'):
        url += '/'

    lm = dspy.LM(
        model=model_id,
        api_key=api_key,
        base_url=url,
        max_tokens=max_tokens,
        timeout_s=timeout,
        temperature=temperature,
        **kwargs
    )
    dspy.configure(lm=lm)
    return lm

if __name__ == '__main__':
    # Example usage:
    try:
        # Make sure to set the GOOGLE_API_KEY environment variable before running
        gemini_lm = get_gemini_lm()
        print("Successfully configured Gemini LM:")
        print(f"Model: {gemini_lm.model}")
        print(f"Base URL: {gemini_lm.base_url}")
    except ValueError as e:
        print(e)
