import dspy

def get_ollama_lm(model_name="ollama/qwen3", base_url=None, max_tokens=130000, timeout=3600, temperature=0.1, **kwargs):
    """
    Configures and returns a dspy.Ollama instance for an Ollama model.

    Args:
        model_name (str): The name of the Ollama model to use.
        base_url (str, optional): The base URL for the Ollama API. Defaults to "http://localhost:11434".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 50000.
        timeout (int, optional): The timeout in seconds for the API request. Defaults to 3600.
        temperature (float, optional): The temperature for sampling. Defaults to 0.1.
        **kwargs: Additional keyword arguments to pass to the dspy.Ollama constructor.

    Returns:
        dspy.Ollama: An instance of the configured Ollama language model.
    """
    if not model_name:
        model_name = "ollama/qwen3"

    config = kwargs
    config.update({
        "base_url": base_url or "http://localhost:11434",
        "model": model_name,
        "max_tokens": max_tokens,
        "timeout_s": timeout,
        "temperature": temperature,
    })
    return dspy.Ollama(**config)
