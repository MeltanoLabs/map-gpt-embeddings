import openai


def get_embeddings(
    text: str, api_key: str, model="text-embedding-ada-002"
) -> list[float]:
    """Gets embedding for a given text, using the OpenAPI Embeddings endpoint.

    https://platform.openai.com/docs/api-reference/embeddings/create

    Returns an object with the following shape:
    ```
    {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [  # <-- this is the embedding we want
                    0.0023064255,
                    -0.009327292,
                    .... (1536 floats total for ada-002)
                    -0.0028842222,
                ],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {
            "prompt_tokens": 8,
            "total_tokens": 8,
        }
    }
    ```

    Args:
        text: The text to create embeddings for.
        model: The model to use. Defaults to "text-embedding-ada-002".

    Returns:
        A list of floats representing the embedding. 1536 floats for ada-002.
    """
    text = text.replace("\n", " ")
    response_dict: dict = openai.Embedding.create(
        input=[text], model=model, api_key=api_key
    )
    return response_dict["data"][0]["embedding"]
