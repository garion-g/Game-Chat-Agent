import os
from typing import List, Dict, Any
from chromadb.api.types import EmbeddingFunction
from openai import OpenAI
class VocareumEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function for use with Vocareum's OpenAI proxy.
    """
    def __init__(self, model_name: str = "text-embedding-ada-002", **kwargs):
        self._client = OpenAI(
            api_key=os.environ.get("VOC_OPENAI_API_KEY"),
            base_url="https://openai.vocareum.com/v1"
        )
        self._model_name = model_name
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        Args:
            texts: A list of strings to embed.
        Returns:
            A list of embeddings, where each embedding is a list of floats.
        """
        if not texts:
            return []
        # OpenAI API can handle multiple texts in one call
        response = self._client.embeddings.create(
            model=self._model_name,
            input=texts
        )
        
        return [embedding.embedding for embedding in response.data]
# Example usage (for testing purposes)
if __name__ == '__main__':
    # This assumes you have set the VOC_OPENAI_API_KEY environment variable
    # In Vocareum, this is typically set for you.
    if "VOC_OPENAI_API_KEY" not in os.environ:
        print("Please set the VOC_OPENAI_API_KEY environment variable to test this script.")
    else:
        embedding_function = VocareumEmbeddingFunction()
        sample_texts = ["Hello, world!", "This is a test."]
        embeddings = embedding_function(sample_texts)
        print(f"Successfully generated {len(embeddings)} embeddings.")
        print(f"Dimension of first embedding: {len(embeddings[0])}")
