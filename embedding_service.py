import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
import time

# Initialize logger for the module
logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    A service for generating and managing text embeddings using OpenAI's models.
    It includes caching mechanisms to reduce API calls and improve performance.
    """
    
    def __init__(self, model: str = "text-embedding-3-large", persistence_manager=None):
        self.model = model
        self.persistence_manager = persistence_manager
        self.embedding_cache = {}  # Temporary cache for current session only
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    timeout=30.0
                )
                embedding = np.array(response.data[0].embedding)
                
                # Cache the result
                self.embedding_cache[text] = embedding
                return embedding
                
            except Exception as e:
                logger.error(f"Failed to generate embedding for '{text[:50]}...' (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    # Return None to indicate failure - caller should handle this
                    raise Exception(f"Failed to generate embedding after {max_retries} attempts: {e}")

    def embed_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        Generates embeddings for a list of texts efficiently by batching API calls.
        Prioritizes cache lookups before making API requests.

        Args:
            texts (List[str]): A list of text strings to embed.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping each text to its embedding.
        """
        embeddings = {}
        texts_to_embed = []
        
        # First, check the cache for each text
        for text in texts:
            if text in self.embedding_cache:
                embeddings[text] = self.embedding_cache[text]
            else:
                texts_to_embed.append(text) # Add to list for API embedding if not in cache
        
        # If there are texts not found in cache, embed them in batches
        if texts_to_embed:
            try:
                # OpenAI API has a limit for input texts per request (e.g., 2048 for some models)
                batch_size = 100 
                for i in range(0, len(texts_to_embed), batch_size):
                    batch = texts_to_embed[i:i + batch_size]
                    
                    # Call OpenAI API for the current batch
                    response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                    )

                    for j, text in enumerate(batch):
                        embedding = np.array(response.data[j].embedding)
                        embeddings[text] = embedding
                        self.embedding_cache[text] = embedding
                
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
        
        return embeddings
        
    def compute_similarity(self, embedding1: Union[np.ndarray, List[float]], 
                      embedding2: Union[np.ndarray, List[float]]) -> float:
        """Computes the cosine similarity between two embedding vectors."""
        # Convert lists to numpy arrays if needed
        if isinstance(embedding1, list):
            embedding1 = np.array(embedding1)
        if isinstance(embedding2, list):
            embedding2 = np.array(embedding2)
        
        # cosine_similarity expects 2D arrays, so reshape the 1D embeddings
        return cosine_similarity([embedding1], [embedding2])[0][0]
        
    def find_similar(self, 
                     query_embedding: np.ndarray, 
                     candidates: Dict[str, np.ndarray], 
                     threshold: float = 0.6, 
                     top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Finds similar items from a dictionary of candidate embeddings based on cosine similarity.

        Args:
            query_embedding (np.ndarray): The embedding of the query item.
            candidates (Dict[str, np.ndarray]): A dictionary where keys are item names and values are their embeddings.
            threshold (float): The minimum similarity score to consider an item "similar". Defaults to 0.6.
            top_k (Optional[int]): If specified, returns only the top 'k' most similar items. Defaults to None (return all above threshold).

        Returns:
            List[Tuple[str, float]]: A sorted list of (item_name, similarity_score) tuples,
                                      with the most similar items first.
        """
        similarities = []
        
        # Iterate through candidates and compute similarity with the query embedding
        for name, embedding in candidates.items():
            similarity = self.compute_similarity(query_embedding, embedding)
            if similarity >= threshold:
                similarities.append((name, similarity))
        
        # Sort the results by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results if specified, otherwise return all similar items
        if top_k:
            return similarities[:top_k]
        return similarities
        
    def precompute_table_embeddings(self, tables: List[str]) -> Dict[str, np.ndarray]:
        """
        Pre-computes and caches embeddings for a list of table names.

        Args:
            tables (List[str]): A list of table names.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping table names to their embeddings.
        """
        logger.info(f"Pre-computing embeddings for {len(tables)} tables")
        return self.embed_batch(tables) # Use embed_batch for efficiency
        
    def precompute_column_embeddings(self, table_columns: List[Tuple[str, str]]) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Pre-computes and caches embeddings for a list of (table_name, column_name) tuples.
        Combines table and column names into a single string for embedding.

        Args:
            table_columns (List[Tuple[str, str]]): A list of (table_name, column_name) tuples.

        Returns:
            Dict[Tuple[str, str], np.ndarray]: A dictionary mapping (table_name, column_name) tuples to their embeddings.
        """
        # Create text representations suitable for embedding (e.g., "table_name.column_name")
        texts = [f"{table}.{column}" for table, column in table_columns]
        
        logger.info(f"Pre-computing embeddings for {len(texts)} columns")
        text_embeddings = self.embed_batch(texts) # Get embeddings for the combined text strings
        
        # Map the embeddings back to the original (table, column) tuples
        column_embeddings = {}
        for (table, column), text in zip(table_columns, texts):
            if text in text_embeddings:
                column_embeddings[(table, column)] = text_embeddings[text]
        
        return column_embeddings