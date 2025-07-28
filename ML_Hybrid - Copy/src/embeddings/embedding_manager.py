"""
Embedding management module for the ML Hybrid Theme Analysis system.

This module provides functionality to generate embeddings using AWS Bedrock
Titan embeddings through LangChain, with support for batch processing
and vector storage.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from langchain_aws import BedrockEmbeddings
from langchain.schema import Document
from loguru import logger
from tqdm import tqdm

from ..utils.config import config
from ..utils.logger import get_logger

# Initialize logger
log = get_logger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation using AWS Bedrock Titan embeddings.

    This class provides functionality to generate embeddings for text
    using AWS Bedrock's Titan embeddings model through LangChain,
    with support for batch processing and error handling.
    """

    def __init__(self):
        """Initialize the embedding manager with AWS Bedrock configuration."""
        self.config = config.get_aws_config()
        self.embedding_model_id = self.config.get("bedrock", {}).get(
            "embedding_model_id", "amazon.titan-embed-text-v1"
        )

        try:
            # Initialize Bedrock embeddings
            self.embeddings = BedrockEmbeddings(
                model_id=self.embedding_model_id,
                region_name=self.config.get("region", "us-east-1"),
            )
            log.info(
                f"Embedding manager initialized with model: {self.embedding_model_id}"
            )

        except Exception as e:
            log.error(f"Failed to initialize embedding manager: {e}")
            raise

    def generate_embeddings(
        self, texts: List[str], batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        log.info(f"Generating embeddings for {len(texts)} texts")

        embeddings = []
        errors = 0

        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i : i + batch_size]

            try:
                batch_embeddings = self.embeddings.embed_documents(batch)
                embeddings.extend(batch_embeddings)

            except Exception as e:
                log.error(f"Error generating embeddings for batch {i//batch_size}: {e}")
                errors += 1
                # Add zero vectors for failed embeddings
                embeddings.extend(
                    [[0.0] * 1536] * len(batch)
                )  # Titan embeddings are 1536-dimensional

        log.info(
            f"Embedding generation completed. {len(embeddings)} embeddings generated, {errors} errors"
        )
        return embeddings

    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding

        except Exception as e:
            log.error(f"Error generating embedding for text: {e}")
            return [0.0] * 1536  # Return zero vector on error

    def create_documents(
        self, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """
        Create LangChain Document objects from texts.

        Args:
            texts: List of text strings
            metadata: Optional list of metadata dictionaries

        Returns:
            List of Document objects
        """
        documents = []

        for i, text in enumerate(texts):
            doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
            document = Document(page_content=text, metadata=doc_metadata)
            documents.append(document)

        return documents

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension
        """
        # Titan embeddings are 1536-dimensional
        return 1536

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            log.error(f"Error computing similarity: {e}")
            return 0.0

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find the most similar embeddings to a query embedding.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top similar embeddings to return

        Returns:
            List of dictionaries with index and similarity score
        """
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({"index": i, "similarity": similarity})

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def batch_embed_with_metadata(
        self, texts: List[str], metadata: List[Dict[str, Any]], batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Generate embeddings with metadata in batches.

        Args:
            texts: List of text strings
            metadata: List of metadata dictionaries
            batch_size: Batch size for processing

        Returns:
            Dictionary containing embeddings and metadata
        """
        log.info(f"Generating embeddings with metadata for {len(texts)} texts")

        embeddings = self.generate_embeddings(texts, batch_size)

        # Combine embeddings with metadata
        results = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            results.append(
                {"index": i, "embedding": embedding, "metadata": meta, "text": texts[i]}
            )

        return {"embeddings": embeddings, "results": results, "count": len(embeddings)}

    def validate_embedding(self, embedding: List[float]) -> bool:
        """
        Validate if an embedding vector is valid.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if embedding is valid
        """
        if not embedding or not isinstance(embedding, list):
            return False

        if len(embedding) != self.get_embedding_dimension():
            return False

        # Check if all values are finite numbers
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in embedding):
            return False

        return True
