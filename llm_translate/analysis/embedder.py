"""Embedding generation and transformation utilities."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)


class Embedder:
    """Handles text embedding with caching and transformations."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize embedder.
        
        Args:
            cache_dir: Directory for caching embeddings
        """
        self.cache_dir = cache_dir or Path("./cache/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def embed_texts(
        self,
        texts: List[str],
        model_name: str,
        model_checkpoint: str,
        device: Optional[str] = None,
        batch_size: int = 64
    ) -> np.ndarray:
        """
        Embed texts with caching support.
        
        Args:
            texts: List of texts to embed
            model_name: Name identifier for caching
            model_checkpoint: HuggingFace model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
        
        Returns:
            Embedding matrix of shape (n_texts, embedding_dim)
        """
        cache_path = self.cache_dir / f"{model_name}_cache.npy"
        cache_index = self.cache_dir / f"{model_name}_index.json"
        
        # Check cache
        if cache_path.exists() and cache_index.exists():
            with open(cache_index, "r") as f:
                cached_data = json.load(f)
            
            if cached_data.get("texts") == texts:
                logger.info(f"Loading embeddings from cache: {cache_path}")
                return np.load(cache_path)
        
        # Generate embeddings
        logger.info(f"Generating embeddings with {model_name} ({model_checkpoint})")
        model = SentenceTransformer(model_checkpoint, device=device)
        
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=False
        )
        
        # Cache results
        np.save(cache_path, embeddings.astype(np.float32))
        with open(cache_index, "w") as f:
            json.dump({"texts": texts}, f)
        
        logger.info(f"Cached embeddings to {cache_path}")
        return embeddings
    
    def embed_multiple_models(
        self,
        texts: List[str],
        models: List[Tuple[str, str]],
        device: Optional[str] = None,
        batch_size: int = 64
    ) -> Dict[str, np.ndarray]:
        """
        Embed texts with multiple models.
        
        Args:
            texts: List of texts to embed
            models: List of (model_name, model_checkpoint) tuples
            device: Device to use
            batch_size: Batch size for encoding
        
        Returns:
            Dictionary mapping model names to embedding matrices
        """
        results = {}
        for model_name, model_checkpoint in models:
            embeddings = self.embed_texts(
                texts, model_name, model_checkpoint, device, batch_size
            )
            results[model_name] = embeddings
        
        return results
    
    @staticmethod
    def zca_whitener(X: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ZCA whitening transformation.
        
        Args:
            X: Input matrix of shape (n_samples, n_features)
            eps: Small constant for numerical stability
        
        Returns:
            Tuple of (mean, whitening_matrix)
        """
        mu = X.mean(axis=0, keepdims=True)
        X_centered = X - mu
        
        # Compute covariance matrix
        C = np.cov(X_centered, rowvar=False)
        
        # SVD
        U, s, Vt = np.linalg.svd(C, full_matrices=False)
        
        # Whitening matrix
        W = U @ np.diag(1.0 / np.sqrt(s + eps)) @ U.T
        
        return mu, W
    
    @staticmethod
    def apply_whitening(
        X: np.ndarray,
        mean: np.ndarray,
        whitening_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Apply ZCA whitening transformation.
        
        Args:
            X: Input matrix
            mean: Mean vector
            whitening_matrix: Whitening transformation matrix
        
        Returns:
            Whitened matrix
        """
        return (X - mean) @ whitening_matrix.T
    
    @staticmethod
    def orthogonal_procrustes(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Compute orthogonal Procrustes alignment from X to Y.
        
        Args:
            X: Source matrix
            Y: Target matrix
        
        Returns:
            Rotation matrix R such that X @ R ≈ Y
        """
        X_centered = X - X.mean(axis=0, keepdims=True)
        Y_centered = Y - Y.mean(axis=0, keepdims=True)
        
        U, _, Vt = np.linalg.svd(X_centered.T @ Y_centered, full_matrices=False)
        R = U @ Vt
        
        return R
    
    @staticmethod
    def l2_normalize(X: np.ndarray) -> np.ndarray:
        """
        L2 normalize vectors.
        
        Args:
            X: Input matrix
        
        Returns:
            Normalized matrix
        """
        norms = np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12
        return X / norms
    
    def whiten_and_align(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        anchor_indices: Optional[np.ndarray] = None,
        reference_model: Optional[str] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple]]:
        """
        Whiten embeddings and align to reference model.
        
        Args:
            embeddings_dict: Dictionary of model name to embeddings
            anchor_indices: Indices to use for computing whitening (None = all)
            reference_model: Model to use as alignment reference (None = first model)
        
        Returns:
            Tuple of (aligned_embeddings_dict, transform_params_dict)
        """
        model_names = list(embeddings_dict.keys())
        
        if reference_model is None:
            reference_model = model_names[0]
        
        if anchor_indices is None:
            first_model = model_names[0]
            anchor_indices = np.arange(len(embeddings_dict[first_model]))
        
        # Whiten all models
        whitened = {}
        transform_params = {}
        
        logger.info("Whitening embeddings")
        for name, X in embeddings_dict.items():
            mu, W = self.zca_whitener(X[anchor_indices])
            X_whitened = self.apply_whitening(X, mu, W)
            whitened[name] = X_whitened
            transform_params[name] = (mu, W)
        
        # Align to reference
        logger.info(f"Aligning embeddings to reference model: {reference_model}")
        aligned = {reference_model: whitened[reference_model].copy()}
        
        for name in model_names:
            if name == reference_model:
                continue
            
            X_model = whitened[name]
            X_ref = whitened[reference_model]
            
            R = self.orthogonal_procrustes(
                X_model[anchor_indices],
                X_ref[anchor_indices]
            )
            
            aligned[name] = X_model @ R
            
            # Store rotation in params
            mu, W = transform_params[name]
            transform_params[name] = (mu, W, R)
        
        return aligned, transform_params

