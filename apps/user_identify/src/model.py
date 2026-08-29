"""
Neural network models for user identification from gaze patterns.

Architecture:
1. Point Encoder: Encodes individual gaze points
2. Session Aggregator: Aggregates point features into session embedding
3. Classifier: Predicts user identity from session embedding
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointEncoder(nn.Module):
    """
    Encodes individual gaze points into feature vectors.

    Each gaze point has features like (original_gaze, target, sim_rbf, spread, residuals).
    This module transforms them into a hidden representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_points, input_dim)

        Returns:
            (batch, num_points, output_dim)
        """
        return self.encoder(x)


class AttentionAggregator(nn.Module):
    """
    Aggregates point features into a session embedding using self-attention.

    Uses multi-head self-attention followed by weighted mean pooling.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Self-attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Attention weights for pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_points, embedding_dim)
            mask: (batch, num_points) - 1 for valid, 0 for padding

        Returns:
            (batch, embedding_dim) - session embedding
        """
        batch_size = x.size(0)

        # Create attention mask (True = ignore)
        if mask is not None:
            key_padding_mask = mask == 0  # True where padding
        else:
            key_padding_mask = None

        # Self-attention to contextualize point features
        attn_out, _ = self.self_attn(
            x, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = self.layer_norm(x + attn_out)

        # Attention pooling: use learnable query to aggregate
        query = self.pool_query.expand(batch_size, -1, -1)  # (batch, 1, dim)
        pooled, attn_weights = self.pool_attn(
            query, x, x, key_padding_mask=key_padding_mask, need_weights=True
        )

        return pooled.squeeze(1)  # (batch, embedding_dim)


class MeanAggregator(nn.Module):
    """Simple mean pooling aggregator."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_points, embedding_dim)
            mask: (batch, num_points)

        Returns:
            (batch, embedding_dim)
        """
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, num_points, 1)
            x = x * mask
            return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return x.mean(dim=1)


class MaxAggregator(nn.Module):
    """Max pooling aggregator."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, num_points, embedding_dim)
            mask: (batch, num_points)

        Returns:
            (batch, embedding_dim)
        """
        if mask is not None:
            # Set padding positions to very negative value
            mask = mask.unsqueeze(-1)  # (batch, num_points, 1)
            x = x.masked_fill(mask == 0, float("-inf"))
        return x.max(dim=1)[0]


class Classifier(nn.Module):
    """Classification head for user identification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [64, 32],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - session embedding

        Returns:
            (batch, num_classes) - logits
        """
        return self.classifier(x)


class GazeIdentityNet(nn.Module):
    """
    Full model for user identification from gaze patterns.

    Architecture:
    1. Point Encoder: Encodes individual gaze points
    2. Session Aggregator: Aggregates point features to session embedding
    3. Classifier: Predicts user identity

    Also supports contrastive learning by outputting normalized embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        point_encoder_dims: List[int] = [128, 64],
        aggregator_type: str = "attention",
        num_heads: int = 4,
        classifier_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        embedding_dim: Optional[int] = None,
    ) -> None:
        """
        Args:
            input_dim: Number of features per gaze point
            num_classes: Number of users to classify
            point_encoder_dims: Hidden dimensions for point encoder
            aggregator_type: "attention", "mean", or "max"
            num_heads: Number of attention heads (for attention aggregator)
            classifier_dims: Hidden dimensions for classifier
            dropout: Dropout rate
            embedding_dim: Dimension of session embedding (default: last encoder dim)
        """
        super().__init__()

        self.point_encoder = PointEncoder(
            input_dim=input_dim,
            hidden_dims=point_encoder_dims,
            dropout=dropout,
        )

        self.embedding_dim = embedding_dim or self.point_encoder.output_dim

        # Optional projection to embedding_dim
        if self.embedding_dim != self.point_encoder.output_dim:
            self.projection = nn.Linear(
                self.point_encoder.output_dim, self.embedding_dim
            )
        else:
            self.projection = nn.Identity()

        # Aggregator
        if aggregator_type == "attention":
            self.aggregator = AttentionAggregator(
                embedding_dim=self.embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif aggregator_type == "mean":
            self.aggregator = MeanAggregator(self.embedding_dim)
        elif aggregator_type == "max":
            self.aggregator = MaxAggregator(self.embedding_dim)
        else:
            raise ValueError(f"Unknown aggregator type: {aggregator_type}")

        self.classifier = Classifier(
            input_dim=self.embedding_dim,
            num_classes=num_classes,
            hidden_dims=classifier_dims,
            dropout=dropout,
        )

    def encode_session(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode a session into an embedding vector.

        Args:
            x: (batch, num_points, input_dim)
            mask: (batch, num_points)

        Returns:
            (batch, embedding_dim) - session embedding
        """
        # Encode points
        point_features = self.point_encoder(x)  # (batch, num_points, encoder_dim)
        point_features = self.projection(point_features)  # (batch, num_points, embedding_dim)

        # Aggregate to session embedding
        session_embedding = self.aggregator(point_features, mask)  # (batch, embedding_dim)

        return session_embedding

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: (batch, num_points, input_dim)
            mask: (batch, num_points)
            return_embedding: Whether to return the session embedding

        Returns:
            logits: (batch, num_classes)
            embedding: (batch, embedding_dim) if return_embedding else None
        """
        session_embedding = self.encode_session(x, mask)
        logits = self.classifier(session_embedding)

        if return_embedding:
            # Normalize embedding for contrastive learning
            normalized_embedding = F.normalize(session_embedding, p=2, dim=-1)
            return logits, normalized_embedding

        return logits, None


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for learning discriminative embeddings.

    Pulls embeddings of the same user together, pushes different users apart.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (batch, embedding_dim) - L2 normalized embeddings
            labels: (batch,) - user IDs

        Returns:
            Scalar loss
        """
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create mask for positive pairs (same user, different sample)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        eye = torch.eye(batch_size, device=embeddings.device)
        mask = mask - eye

        # Check if there are any positive pairs
        if mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # For numerical stability
        logits_max, _ = similarity.max(dim=1, keepdim=True)
        similarity = similarity - logits_max.detach()

        # Compute log_softmax over all pairs except self
        exp_sim = torch.exp(similarity) * (1 - eye)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average log probability over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Loss is negative log probability
        loss = -mean_log_prob_pos.mean()

        return loss


def build_model(
    input_dim: int,
    num_classes: int,
    config: dict,
) -> GazeIdentityNet:
    """
    Build model from configuration dictionary.

    Args:
        input_dim: Number of features per gaze point
        num_classes: Number of users
        config: Configuration dictionary with model parameters
    """
    point_cfg = config.get("point_encoder", {})
    agg_cfg = config.get("aggregator", {})
    cls_cfg = config.get("classifier", {})

    return GazeIdentityNet(
        input_dim=input_dim,
        num_classes=num_classes,
        point_encoder_dims=point_cfg.get("hidden_dims", [128, 64]),
        aggregator_type=agg_cfg.get("type", "attention"),
        num_heads=agg_cfg.get("num_heads", 4),
        classifier_dims=cls_cfg.get("hidden_dims", [64, 32]),
        dropout=point_cfg.get("dropout", 0.1),
        embedding_dim=agg_cfg.get("embedding_dim", 64),
    )
