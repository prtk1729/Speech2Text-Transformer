import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def calculate_attention(
    values: torch.Tensor,
    keys: torch.Tensor,
    query: torch.Tensor,
):
    """
    Calculates the attention weights and applies them to the values.
    Args:
        values: Tensor of shape (batch_size, num_heads, seq_len_v, head_dim) or (batch_size, seq_len_v, embed_size)
        keys: Tensor of shape (batch_size, num_heads, seq_len_k, head_dim) or (batch_size, seq_len_k, embed_size)
        query: Tensor of shape (batch_size, num_heads, seq_len_q, head_dim) or (batch_size, seq_len_q, embed_size)
    Returns:
        A tuple containing:
        - attention: The context vectors after applying attention, with shape
                     (batch_size, num_heads, seq_len_q, head_dim) or (batch_size, seq_len_q, embed_size)
        - attention_scores: The attention weights, with shape
                           (batch_size, num_heads, seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
    """
    # Calculate raw attention scores (dot product between query and keys)
    attention_scores = torch.matmul(query, keys.transpose(-2, -1))
    # Scale scores by the square root of the key dimension
    attention_scores = attention_scores / math.sqrt(keys.shape[-1])
    # Apply softmax to get attention weights
    attention_scores = F.softmax(attention_scores, dim=-1)
    # Apply attention weights to values
    attention = torch.matmul(attention_scores, values)
    return attention, attention_scores


class FeedForward(nn.Module):
    """
    A simple two-layer feed-forward neural network with GELU activation.
    Used as a sub-layer in the Transformer block.
    """

    def __init__(self, embed_size: int):
        """
        Initializes the FeedForward layer.
        Args:
            embed_size: The dimensionality of the input and output embeddings.
        """
        super().__init__()
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """
        Passes the input through the feed-forward network.
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_size).
        """
        x = self.layer1(x)
        x = F.gelu(x)  # Apply GELU activation function
        x = self.layer2(x)
        return x


class SelfAttentionLayer(nn.Module):
    """
    A basic self-attention layer without multiple heads or scaling.
    Computes attention based on query, key, and value derived from the same input.
    """

    def __init__(self, embed_size: int):
        """
        Initializes the SelfAttentionLayer.
        Args:
            embed_size: The dimensionality of the input embeddings and the
                        internal query, key, value projections.
        """
        super().__init__()
        self.embed_size = embed_size
        # Linear layers to project embeddings into query, key, and value spaces
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)

    def forward(self, embeddings: torch.Tensor):
        """
        Applies self-attention to the input embeddings.
        Args:
            embeddings: Input tensor of shape (batch_size, seq_length, embed_size).
        Returns:
            The context vectors after applying self-attention, with shape
            (batch_size, seq_length, embed_size).
        """
        # Project embeddings into query, key, and value
        query = self.query_dense(embeddings)
        key = self.key_dense(embeddings)
        value = self.value_dense(embeddings)
        # Calculate attention using the projected Q, K, V
        attention, _ = calculate_attention(value, key, query)
        return attention


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention as described in "Attention Is All You Need".
    Splits the embedding dimension into multiple heads, computes attention independently
    for each head, and then concatenates the results.
    """

    def __init__(self, embed_size: int, num_heads: int):
        """
        Initializes the MultiHeadAttention layer.
        Args:
            embed_size: The total dimensionality of the input embeddings.
            num_heads: The number of attention heads to split the embedding into.
                       Must divide embed_size evenly.
        """
        super().__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        # Dimensionality of each attention head
        self.head_dim = embed_size // num_heads

        # Single linear layers for combined Q, K, V projections across all heads
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # Final linear layer to project the concatenated heads back to embed_size
        self.output_linear = nn.Linear(embed_size, embed_size)

    def forward(self, embeddings: torch.Tensor):
        """
        Applies multi-head self-attention to the input embeddings.
        Args:
            embeddings: Input tensor of shape (batch_size, seq_length, embed_size).
        Returns:
            The context vectors after applying multi-head attention and projection,
            with shape (batch_size, seq_length, embed_size).
        """
        batch_size = embeddings.shape[0]
        seq_length = embeddings.shape[1]

        # 1. Linear projections and reshape to separate heads
        # Project embeddings into Q, K, V for all heads simultaneously
        # Then reshape: (batch_size, seq_length, embed_size) -> (batch_size, seq_length, num_heads, head_dim)
        query = self.query(embeddings).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key = self.key(embeddings).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        value = self.value(embeddings).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )

        # 2. Transpose for attention calculation
        # Change shape to: (batch_size, num_heads, seq_length, head_dim)
        # This groups data by head, making attention calculation per head efficient
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # 3. Calculate attention scores and apply attention (per head)
        # Matmul query and keys: (..., seq_len_q, head_dim) x (..., head_dim, seq_len_k) -> (..., seq_len_q, seq_len_k)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim  # Scale by sqrt(head_dim)
        )
        attention_scores = F.softmax(attention_scores, dim=-1)  # Apply softmax
        # Matmul scores and values: (..., seq_len_q, seq_len_k) x (..., seq_len_v, head_dim) -> (..., seq_len_q, head_dim)
        # Note: seq_len_k == seq_len_v for self-attention
        attention = torch.matmul(attention_scores, value)

        # 4. Reshape and concatenate heads
        # Transpose back: (batch_size, num_heads, seq_length, head_dim) -> (batch_size, seq_length, num_heads, head_dim)
        # Reshape to combine heads: (batch_size, seq_length, embed_size)
        attention = attention.transpose(1, 2).reshape(
            batch_size, seq_length, self.embed_size
        )

        # 5. Final linear projection
        output = self.output_linear(attention)

        return output


class TransformerBlock(nn.Module):
    """
    A single block of the Transformer encoder architecture.
    Consists of a self-attention layer followed by a feed-forward network,
    with residual connections and layer normalization.
    Note: This implementation uses LayerNorm *after* the attention/FFN layers,
          and applies GELU *after* the FFN layer, which differs slightly from
          the original "Attention Is All You Need" paper (which used pre-norm).
          It also omits dropout layers often found in Transformer blocks.
    """

    def __init__(self, embed_size: int):
        """
        Initializes the TransformerBlock.
        Args:
            embed_size: The dimensionality of the input embeddings.
        """
        super().__init__()
        # Uses the basic SelfAttentionLayer (could be replaced with MultiHeadAttention)
        self.attention_layer = SelfAttentionLayer(embed_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        # Note: Typically, two LayerNorms are used (one after attention, one after FFN)
        # and residual connections wrap both sub-layers. This implementation
        # combines them differently.

    def forward(self, x: torch.Tensor):
        """
        Passes the input through the Transformer block.
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
        Returns:
            Output tensor of shape (batch_size, seq_length, embed_size).
        """
        # Apply self-attention
        context = self.attention_layer(x)
        # Apply layer normalization
        context = self.layer_norm1(context)
        # Apply feed-forward network
        context = self.feed_forward(context)
        # Apply GELU activation (often part of the FFN layer itself)
        context = F.gelu(context)
        # Add residual connection (input x is added to the processed context)
        output = context + x
        return output


class SinusoidalPositionEncoding(nn.Module):
    """
    Injects positional information into the input embeddings using sinusoidal functions.
    This allows the model to understand the order of elements in the sequence.
    """

    def __init__(self, embed_size: int, max_seq_length: int):
        """
        Initializes the SinusoidalPositionEncoding layer.
        Args:
            embed_size: The dimensionality of the embeddings.
            max_seq_length: The maximum expected sequence length.
        """
        super().__init__()
        # Create position indices (0 to max_seq_length - 1)
        position = torch.arange(max_seq_length).unsqueeze(
            1
        )  # Shape: (max_seq_length, 1)
        # Calculate the division term for the sinusoidal functions
        # Uses log-space for numerical stability
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )  # Shape: (embed_size / 2)
        # Initialize positional encoding matrix
        pe = torch.zeros(
            max_seq_length, embed_size
        )  # Shape: (max_seq_length, embed_size)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register 'pe' as a buffer, so it's part of the model state but not trained
        self.register_buffer("positional_embedding", pe)

    def forward(self, x: torch.Tensor):
        """
        Adds positional encodings to the input embeddings.
        Args:
            x: Input tensor of shape (batch_size, seq_length, embed_size).
        Returns:
            Tensor with positional encodings added, shape
            (batch_size, seq_length, embed_size).
        """
        # Add the pre-computed positional embeddings (up to the input sequence length)
        # x.size(1) gives the actual sequence length of the input batch
        return x + self.positional_embedding[: x.size(1), :]


class Transformer(nn.Module):
    """
    A basic Transformer encoder model composed of multiple TransformerBlocks.
    Includes positional encoding.
    """

    def __init__(self, embed_size: int, num_layers: int, max_seq_length: int):
        """
        Initializes the Transformer model.
        Args:
            embed_size: The dimensionality of the embeddings.
            num_layers: The number of TransformerBlocks to stack.
            max_seq_length: The maximum sequence length for positional encoding.
        """
        super().__init__()
        # Add positional encoding to the input
        self.positional_encoding = SinusoidalPositionEncoding(
            embed_size, max_seq_length
        )
        # Stack multiple TransformerBlocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        """
        Passes the input through the positional encoding and Transformer blocks.
        Args:
            x: Input tensor (embeddings) of shape (batch_size, seq_length, embed_size).
        Returns:
            Output tensor from the final Transformer block, shape
            (batch_size, seq_length, embed_size).
        """
        # Apply positional encoding
        x = self.positional_encoding(x)
        # Pass through each Transformer block sequentially
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x


if __name__ == "__main__":
    # Example usage:

    # Hyperparameters
    embed_size = 32
    num_layers = 3
    max_seq_length = 600  # Maximum expected sequence length
    batch_size = 2
    seq_len = 549  # Actual sequence length for this example batch

    # Instantiate the Transformer model
    transformer = Transformer(
        embed_size=embed_size, num_layers=num_layers, max_seq_length=max_seq_length
    )

    # Create a dummy input tensor (e.g., embeddings from a previous layer)
    # Shape: (batch_size, seq_len, embed_size)
    x = torch.randn(batch_size, seq_len, embed_size)

    # Pass the input through the model
    output = transformer(x)

    # Print the shape of the output tensor
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    # Expected output shape: (batch_size, seq_len, embed_size) -> (2, 549, 32)