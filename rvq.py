"""Implements Vector Quantization (VQ) and Residual Vector Quantization (RVQ).
Vector Quantization maps continuous or discrete data to a finite set of vectors
(the codebook). Residual VQ applies VQ iteratively to the residual of the
previous stage, allowing for higher fidelity representation.
"""

import torch
import torch.nn as nn


class VectorQuantizer(nn.Module):
    """Applies Vector Quantization to an input tensor.
    Maps input vectors to the closest vector in the codebook/embedding space.
    Args:
        num_embeddings (int): The number of vectors in the codebook (codebook size).
        embedding_dim (int): The dimensionality of the embedding vectors.
        commitment_cost (float, optional): Weight for the commitment loss term.
                                          Defaults to 0.25.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """Initializes the VectorQuantizer module."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Initialize embedding with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        self.commitment_cost = commitment_cost

    def forward(self, x):
        """Forward pass of the VectorQuantizer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D), where B is batch size,
                              T is sequence length, and D is embedding dimension.
        Returns:
            tuple:
                - torch.Tensor: Quantized output tensor of the same shape as input.
                - torch.Tensor: The VQ loss (scalar).
        """
        # Shape: (B, T, D)
        batch_size, sequence_length, embedding_dim = x.shape
        flat_x = x.reshape(batch_size * sequence_length, embedding_dim)  # (B*T, D)

        # Compute distances between flattened input and embedding vectors
        # distances shape: (B*T, N) where N is num_embeddings
        distances = torch.cdist(
            flat_x, self.embedding.weight, p=2
        )  # p=2 for Euclidean distance

        # Encoding: Find the index of the closest embedding vector for each input vector
        encoding_indices = torch.argmin(distances, dim=1)  # Shape: (B*T)
        # Retrieve the quantized vectors using the indices
        quantized = self.embedding(encoding_indices).view(
            batch_size, sequence_length, embedding_dim
        )  # Shape: (B, T, D)

        # VQ Loss Calculation (VQ-VAE paper approach)
        # 1. Embedding Loss (e_latent_loss): Encourages embedding vectors to match encoder output.
        #    Uses .detach() on quantized to stop gradients flowing back to embeddings from this term.
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        # 2. Commitment Loss (q_latent_loss): Encourages encoder output to commit to an embedding vector.
        #    Uses .detach() on x to stop gradients flowing back to the encoder from this term.
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)

        # Total VQ loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-Through Estimator (STE)
        # Copies the gradient from the quantized output back to the input 'x'.
        # This allows gradients to flow through the non-differentiable quantization step.
        quantized = x + (quantized - x).detach()

        return quantized, loss


class ResidualVectorQuantizer(nn.Module):
    """Applies Residual Vector Quantization using multiple VectorQuantizer codebooks.
    Quantizes the input by iteratively quantizing the residual from the previous
    quantization stage.
    Args:
        num_codebooks (int): The number of cascaded VectorQuantizer modules.
        codebook_size (int): The size of the codebook for each VectorQuantizer.
        embedding_dim (int): The dimensionality of the embedding vectors.
    """

    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        """Initializes the ResidualVectorQuantizer module."""
        super().__init__()
        # Create a list of VectorQuantizer modules (codebooks)
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim)
                for _ in range(num_codebooks)
            ]
        )

    def forward(self, x):
        """Forward pass of the ResidualVectorQuantizer.
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D).
        Returns:
            tuple:
                - torch.Tensor: The final quantized output tensor, sum of all codebook outputs.
                - torch.Tensor: The total VQ loss accumulated across all codebooks.
        """
        out = 0  # Stores the accumulated quantized output
        total_loss = 0  # Stores the accumulated VQ loss
        residual = x  # Initialize residual with the input

        # Iterate through each codebook
        for codebook in self.codebooks:
            # Quantize the current residual
            this_output, this_loss = codebook(residual)
            # Update the residual for the next stage
            residual = residual - this_output
            # Accumulate the quantized output
            out = out + this_output
            # Accumulate the loss
            total_loss += this_loss
        return out, total_loss


if __name__ == "__main__":
    # --- Example Usage ---
    torch.manual_seed(42)

    # Configuration
    num_codebooks = 3
    codebook_size = 16
    embedding_dim = 32
    batch_size = 2
    sequence_length = 12

    # Instantiate the Residual Vector Quantizer
    rvq = ResidualVectorQuantizer(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
    )

    # Create a random input tensor
    x = torch.randn(batch_size, sequence_length, embedding_dim, requires_grad=True)

    # Set up optimizer
    optimizer = torch.optim.Adam(rvq.parameters(), lr=0.01)

    print("Starting training loop example...")
    # Example training loop
    for i in range(1000):
        # Forward pass through the RVQ
        output, vq_loss = rvq(x)

        # Example: Add a task-specific loss (e.g., reconstruction loss)
        # In a real scenario, this might be part of a larger model's loss.
        recon_loss = torch.mean((output - x) ** 2)
        # Combine the VQ loss with the task-specific loss
        total_loss = recon_loss + vq_loss

        # Backward pass and optimization step
        optimizer.zero_grad()  # Clear previous gradients
        total_loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters

        # Print progress (optional)
        if (i + 1) % 100 == 0:
            print(
                f"Iteration {i+1}, Total Loss: {total_loss.item():.4f}, VQ Loss: {vq_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}"
            )

    print("Training loop finished.")
    print("Final Output Shape:", output.shape)
    print("Final VQ Loss:", vq_loss.item())