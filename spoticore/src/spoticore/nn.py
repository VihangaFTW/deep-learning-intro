from bigram import lyrics_to_indices, build_vocabulary, SAMPLE_SEED
from reader import process_csv_file
import torch
import torch.nn as nn

EMBEDDING_DIM = 10
LEARNING_RATE = 0.1


def create_bigram_set() -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Create input-output pairs for bigram language modeling.

    Processes lyrics, builds vocabulary, converts to indices, and creates
    pairs of consecutive characters for training a bigram model.

    Returns:
        Tuple of (first_chars tensor, second_chars tensor, vocab_size).
        first_chars contains all characters except the last.
        second_chars contains all characters except the first.
        Together they form bigram pairs (first_chars[i], second_chars[i]).
    """
    lyrics = process_csv_file()

    stoi, _, vocab_size = build_vocabulary(lyrics)

    all_indices: torch.Tensor = lyrics_to_indices(lyrics, stoi)

    return all_indices[:-1], all_indices[1:], vocab_size


def encode_integer_inputs(
    indices: torch.Tensor,
    vocab_size: int,
    embedding_dim: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Encode integer indices using embedding layer instead of one-hot encoding.

    Memory-efficient alternative to F.one_hot() for large datasets.
    An integer character index is represented via a dense vector instead of
    a sparse one-hot vector.

    Creates a simple embedding layer to convert indices to dense vectors.

    Args:
        indices: Tensor of integer character indices.
        vocab_size: Size of vocabulary.
        embedding_dim: Dimension of embedding vectors.
        generator: Optional random number generator for reproducible initialization.

    Returns:
        Tensor of embedding vectors of shape (N, embedding_dim).
    """
    # Create embedding layer (lookup table: vocab_size rows × embedding_dim columns).
    # nn.Embedding initializes weights randomly, so we overwrite with seeded generator.
    embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
    if generator:
        # Use no_grad() for initialization operations (not part of training loop).
        with torch.no_grad():
            embedding.weight.data = torch.randn(
                vocab_size, embedding_dim, generator=generator
            )
    # Detach embeddings since we are not training the embedding layer weights.
    return embedding(indices.long()).detach()


def forward_pass(
    emb_inputs: torch.Tensor,
    second_chars: torch.Tensor,
    W: torch.Tensor,
    num_samples: int = 50_000,
) -> torch.Tensor:
    """
    Perform forward pass through the neural network and compute loss.

    Computes logits from embeddings, applies softmax to get probabilities,
    and calculates negative log likelihood loss.

    Args:
        emb_inputs: Input embeddings tensor of shape (N, embedding_dim).
        second_chars: Target character indices for computing loss.
        W: Weight matrix tensor of shape (embedding_dim, vocab_size).
        num_samples: Number of samples to process (default: 50,000).

    Returns:
        Loss tensor (scalar).
    """

    # Compute logits by matrix multiplication: embeddings @ weights.
    # Logits are raw scores for each possible next character.
    # Shape: (num_samples, embedding_dim) @ (embedding_dim, vocab_size) = (num_samples, vocab_size)
    logits = emb_inputs[:num_samples] @ W

    # Apply softmax activation to convert logits to probabilities.
    # Step 1: Exponentiate logits (maps any real number to positive number).
    counts = logits.exp()
    # Step 2: Normalize each row so probabilities sum to 1.
    probs = counts / counts.sum(1, keepdim=True)

    # Get probabilities for the actual observed bigrams.
    predictions = probs[torch.arange(len(probs)), second_chars[: len(probs)]]

    # Compute negative log likelihood loss.
    nll_loss = -predictions.log().mean()

    return nll_loss


def backward_pass(W: torch.Tensor, loss: torch.Tensor) -> None:
    """
    Perform backward pass to compute gradients and update weights.

    Computes gradients via backpropagation and applies gradient descent update
    with a learning rate of 0.1.

    Args:
        W: Weight tensor to update.
        loss: Scalar loss tensor.
    """
    # Clear previous gradients.
    W.grad = None
    # Compute gradients via backpropagation.
    loss.backward()
    # Verify gradients were computed.
    assert W.grad is not None
    # Update weights using gradient descent.
    W.data += -LEARNING_RATE * W.grad


if __name__ == "__main__":
    # Initialize random number generator with fixed seed for reproducibility.
    g = torch.Generator().manual_seed(SAMPLE_SEED)

    first_chars, second_chars, vocab_size = create_bigram_set()

    print(f"Vocabulary size: {vocab_size}")

    # Use the same generator for embedding initialization to ensure reproducibility.
    emb_inputs = encode_integer_inputs(
        first_chars, vocab_size, EMBEDDING_DIM, generator=g
    )

    print(f"Embedding shape: {emb_inputs.shape}")
    print(f"First embedding: {emb_inputs[1]}")

    # Initialize weight matrix with seeded generator for reproducibility.
    W = torch.randn((EMBEDDING_DIM, vocab_size), generator=g, requires_grad=True)

    # Training loop.
    num_iterations = 100
    for i in range(num_iterations):
        # Forward pass: compute predictions and loss.
        loss = forward_pass(emb_inputs, second_chars, W)

        # Print loss every 10 iterations.
        if i % 10 == 0:
            print(f"Iteration {i}: loss = {loss.item()}")

        # Backward pass: compute gradients and update weights.
        backward_pass(W, loss)
