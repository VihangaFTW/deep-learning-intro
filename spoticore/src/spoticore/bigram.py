from reader import process_csv_file
import matplotlib.pyplot as plt
import torch


SAMPLE_SEED = 534150593


def build_vocabulary(lyrics: list[str]) -> tuple[dict[str, int], dict[int, str], int]:
    """
    Build vocabulary from lyrics and create string-to-index and index-to-string mappings.

    Args:
        lyrics: List of lyrics strings.

    Returns:
        Tuple of (stoi dictionary, itos dictionary, vocab_size).
    """
    whole_text = "".join(lyrics)
    chars = sorted(list(set(whole_text)))

    # Create mappings with special token "*" at index 0.
    stoi: dict[str, int] = {"*": 0}
    for i, char in enumerate(chars):
        stoi[char] = i + 1

    itos: dict[int, str] = {i: char for char, i in stoi.items()}
    vocab_size = len(stoi)

    return stoi, itos, vocab_size


def lyrics_to_indices(
    lyrics: list[str], stoi: dict[str, int], progress_interval: int = 10000
) -> torch.Tensor:
    """
    Convert all lyrics to character indices with start/end tokens.

    Args:
        lyrics: List of lyrics strings.
        stoi: String-to-index mapping dictionary.
        progress_interval: Print progress every N songs.

    Returns:
        Tensor of character indices.
    """
    print(f"Processing {len(lyrics)} songs...")
    all_indices = []

    for i, sample in enumerate(lyrics):
        if (i + 1) % progress_interval == 0:
            print(f"Processed {i + 1}/{len(lyrics)} songs...")

        # Add start token, character indices, and end token.
        sample_indices = (
            [stoi["*"]] + [stoi.get(c, stoi["*"]) for c in sample] + [stoi["*"]]
        )
        all_indices.extend(sample_indices)

    return torch.tensor(all_indices, dtype=torch.int32)


def count_bigrams(indices_tensor: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Count bigram frequencies using vectorized operations.

    Args:
        indices_tensor: Tensor of character indices.
        vocab_size: Size of vocabulary.

    Returns:
        2D tensor of bigram frequencies (vocab_size x vocab_size).
    """
    print("Converting to tensor and counting bigrams...")

    # Create bigram pairs using slicing.
    first_chars = indices_tensor[:-1]
    second_chars = indices_tensor[1:]

    # Flatten 2D indices to 1D for scatter_add.
    # PyTorch requires long dtype for indexing operations.
    bigram_flat_indices = (first_chars * vocab_size + second_chars).long()
    N_flat = torch.zeros(vocab_size * vocab_size, dtype=torch.int32)
    N_flat.scatter_add_(
        0, bigram_flat_indices, torch.ones_like(bigram_flat_indices, dtype=torch.int32)
    )

    return N_flat.view(vocab_size, vocab_size)


def plot_bigram_heatmap(
    N: torch.Tensor,
    itos: dict[int, str],
    vocab_size: int,
    figsize: tuple[int, int] = (24, 24),
    cmap: str = "Blues",
) -> None:
    """
    Create and display a bigram frequency heatmap.

    Args:
        N: 2D tensor of bigram frequencies.
        itos: Index-to-string mapping dictionary.
        vocab_size: Size of vocabulary.
        figsize: Figure size tuple.
        cmap: Colormap name.
    """

    # Use log scale for better visualization.
    N_log = torch.log1p(N).numpy()

    _, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(N_log, cmap=cmap, aspect="auto")

    # Add colorbar.
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Log Frequency", rotation=270, labelpad=20)

    # Set axis labels to show characters.
    ax.set_xticks(range(vocab_size))
    ax.set_yticks(range(vocab_size))
    ax.set_xticklabels([itos[i] for i in range(vocab_size)], fontsize=8)
    ax.set_yticklabels([itos[i] for i in range(vocab_size)], fontsize=8)
    ax.set_xlabel("Second Character", fontsize=12)
    ax.set_ylabel("First Character", fontsize=12)

    for row in range(vocab_size):
        for col in range(vocab_size):
            count = N[row, col].item()
            bigram: str = itos[row] + itos[col]

            # Use white text on dark blue, black on light blue.
            text_color = "white" if N_log[row, col] > N_log.max() * 0.5 else "black"

            # Smaller font size to fit all bigrams.
            fontsize = 7 if count > 1000 else 6

            # Show bigram.
            ax.text(
                col,
                row,
                bigram,
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
                fontweight="bold" if count > 10000 else "normal",
            )

    plt.title("Bigram Frequency Heatmap (Log Scale)", fontsize=14, pad=20)
    plt.show()


def sampling(N: torch.Tensor, itos: dict[int, str]) -> list[str]:
    """
    Generate text samples using bigram probability distributions.

    Samples characters sequentially based on bigram probabilities, starting from
    the start token and continuing until the end token is sampled. Generates
    20 sample sequences and prints each one.

    Args:
        N: 2D tensor of bigram frequency counts (vocab_size x vocab_size).
        itos: Index-to-string mapping dictionary.

    Returns:
        List of characters from the last generated sample sequence.
    """
    # Initialize random number generator with fixed seed for reproducibility.
    generator = torch.Generator().manual_seed(SAMPLE_SEED)

    # Convert bigram counts to float for probability calculations.
    P = N.float()
    # Normalize each row to convert counts into probabilities.
    # Each row sums to 1.0, representing the probability distribution of the next character given the current character.
    P /= P.sum(1, keepdim=True)

    print(P[0].sum())

    # Generate 20 samples of text sequences.
    for i in range(20):
        outputs = []
        # Start sampling from index 0, which represents the start token "*".
        sample_idx = 0
        while True:
            # Get the probability distribution for the next character given the current character.
            row_probs = P[sample_idx]

            # Sample the next character index from the probability distribution.
            # multinomial samples according to the probabilities in row_probs.
            sample_idx = int(
                torch.multinomial(
                    row_probs, num_samples=1, replacement=True, generator=generator
                ).item()
            )

            # Convert the sampled index to its corresponding character and append to outputs.
            outputs.append(itos[sample_idx])

            # Stop sampling when the end token "*" (index 0) is chosen.
            if not sample_idx:
                # index 0 means end character * chosen
                break

        print(f"\nIteration {i}:\n", "".join(outputs))

    return outputs


def main() -> None:
    """Main function to process lyrics and create bigram heatmap."""
    # Load lyrics.
    lyrics = process_csv_file()

    # Build vocabulary.
    stoi, itos, vocab_size = build_vocabulary(lyrics)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {stoi}")

    # Convert lyrics to indices.
    indices_tensor = lyrics_to_indices(lyrics, stoi)

    # Count bigrams.
    N = count_bigrams(indices_tensor, vocab_size)

    print("Done processing.")

    # Visualize.
    # plot_bigram_heatmap(N, itos, vocab_size)

    choices = sampling(N, itos)

    print(f"{choices=}")


if __name__ == "__main__":
    main()
