"""
Bigram Language Model Baseline

A minimal character-level bigram language model implemented in PyTorch.
This model uses only the current character to predict the next character,
serving as the baseline reference before introducing Transformer attention.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

# Configuration and Hyperparameters
parser = argparse.ArgumentParser(description="Train a Bigram Language Model Baseline")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--block_size", type=int, default=8, help="Context length")
parser.add_argument("--max_iters", type=int, default=3000, help="Total training iterations")
parser.add_argument("--eval_interval", type=int, default=300, help="Interval for evaluating loss")
parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate for AdamW")
parser.add_argument("--eval_iters", type=int, default=200, help="Number of batches for loss estimation")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
args = parser.parse_args()

# Set random seed
torch.manual_seed(args.seed)
device = torch.device(args.device)

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found at {data_path}. Please download input.txt first.")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Build character vocabulary mapping
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# Train and validation split (90% train, 10% validation)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split: str):
    """
    Generate a small batch of inputs (x) and targets (y) shifted by one position.
    """
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - args.block_size, (args.batch_size,))
    x = torch.stack([source[i : i + args.block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + args.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: nn.Module):
    """
    Estimate train and validation loss over a fixed number of batches.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    """
    Bigram Language Model: each token directly looks up logits for the next token.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        # Direct lookup table from current token index to next token logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """
        Forward pass.
        Args:
            idx: Tensor of shape (B, T) containing token indices.
            targets: Optional tensor of shape (B, T) containing target labels.
        Returns:
            logits: Tensor of shape (B, T, C)
            loss: Cross-entropy scalar loss (or None if targets not provided)
        """
        # (B, T) -> (B, T, C) where C = vocab_size
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Autoregressively sample max_new_tokens conditioned on context idx.
        Args:
            idx: Tensor of shape (B, T) containing context token indices.
            max_new_tokens: Number of characters to generate.
        Returns:
            Tensor of shape (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Evaluate predictions for current context
            logits, _ = self(idx)
            # Focus only on the last time step: (B, C)
            logits = logits[:, -1, :]
            # Apply softmax to obtain probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution: (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    print("=" * 60)
    print("Bigram Language Model Baseline")
    print(f"Device: {device} | Vocab size: {vocab_size} | Training samples: {len(train_data):,}")
    print("=" * 60)

    model = BigramLanguageModel(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("\nStarting training...\n")
    for step in range(args.max_iters):
        # Periodically evaluate loss on train and validation splits
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {step:5d}/{args.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

        # Sample a batch and compute loss
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\n" + "=" * 60)
    print("Training complete. Generating sample text from Bigram model:")
    print("=" * 60)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=400)[0].tolist()
    print(decode(generated))
    print("=" * 60)


if __name__ == "__main__":
    main()
