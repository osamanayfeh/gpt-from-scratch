"""
MiniGPT: Character-Level Decoder-Only Transformer

A from-scratch implementation of an autoregressive Transformer language model,
trained on the complete works of Shakespeare (Tiny Shakespeare dataset).

Architecture:
- Token and learned Positional Embeddings
- N stacked Transformer Blocks with Pre-LayerNorm formulation
- Multi-Head Causal Self-Attention with lower-triangular masking
- Position-wise Feed-Forward Network (4x expansion)
- Residual skip connections throughout
- Language Modeling linear projection head
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="MiniGPT: Character-Level Transformer")
    # Architectural hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--block_size", type=int, default=256, help="Maximum context length")
    parser.add_argument("--n_embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n_head", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--n_layer", type=int, default=6, help="Number of Transformer layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Optimization hyperparameters
    parser.add_argument("--max_iters", type=int, default=5000, help="Total training iterations")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW peak learning rate")
    parser.add_argument("--eval_iters", type=int, default=200, help="Evaluation iteration count")

    # Generation parameters
    parser.add_argument("--generate", action="store_true", help="Run text generation mode")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling threshold")

    # Execution and persistence
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda/mps/cpu)")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, default="checkpoint.pt", help="Path to model checkpoint")
    return parser.parse_args()


# Resolve compute device
args = parse_args()
if args.device is None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device(args.device)

torch.manual_seed(args.seed)

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset file not found at {data_path}. Please download input.txt first.")

with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()

# Character vocabulary and mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# Data splits: 90% train, 10% validation
data = torch.tensor(encode(text), dtype=torch.long)
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]


def get_batch(split: str, batch_size: int, block_size: int):
    """
    Sample a random batch of sequences and shifted targets.
    """
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int, batch_size: int, block_size: int):
    """
    Compute average loss across training and validation splits.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, batch_size, block_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


class CausalSelfAttentionHead(nn.Module):
    """
    A single head of causal self-attention.
    """
    def __init__(self, n_embd: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # Scaled dot-product attention scores
        wei = (q @ k.transpose(-2, -1)) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Context-weighted value aggregation
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel with linear projection.
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(n_embd, head_size, block_size, dropout)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Concatenate outputs across heads along the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with a 4x expansion factor.
    """
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer Block: Pre-LayerNorm formulation with communication (attention)
    followed by computation (feed-forward network).
    """
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """
    Full Decoder-Only Character-Level Transformer Language Model.
    """
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, n_head: int, n_layer: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights with standard normal distribution (std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding_table.weight.numel()
        return n_params

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"

        tok_emb = self.token_embedding_table(idx)                         # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb                                             # (B, T, n_embd)
        x = self.blocks(x)                                                # (B, T, n_embd)
        x = self.ln_f(x)                                                  # (B, T, n_embd)
        logits = self.lm_head(x)                                          # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        """
        Autoregressively sample next tokens conditioned on context idx.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds model context length
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def main():
    print("=" * 70)
    print("MiniGPT: Character-Level Decoder-Only Transformer")
    print(f"Device: {device} | Vocabulary Size: {vocab_size} | Total Characters: {len(data):,}")
    print("=" * 70)

    model = GPTLanguageModel(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        block_size=args.block_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout
    ).to(device)

    print(f"Model Parameters: {model.get_num_params():,} ({model.get_num_params()/1e6:.2f}M)")

    # Generation-only mode if requested or if checkpoint exists
    if args.generate:
        if os.path.exists(args.checkpoint):
            print(f"[INFO] Loading weights from {args.checkpoint}...")
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        else:
            print(f"[WARN] No checkpoint found at {args.checkpoint}. Sampling from untrained initialization.")

        print("\n" + "=" * 70)
        print("Generated Output:")
        print("=" * 70)
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        out = model.generate(context, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(decode(out[0].tolist()))
        print("=" * 70)
        return

    # Optimization setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    start_time = time.time()

    print("\nStarting training loop...")
    print("-" * 70)

    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss(model, args.eval_iters, args.batch_size, args.block_size)
            elapsed = time.time() - start_time
            print(f"Step {step:5d}/{args.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Time: {elapsed:.1f}s")

        xb, yb = get_batch("train", args.batch_size, args.block_size)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save trained checkpoint
    torch.save(model.state_dict(), args.checkpoint)
    print(f"\n[INFO] Model checkpoint successfully saved to {args.checkpoint}")

    # Final text generation sample
    print("\n" + "=" * 70)
    print("Post-Training Generation Sample:")
    print("=" * 70)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(decode(out[0].tolist()))
    print("=" * 70)


if __name__ == "__main__":
    main()
