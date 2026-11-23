"""Generar texto: temperatura, top-k y top-p (muestreo por núcleo)."""

import torch
import torch.nn.functional as F


def filter_logits(logits: torch.Tensor, temperature=1.0, top_k=None, top_p=None) -> torch.Tensor:
    """Aplica temperatura, top-k y top-p a los «logits» (puntuaciones) del modelo.

    - temperatura < 1: más conservador; > 1: más creativo (y más caótico).
    - top_k: solo se consideran los k tokens más probables.
    - top_p: solo los tokens más probables cuya probabilidad acumulada llega a p.
    """
    logits = logits / max(float(temperature), 1e-5)
    if top_k:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = sorted_logits.softmax(-1).cumsum(-1)
        remove = cum - sorted_logits.softmax(-1) > top_p  # siempre conserva el primero
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


def _is_hf(model) -> bool:
    return hasattr(model, "config") and hasattr(model, "generate")


def _encode(tokenizer, text):
    if hasattr(tokenizer, "tokens"):  # nuestro SimpleTokenizer
        return tokenizer.encode(text)
    return tokenizer.encode(text, add_special_tokens=False)


def _next_logits(model, ids: torch.Tensor) -> torch.Tensor:
    if _is_hf(model):
        return model(input_ids=ids).logits[:, -1, :]
    ids = ids[:, -model.cfg.block_size:]
    logits, _ = model(ids)
    return logits[:, -1, :]


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens=100, temperature=0.8,
             top_k=50, top_p=0.95, seed=None, stop_at_eos=True) -> str:
    """Continúa un texto (prompt) token a token. Funciona con TinyGPT y con modelos de Hugging Face."""
    model.eval()
    device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    ids = _encode(tokenizer, prompt) or [getattr(tokenizer, "eos_id", 0)]
    ids = torch.tensor([ids], device=device)
    eos = getattr(tokenizer, "eos_id", None) if hasattr(tokenizer, "tokens") else tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        logits = _next_logits(model, ids)
        if temperature == 0:  # «greedy»: siempre el más probable
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = F.softmax(filter_logits(logits, temperature, top_k, top_p), dim=-1)
            nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        if stop_at_eos and eos is not None and nxt.item() == eos:
            break
    return tokenizer.decode(ids[0].tolist())


@torch.no_grad()
def next_token_table(model, tokenizer, prompt: str, temperature=1.0, top_k=None, top_p=None,
                     n: int = 15, plot: bool = True) -> list[tuple[str, float]]:
    """Muestra los n tokens siguientes más probables y su probabilidad."""
    model.eval()
    device = next(model.parameters()).device
    ids = torch.tensor([_encode(tokenizer, prompt)], device=device)
    probs = F.softmax(filter_logits(_next_logits(model, ids), temperature, top_k, top_p), dim=-1)[0]
    top = torch.topk(probs, n)
    rows = [(tokenizer.decode([i]), p) for i, p in zip(top.indices.tolist(), top.values.tolist())]
    if plot:
        import matplotlib.pyplot as plt
        labels = [repr(t) for t, _ in rows][::-1]
        values = [p for _, p in rows][::-1]
        plt.figure(figsize=(7, 0.35 * n + 1))
        plt.barh(labels, values, color="#1f5f8b")
        plt.xlabel("probabilidad")
        plt.title(f"¿Qué viene después de «…{prompt[-30:]}»?\n(temperatura={temperature}, top_k={top_k}, top_p={top_p})")
        plt.tight_layout()
        plt.show()
    else:
        for t, p in rows:
            print(f"{t!r:>20}  {p:.3f}")
    return rows


def compare_settings(model, tokenizer, prompt: str, settings: list[dict], max_new_tokens=60, seed=0):
    """Genera con varios ajustes distintos para compararlos lado a lado."""
    out = []
    for s in settings:
        text = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, seed=seed, **s)
        out.append((s, text))
        print(f"=== {s} ===\n{text}\n")
    return out
