"""Preentrenar un GPT diminuto desde cero con tus textos."""

import json
import time
from pathlib import Path

import torch

from .model import GPTConfig, TinyGPT
from .sampling import generate
from .utils import count_parameters, get_device, human_number


def encode_corpus(tokenizer, text: str, val_fraction: float = 0.1):
    """Convierte todo el texto en números y lo separa en entrenamiento/validación."""
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    cut = int(len(ids) * (1 - val_fraction))
    print(f"Corpus: {len(ids):,} tokens  ->  entrenamiento {cut:,} · validación {len(ids) - cut:,}".replace(",", "."))
    return ids[:cut], ids[cut:]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Elige trozos al azar del texto. x = entrada, y = lo mismo desplazado un token."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def build_model(tokenizer, settings: dict, device: str | None = None) -> TinyGPT:
    """Crea un TinyGPT vacío (con pesos aleatorios) a partir de los ajustes."""
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=settings["block_size"],
        n_layer=settings["n_layer"],
        n_head=settings["n_head"],
        n_embd=settings["n_embd"],
        dropout=settings.get("dropout", 0.1),
    )
    model = TinyGPT(cfg).to(device or get_device())
    print(f"Modelo creado: {human_number(count_parameters(model))} parámetros "
          f"({cfg.n_layer} capas, {cfg.n_head} cabezas, embeddings de {cfg.n_embd}).")
    return model


@torch.no_grad()
def estimate_loss(model, train_ids, val_ids, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in [("train", train_ids), ("val", val_ids)]:
        losses = [model(*get_batch(data, block_size, batch_size, device))[1].item() for _ in range(iters)]
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def train_tiny_gpt(model, tokenizer, train_ids, val_ids, settings: dict,
                   sample_prompt: str = "En un lugar de la Mancha", sample_every: int | None = None,
                   live_plot: bool = False) -> dict:
    """Bucle de entrenamiento. Devuelve el historial de la pérdida (loss)."""
    device = next(model.parameters()).device.type
    bs, T = settings["batch_size"], model.cfg.block_size
    steps, eval_every = settings["max_steps"], settings["eval_every"]
    if len(val_ids) <= T + 1:
        raise ValueError("El texto es demasiado corto para este block_size. Añade más texto o reduce block_size.")
    opt = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=0.1)
    # El ritmo de aprendizaje baja poco a poco al final (cosine schedule)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=settings["learning_rate"] / 10)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"step": [], "train": [], "val": [], "samples": []}
    sample_every = sample_every or max(eval_every * 5, steps // 3)
    t0 = time.time()
    model.train()
    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, T, bs, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()

        if step % eval_every == 0 or step == steps or step == 1:
            losses = estimate_loss(model, train_ids, val_ids, T, bs, device,
                                   iters=5 if steps < 100 else 20)
            history["step"].append(step)
            history["train"].append(losses["train"])
            history["val"].append(losses["val"])
            print(f"paso {step:>5}/{steps}  ·  pérdida entrenamiento {losses['train']:.3f}  ·  "
                  f"validación {losses['val']:.3f}  ·  {time.time() - t0:.0f} s")
            if live_plot:
                plot_loss(history, clear=True)
        if sample_prompt and (step % sample_every == 0 or step == steps):
            s = generate(model, tokenizer, sample_prompt, max_new_tokens=40, temperature=0.8, seed=0)
            history["samples"].append((step, s))
            print(f"   ✍️  muestra: {s!r}")
            model.train()
    print(f"Entrenamiento terminado en {time.time() - t0:.0f} segundos.")
    return history


def plot_loss(history: dict, clear: bool = False):
    """Dibuja la curva de aprendizaje: la pérdida debería bajar."""
    import matplotlib.pyplot as plt
    if clear:
        from IPython.display import clear_output
        clear_output(wait=True)
    plt.figure(figsize=(8, 4))
    plt.plot(history["step"], history["train"], label="entrenamiento", marker="o")
    plt.plot(history["step"], history["val"], label="validación", marker="o")
    plt.xlabel("paso de entrenamiento")
    plt.ylabel("pérdida (loss)")
    plt.title("Curva de aprendizaje")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_tiny_gpt(model, tokenizer, folder) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "model.pt")
    (folder / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2))
    tokenizer.save(folder)
    print(f"Modelo guardado en {folder}")
    return folder


def load_tiny_gpt(folder, device: str | None = None):
    from .tokenizer import SimpleTokenizer
    folder = Path(folder)
    cfg = GPTConfig(**json.loads((folder / "config.json").read_text()))
    model = TinyGPT(cfg)
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu"))
    model.to(device or get_device()).eval()
    return model, SimpleTokenizer.load(folder)
