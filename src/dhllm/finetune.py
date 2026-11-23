"""Ajuste fino (fine-tuning) de un modelo pequeño ya preentrenado de Hugging Face.

Dos métodos:
  - "lora": solo se entrenan unas pocas matrices pequeñas añadidas al modelo
            (rápido, poca memoria; recomendado en Colab gratuito).
  - "full": se entrenan todos los parámetros (más memoria, más riesgo de
            «olvidar» lo aprendido antes).
"""

import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from .utils import count_parameters, get_device, human_number


def load_base_model(name: str = "HuggingFaceTB/SmolLM2-135M", device: str | None = None):
    """Descarga (la primera vez) y carga un modelo y su tokenizador desde Hugging Face."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or get_device()
    print(f"Cargando '{name}' ... (la primera vez se descarga, puede tardar un poco)")
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name)  # float32: funciona en CPU y en T4
    model.to(device)
    print(f"Modelo listo: {human_number(count_parameters(model))} parámetros · dispositivo: {device}")
    return model, tokenizer


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Añade adaptadores LoRA. Solo esos adaptadores se entrenarán."""
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = count_parameters(model)
    print(f"LoRA activado: se entrenan {human_number(trainable)} de {human_number(total)} "
          f"parámetros ({100 * trainable / total:.2f} %).")
    return model


def prepare_model(model, method: str = "lora", lora_r: int = 16, lora_alpha: int = 32):
    if method == "lora":
        return add_lora(model, r=lora_r, alpha=lora_alpha)
    if method == "full":
        print(f"Ajuste completo: se entrenan los {human_number(count_parameters(model))} parámetros.")
        return model
    raise ValueError("method debe ser 'lora' o 'full'")


def make_chunks(texts, tokenizer, block_size: int = 256, val_fraction: float = 0.1):
    """Convierte tus textos en trozos de `block_size` tokens para entrenar."""
    text = "\n\n".join(getattr(t, "content", t) for t in texts) if not isinstance(texts, str) else texts
    old_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e12)  # evita un aviso: aquí tokenizamos todo el corpus de una vez
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    tokenizer.model_max_length = old_max
    n = len(ids) // block_size
    if n < 2:
        raise ValueError("Hay muy poco texto. Añade más textos o reduce block_size.")
    chunks = torch.tensor(ids[: n * block_size]).view(n, block_size)
    n_val = max(1, int(n * val_fraction))
    print(f"{len(ids):,} tokens -> {n - n_val} trozos de entrenamiento y {n_val} de validación "
          f"({block_size} tokens cada uno).".replace(",", "."))
    return chunks[n_val:], chunks[:n_val]  # validación = el principio del corpus


def _amp(device: str):
    if device == "cuda":  # en GPU: media precisión (float16), más rápido y menos memoria
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, chunks, batch_size: int = 8, max_batches: int = 10) -> dict:
    """Pérdida media y «perplejidad» en los trozos de validación (más bajo = mejor)."""
    model.eval()
    device = next(model.parameters()).device.type
    losses = []
    for i in range(0, min(len(chunks), batch_size * max_batches), batch_size):
        x = chunks[i: i + batch_size].to(device)
        with _amp(device):
            losses.append(model(input_ids=x, labels=x).loss.float().item())
    loss = sum(losses) / len(losses)
    return {"loss": round(loss, 3), "perplejidad": round(math.exp(min(loss, 20)), 1)}


def train_loop(model, next_batch, settings: dict, steps: int, eval_fn=None, eval_every: int = 50) -> dict:
    """Bucle de entrenamiento genérico (lo usan el ajuste fino y el SFT)."""
    device = next(model.parameters()).device.type
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=settings["learning_rate"], weight_decay=0.0)
    warmup = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.1, 1 - s / steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    history = {"step": [], "train": [], "val": []}
    running, t0 = [], time.time()
    model.train()
    for step in range(1, steps + 1):
        batch = {k: v.to(device) for k, v in next_batch().items()}
        with _amp(device):
            loss = model(**batch).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running.append(loss.item())
        if step % eval_every == 0 or step == steps:
            history["step"].append(step)
            history["train"].append(sum(running) / len(running))
            running = []
            msg = f"paso {step:>4}/{steps} · pérdida {history['train'][-1]:.3f}"
            if eval_fn:
                val = eval_fn()
                history["val"].append(val["loss"])
                msg += f" · validación {val['loss']:.3f} (perplejidad {val['perplejidad']})"
                model.train()
            print(msg + f" · {time.time() - t0:.0f} s")
    print(f"Listo en {time.time() - t0:.0f} segundos.")
    return history


def finetune_on_texts(model, train_chunks, val_chunks, settings: dict) -> dict:
    """Ajuste fino con tus textos: el modelo aprende a «escribir como» tu corpus."""
    bs, steps = settings["batch_size"], settings["max_steps"]

    def next_batch():
        idx = torch.randint(len(train_chunks), (bs,))
        x = train_chunks[idx]
        return {"input_ids": x, "labels": x}

    return train_loop(model, next_batch, settings, steps,
                      eval_fn=lambda: evaluate(model, val_chunks, batch_size=bs,
                                               max_batches=settings.get("eval_batches", 10)),
                      eval_every=max(1, min(50, steps // 6)))


def generate_hf(model, tokenizer, prompt: str, max_new_tokens=80, temperature=0.8,
                top_p=0.95, top_k=50, seed=0) -> str:
    """Genera texto con un modelo de Hugging Face."""
    model.eval()
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=top_p, top_k=top_k,
                             pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.1)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def generations_table(model, tokenizer, prompts, **kwargs) -> dict:
    return {p: generate_hf(model, tokenizer, p, **kwargs) for p in prompts}


def compare_before_after(before: dict, after: dict):
    """Muestra lado a lado lo que generaba el modelo antes y después del ajuste."""
    try:
        import html
        from IPython.display import HTML, display
        rows = "".join(
            f"<tr><td style='vertical-align:top'><b>{html.escape(p)}</b></td>"
            f"<td style='vertical-align:top;white-space:pre-wrap'>{html.escape(before[p])}</td>"
            f"<td style='vertical-align:top;white-space:pre-wrap'>{html.escape(after[p])}</td></tr>"
            for p in before)
        display(HTML("<table><tr><th>Inicio (prompt)</th><th>ANTES</th><th>DESPUÉS</th></tr>" + rows + "</table>"))
    except ImportError:
        for p in before:
            print(f"### {p}\n--- ANTES ---\n{before[p]}\n--- DESPUÉS ---\n{after[p]}\n")


def save_model(model, tokenizer, folder) -> Path:
    """Guarda el modelo (o solo los adaptadores LoRA, que ocupan pocos MB)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)
    size = sum(f.stat().st_size for f in folder.glob("**/*") if f.is_file()) / 1e6
    print(f"Guardado en {folder} ({size:.1f} MB)")
    return folder


def save_adapter(model, folder, half: bool = True) -> Path:
    """Guarda solo los adaptadores LoRA (en float16 ocupan la mitad). Sirve como copia de seguridad."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if half:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.half()
    model.save_pretrained(folder)
    if half:
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
    size = sum(f.stat().st_size for f in folder.glob("*") if f.is_file()) / 1e6
    print(f"Adaptador LoRA guardado en {folder} ({size:.1f} MB)")
    return folder


def load_adapter(model, folder):
    """Carga unos adaptadores LoRA ya entrenados (p. ej. la copia de seguridad del repositorio)."""
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(folder))
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
    print(f"Adaptador cargado desde {folder}")
    return model
