"""Ajuste por instrucciones (SFT): enseñar al modelo a responder preguntas.

Formato de los datos (un objeto JSON por línea, archivo .jsonl):
  {"pregunta": "¿Quién es Sancho Panza?", "respuesta": "El escudero de don Quijote..."}
También se aceptan las claves "instruction"/"output" o "prompt"/"response".
"""

import json
import random
from pathlib import Path

import torch

from .finetune import train_loop

KEYS = [("pregunta", "respuesta"), ("instruction", "output"), ("prompt", "response"), ("question", "answer")]
SYSTEM_PROMPT = "Eres un asistente experto en literatura del Siglo de Oro. Responde en español, de forma breve."


def load_qa(path) -> list[dict]:
    """Lee pares pregunta/respuesta de un archivo .jsonl o .csv."""
    path = Path(path)
    pairs = []
    if path.suffix == ".csv":
        import csv
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    for r in rows:
        for q, a in KEYS:
            if q in r and a in r:
                pairs.append({"pregunta": r[q].strip(), "respuesta": r[a].strip()})
                break
    print(f"{len(pairs)} pares pregunta-respuesta cargados de {path.name}.")
    return pairs


def format_prompt(tokenizer, question: str, system: str | None = None, context: str | None = None) -> str:
    """Construye el texto de entrada con la «plantilla de chat» del modelo.

    context: fragmentos de texto recuperados (RAG) que el modelo debe usar para responder.
    """
    system = system or SYSTEM_PROMPT
    if context:
        question = (f"FRAGMENTOS:\n{context}\n\n"
                    "Responde en una sola frase usando SOLO la información de los fragmentos. "
                    "Termina con la fuente entre corchetes, por ejemplo: [1]. "
                    "Si la respuesta no está en los fragmentos, responde «No lo sé».\n\n"
                    f"PREGUNTA: {question}")
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n### Pregunta:\n{question}\n\n### Respuesta:\n"


def encode_pair(tokenizer, pair: dict, max_length: int = 384) -> dict:
    """Codifica un par. La pérdida solo se calcula sobre la RESPUESTA (labels=-100 en la pregunta).

    Si el par es demasiado largo y lleva fragmentos (contexto RAG), se recortan los
    fragmentos para que la respuesta quepa siempre dentro de `max_length`.
    """
    answer_ids = tokenizer(pair["respuesta"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    context = pair.get("contexto")

    def prompt_ids_for(ctx):
        prompt = format_prompt(tokenizer, pair["pregunta"], system=pair.get("system"), context=ctx)
        return tokenizer(prompt, add_special_tokens=False)["input_ids"]

    prompt_ids = prompt_ids_for(context)
    exceso = len(prompt_ids) + len(answer_ids) - max_length
    if exceso > 0 and context:
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        keep = max(0, len(ctx_ids) - exceso - 8)
        recorte = tokenizer.decode(ctx_ids[:keep])
        recorte = recorte[: recorte.rfind("\n")] if "\n" in recorte else recorte
        prompt_ids = prompt_ids_for(recorte.rstrip() + "\n[…]")
    ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": ids, "labels": labels}


def collate(examples: list[dict], pad_id: int) -> dict:
    L = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), L), pad_id)
    labels = torch.full((len(examples), L), -100)
    mask = torch.zeros((len(examples), L), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        labels[i, :n] = torch.tensor(e["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train_sft(model, tokenizer, pairs: list[dict], settings: dict) -> dict:
    """Entrena con los pares pregunta-respuesta durante `epochs` pasadas."""
    data = [encode_pair(tokenizer, p, settings["max_length"]) for p in pairs]
    # Un ejemplo sin ningún token de respuesta daría pérdida NaN: lo descartamos.
    utiles = [d for d in data if any(l != -100 for l in d["labels"])]
    if len(utiles) < len(data):
        print(f"Aviso: se descartan {len(data) - len(utiles)} pares demasiado largos (sube max_length).")
    data = utiles
    bs = settings["batch_size"]
    steps = max(1, settings["epochs"] * len(data) // bs)
    batches = []

    def new_epoch():
        # Agrupamos pares de longitud parecida (menos relleno = entrenamiento más rápido en la GPU)
        order = list(range(len(data)))
        random.shuffle(order)
        out = []
        for i in range(0, len(order), bs * 32):
            grupo = sorted(order[i: i + bs * 32], key=lambda j: len(data[j]["input_ids"]))
            out += [grupo[k: k + bs] for k in range(0, len(grupo), bs) if len(grupo[k: k + bs]) == bs]
        random.shuffle(out)
        return out or [order[:bs]]

    def next_batch():
        nonlocal batches
        if not batches:
            batches = new_epoch()
        return collate([data[j] for j in batches.pop()], tokenizer.pad_token_id)

    print(f"Entrenando {settings['epochs']} épocas · {steps} pasos")
    return train_loop(model, next_batch, settings, steps, eval_every=max(1, steps // 5))


def ask(model, tokenizer, question: str, max_new_tokens=120, temperature=0.3, seed=0,
        system: str | None = None, context: str | None = None) -> str:
    """Hace una pregunta al modelo y devuelve solo la respuesta."""
    model.eval()
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    prompt = format_prompt(tokenizer, question, system=system, context=context)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=0.9,
                             pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.1)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
