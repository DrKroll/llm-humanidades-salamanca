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


def format_prompt(tokenizer, question: str, system: str = SYSTEM_PROMPT) -> str:
    """Construye el texto de entrada con la «plantilla de chat» del modelo."""
    if getattr(tokenizer, "chat_template", None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": question}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n### Pregunta:\n{question}\n\n### Respuesta:\n"


def encode_pair(tokenizer, pair: dict, max_length: int = 384) -> dict:
    """Codifica un par. La pérdida solo se calcula sobre la RESPUESTA (labels=-100 en la pregunta)."""
    prompt_ids = tokenizer(format_prompt(tokenizer, pair["pregunta"]), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(pair["respuesta"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
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
    bs = settings["batch_size"]
    steps = max(1, settings["epochs"] * len(data) // bs)
    order = []

    def next_batch():
        nonlocal order
        if len(order) < bs:
            order = list(range(len(data)))
            random.shuffle(order)
        batch = [data[order.pop()] for _ in range(bs)]
        return collate(batch, tokenizer.pad_token_id)

    print(f"Entrenando {settings['epochs']} épocas · {steps} pasos")
    return train_loop(model, next_batch, settings, steps, eval_every=max(1, steps // 5))


def ask(model, tokenizer, question: str, max_new_tokens=120, temperature=0.3, seed=0) -> str:
    """Hace una pregunta al modelo y devuelve solo la respuesta."""
    model.eval()
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    prompt = format_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                             temperature=max(temperature, 1e-5), top_p=0.9,
                             pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.1)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
