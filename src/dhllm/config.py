"""Lectura de la configuración (config.yaml) con valores por defecto.

Cada cuaderno tiene además una celda de «Ajustes» arriba del todo:
lo que se escribe allí tiene prioridad sobre config.yaml.
"""

from copy import deepcopy
from pathlib import Path

import yaml

from .utils import smoke_test_mode

DEFAULTS = {
    "data": {
        "folders": ["data/sample"],  # carpetas con archivos .txt
        "encoding": "utf-8",
        "lowercase": False,
    },
    "tokenizer": {
        "vocab_size": 4000,
        "min_frequency": 2,
        "save_dir": "outputs/tokenizer",
    },
    "tiny_gpt": {
        "block_size": 128,   # longitud del contexto (en tokens)
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 192,
        "dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 3e-4,
        "max_steps": 1500,
        "eval_every": 100,
        "save_dir": "outputs/tiny_gpt",
    },
    "finetune": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "method": "lora",    # "lora" o "full"
        "lora_r": 16,
        "lora_alpha": 32,
        "block_size": 256,
        "batch_size": 8,
        "learning_rate": 2e-4,
        "max_steps": 300,
        "eval_batches": 10,
        "save_dir": "outputs/finetuned",
    },
    "sft": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "qa_file": "data/qa/ejemplo_preguntas_respuestas.jsonl",
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "epochs": 3,
        "max_length": 384,
        "save_dir": "outputs/sft",
    },
    "calderon": {
        "tei_folder": "data/calderon/tei",
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "quotes_per_play": 150,      # citas por obra para generar preguntas
        "max_train_pairs": 2000,
        "test_size": 24,             # pares reservados para evaluar
        "context_fraction": 0.3,     # parte de los pares que se entrena con fragmentos (SFT + RAG)
        "rag_k": 3,                  # fragmentos que se dan al modelo
        "method": "lora",
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 4,
        "learning_rate": 0.0002,
        "epochs": 2,
        "max_length": 512,           # pares con fragmentos: ~470 tokens (se recortan si no caben)
        "save_dir": "outputs/calderon_lora",
        "backup_adapter": "backups/calderon_lora",
        "backup_index": "backups/calderon_indice_rag.json.gz",
    },
    "sampling": {
        "max_new_tokens": 120,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
    },
    "seed": 42,
}

# Valores mínimos para la prueba automática (DHLLM_SMOKE_TEST=1)
SMOKE = {
    "tokenizer": {"vocab_size": 600},
    "tiny_gpt": {"block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 64,
                 "batch_size": 8, "max_steps": 30, "eval_every": 10},
    "finetune": {"block_size": 64, "batch_size": 2, "max_steps": 2, "eval_batches": 1},
    "sft": {"batch_size": 2, "epochs": 1, "max_length": 128},
    "calderon": {"quotes_per_play": 3, "max_train_pairs": 8, "test_size": 3, "batch_size": 2,
                 "epochs": 1, "max_length": 256},
    "sampling": {"max_new_tokens": 20},
}


class Config(dict):
    """Un diccionario al que también se puede acceder con puntos: cfg.tiny_gpt.n_layer"""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(f"La configuración no tiene la clave '{key}'") from e
        return value

    def __setattr__(self, key, value):
        self[key] = value


def _merge(base: dict, extra: dict) -> dict:
    out = deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_config(d):
    if isinstance(d, dict):
        return Config({k: _to_config(v) for k, v in d.items()})
    return d


def find_repo_root(start: Path | None = None) -> Path:
    """Busca la carpeta principal del repositorio (donde está config.yaml)."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "config.yaml").exists() and (candidate / "src" / "dhllm").exists():
            return candidate
    return p


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """Carga config.yaml, añade valores por defecto y los ajustes del cuaderno.

    Ejemplo:
        cfg = load_config(overrides={"tiny_gpt": {"max_steps": 500}})
    """
    root = find_repo_root()
    path = Path(path) if path else root / "config.yaml"
    user = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    cfg = _merge(cfg, overrides or {})
    if smoke_test_mode():
        cfg = _merge(cfg, SMOKE)
    cfg["root"] = str(root)
    return _to_config(cfg)


def resolve(cfg: Config, relative_path: str) -> Path:
    """Convierte una ruta relativa del config en ruta absoluta dentro del repo."""
    p = Path(relative_path)
    return p if p.is_absolute() else Path(cfg.root) / p
