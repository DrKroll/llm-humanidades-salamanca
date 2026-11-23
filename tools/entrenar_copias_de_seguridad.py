"""Entrena las copias de seguridad (plan B) para los cuadernos 04 y 07.

Uso (mejor con GPU, p. ej. una T4 en Colab):
    python tools/entrenar_copias_de_seguridad.py [calderon|quijote|todo] [--pasos-quijote 120] [--epocas-calderon 1]
        [--max-pares-calderon N] [--lote N]

Versión completa recomendada (GPU, ~15-20 min):
    python tools/entrenar_copias_de_seguridad.py todo --epocas-calderon 2 --pasos-quijote 300
Versión corta para CPU (~15 min, calidad reducida):
    python tools/entrenar_copias_de_seguridad.py todo --max-pares-calderon 400 --lote 2 --pasos-quijote 60

Resultado (carpeta backups/, se sube al repositorio, NO a Hugging Face):
    backups/calderon_indice_rag.json.gz   índice RAG precalculado (cuaderno 07)
    backups/calderon_lora/                adaptador LoRA para el cuaderno 07
    backups/quijote_lora/                 adaptador LoRA para el cuaderno 04
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dhllm import load_config, set_seed  # noqa: E402
from dhllm.config import resolve  # noqa: E402
from dhllm import calderon as cal  # noqa: E402
from dhllm.finetune import load_base_model, prepare_model, save_adapter, make_chunks, finetune_on_texts  # noqa: E402
from dhllm.sft import train_sft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("que", nargs="?", default="todo")
ap.add_argument("--pasos-quijote", type=int, default=120)
ap.add_argument("--epocas-calderon", type=int, default=1)
ap.add_argument("--max-pares-calderon", type=int, default=None, help="limita los pares de entrenamiento (CPU)")
ap.add_argument("--lote", type=int, default=None, help="batch_size (en CPU con poca memoria: 2)")
args = ap.parse_args()
cfg = load_config(ROOT / "config.yaml")
set_seed(cfg.seed)

if args.que in ("calderon", "todo"):
    c = dict(cfg.calderon, epochs=args.epocas_calderon)
    if args.lote:
        c["batch_size"] = args.lote
    plays = cal.load_plays(resolve(cfg, c["tei_folder"]), show=False)
    index = cal.BM25Index(cal.make_fragments(plays))
    index.save(resolve(cfg, c["backup_index"]))
    train, test = cal.prepare_training_pairs(plays, index, c, seed=cfg.seed)
    if args.max_pares_calderon:
        train = train[: args.max_pares_calderon]
    model, tok = load_base_model(c["base_model"])
    model = prepare_model(model, c["method"], c["lora_r"], c["lora_alpha"])
    t0 = time.time()
    hist = train_sft(model, tok, train, c)
    out = save_adapter(model, resolve(cfg, c["backup_adapter"]))
    (out / "entrenamiento.json").write_text(json.dumps(
        {"modelo_base": c["base_model"], "pares": len(train), "epocas": c["epochs"],
         "lote": c["batch_size"], "max_length": c["max_length"], "dispositivo": str(next(model.parameters()).device),
         "minutos": round((time.time() - t0) / 60, 1), "perdida": hist}, ensure_ascii=False, indent=1))
    del model

if args.que in ("quijote", "todo"):
    from dhllm.data import load_texts
    f = dict(cfg.finetune, max_steps=args.pasos_quijote)
    if args.lote:
        f["batch_size"] = args.lote
    model, tok = load_base_model(f["base_model"])
    texts = load_texts(cfg.data.folders, root=cfg.root)
    tr, va = make_chunks(texts, tok, block_size=f["block_size"])
    model = prepare_model(model, f["method"], f["lora_r"], f["lora_alpha"])
    t0 = time.time()
    hist = finetune_on_texts(model, tr, va, f)
    out = save_adapter(model, resolve(cfg, "backups/quijote_lora"))
    (out / "entrenamiento.json").write_text(json.dumps(
        {"modelo_base": f["base_model"], "pasos": f["max_steps"], "lote": f["batch_size"],
         "dispositivo": str(next(model.parameters()).device), "minutos": round((time.time() - t0) / 60, 1),
         "perdida": hist}, ensure_ascii=False, indent=1))
