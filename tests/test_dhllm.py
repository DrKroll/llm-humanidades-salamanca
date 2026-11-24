"""Pruebas rápidas de las funciones del paquete (no necesitan internet)."""

import os

import torch

os.environ["DHLLM_SMOKE_TEST"] = "1"

from dhllm import load_config  # noqa: E402
from dhllm.data import corpus_stats, kwic, load_texts, words  # noqa: E402
from dhllm.pretrain import build_model, encode_corpus, load_tiny_gpt, save_tiny_gpt, train_tiny_gpt  # noqa: E402
from dhllm.sampling import filter_logits, generate  # noqa: E402
from dhllm.tokenizer import train_bpe  # noqa: E402

TEXT = ("En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo "
        "que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. ") * 60


def test_words_keep_accents():
    assert words("¿Qué dijo Sancho? ¡Ñoño!") == ["qué", "dijo", "sancho", "ñoño"]


def test_load_sample_texts():
    cfg = load_config()
    texts = load_texts(cfg.data.folders, root=cfg.root)
    assert len(texts) >= 2
    assert "Gutenberg" not in texts[0].content[:2000]
    assert corpus_stats(texts)[0]["palabras"] > 1000
    assert kwic(texts, "Sancho", max_results=2)


def test_tokenizer_roundtrip():
    tok = train_bpe(TEXT, vocab_size=300)
    s = "Un hidalgo de la Mancha, ¿verdad? ñ"
    assert tok.decode(tok.encode(s)) == s


def test_filter_logits_top_k_top_p():
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
    assert torch.isinf(filter_logits(logits, top_k=2)).sum() == 2
    assert torch.isfinite(filter_logits(logits, top_p=0.01)).sum() == 1


def test_tiny_gpt_trains_and_saves(tmp_path):
    cfg = load_config()
    tok = train_bpe(TEXT, vocab_size=300)
    tr, va = encode_corpus(tok, TEXT)
    model = build_model(tok, cfg.tiny_gpt, device="cpu")
    hist = train_tiny_gpt(model, tok, tr, va, cfg.tiny_gpt, sample_prompt=None)
    assert hist["train"][-1] < hist["train"][0]
    save_tiny_gpt(model, tok, tmp_path)
    model2, tok2 = load_tiny_gpt(tmp_path, device="cpu")
    assert isinstance(generate(model2, tok2, "En un", max_new_tokens=5, seed=0), str)


def test_calderon_parse_rag_and_qa():
    from dhllm import calderon as cal
    cfg = load_config()
    plays = cal.load_plays(f"{cfg.root}/data/calderon/tei", show=False)
    assert len(plays) == 14
    vida = next(p for p in plays if p["title"] == "La vida es sueño")
    assert vida["n_verses"] == 3319 and vida["n_acts"] == 3
    index = cal.BM25Index(cal.make_fragments(plays))
    hits = index.search("¿Quién es Cosme en «La dama duende»?", k=3)
    assert hits and all(h["obra"] == "La dama duende" for h in hits)
    assert "gracioso" in " ".join(h["texto"] for h in hits).lower()
    pairs = cal.generate_qa(plays, quotes_per_play=5)
    assert pairs and all(p["respuesta"] for p in pairs)
    train, test = cal.split_qa(pairs, test_size=5)
    assert len(test) == 5 and not set(p["pregunta"] for p in test) & set(p["pregunta"] for p in train)


def test_notebook_layout():
    """7 cuadernos obligatorios + extras; cada cuaderno con una sola celda de preparación y su insignia de Colab."""
    import json
    from pathlib import Path
    root = Path(load_config().root) / "notebooks"
    dia1 = sorted(p.name for p in (root / "dia1_python").glob("*.ipynb"))
    taller = sorted(p.name for p in (root / "taller_llm").glob("*.ipynb"))
    assert dia1 == ["P1_colab_y_textos.ipynb", "P2_listas_bucles_funciones.ipynb", "P3_leer_y_analizar_el_quijote.ipynb"]
    assert [n[:2] for n in taller] == ["01", "02", "03", "04"]
    assert (root / "extras" / "E1_instrucciones_sft.ipynb").exists()
    for p in root.glob("**/*.ipynb"):
        cells = json.loads(p.read_text(encoding="utf-8"))["cells"]
        src = ["".join(c["source"]) for c in cells]
        assert sum("Preparación: ejecuta esta celda primero" in s for s in src) <= 1, p.name
        assert str(p.relative_to(root.parent)) in src[0], f"insignia de Colab incorrecta en {p.name}"
