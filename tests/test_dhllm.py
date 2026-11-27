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


def test_calderon_character_graph():
    """Grafo de personajes (cuaderno 04, opcional): papeles del reparto, versos y réplicas."""
    import matplotlib
    matplotlib.use("Agg")
    from dhllm import calderon as cal
    cfg = load_config()
    plays = cal.load_plays(f"{cfg.root}/data/calderon/tei", show=False)
    graphs = cal.character_graphs(plays)
    assert len(graphs) == 14
    G = graphs["La vida es sueño"]
    assert G.nodes["Basilio"]["rol"] == "rey" and G.nodes["Segismundo"]["versos"] == 876
    assert not any(cal.NOISE.search(n) for n in G.nodes)      # sin «Criado 2», «Todos»...
    assert G.has_edge("Segismundo", "Clotaldo") and G.edges["Segismundo", "Clotaldo"]["replicas"] > 10
    top = cal.central_characters(G, n=3, show=False)
    assert top[0]["personaje"] in {"Clotaldo", "Rosaura", "Segismundo"}
    zal = graphs["El alcalde de Zalamea"]
    assert zal.nodes["Crespo"]["nombre_reparto"] == "Pedro Crespo"       # «Crespo» → «Pedro Crespo»
    assert zal.nodes["Capitán"]["nombre_reparto"] == "Don Álvaro De Ataide"  # por el papel
    facts = cal.graph_facts(zal, "¿Quién es el Capitán en «El alcalde de Zalamea»?")
    assert "Don Álvaro De Ataide" in facts and "capitán" in facts
    cal.draw_graph(G)


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


# --------------------------------------------------------------- dhllm.taller (funciones de alto nivel)
def test_taller_traduce_ajustes(monkeypatch):
    monkeypatch.delenv("DHLLM_SMOKE_TEST", raising=False)
    from dhllm import taller
    cfg = taller._cfg("calderon", {"epocas": 3, "ritmo_aprendizaje": 1e-4, "max_minutos": 2})
    assert cfg.calderon.epochs == 3 and cfg.calderon.learning_rate == 1e-4 and cfg.calderon.max_minutes == 2
    assert taller.comprobar_gpu() == taller.hay_gpu()


def test_taller_preparar_datos_incluye_obras_nuevas(monkeypatch):
    monkeypatch.setenv("DHLLM_SMOKE_TEST", "1")
    from dhllm import taller
    datos = taller.preparar_datos({"obras": "todas"})
    titulos = {o["title"] for o in datos.obras}
    assert {"Eco y Narciso", "El secreto a voces"} <= titulos and len(titulos) == 14
    solo = taller.preparar_datos({"obras": ["Eco y Narciso"], "pares": 5})
    assert [o["title"] for o in solo.obras] == ["Eco y Narciso"]
    assert solo.pares_train and len(solo.indice.fragments) < len(datos.indice.fragments)
    G = taller.grafo(solo, "eco y narciso", mostrar=False)
    assert G.number_of_nodes() > 3


def test_taller_gpt_diminuto(monkeypatch, tmp_path):
    monkeypatch.setenv("DHLLM_SMOKE_TEST", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from dhllm import taller
    tok = taller.entrenar_tokenizador(600)
    assert tok.vocab_size <= 600
    gpt = taller.preparar_gpt({"pasos": 4, "save_dir": str(tmp_path / "gpt")})
    hist = taller.entrenar_gpt(gpt, "En un lugar")
    assert hist["step"][-1] == gpt.ajustes["max_steps"] and (tmp_path / "gpt" / "model.pt").exists()


def test_taller_calderon_entrena_y_copia_si_falla(monkeypatch, tmp_path):
    monkeypatch.setenv("DHLLM_SMOKE_TEST", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    from dhllm import taller, sft, finetune
    datos = taller.preparar_datos({"obras": ["La dama duende"], "save_dir": str(tmp_path / "lora")})
    m = taller.cargar_modelo(datos)
    assert isinstance(taller.preguntar(m, "¿Quién es Cosme?", datos, mostrar=False, tokens=5), str)
    m = taller.entrenar_calderon(m, datos)                      # entrena de verdad (pocos pasos)
    assert m.origen == "entrenado por ti" and (tmp_path / "lora" / "adapter_config.json").exists()
    assert isinstance(taller.preguntar(m, "¿Quién es Cosme?", sin_ajuste=True, mostrar=False, tokens=5), str)

    # Si el entrenamiento falla, se usa la copia de seguridad (aquí simulada)
    m2 = taller.cargar_modelo(datos)
    monkeypatch.setattr(sft, "train_sft", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fallo simulado")))
    monkeypatch.setattr(finetune, "load_adapter", lambda modelo, carpeta: modelo)
    m2 = taller.entrenar_calderon(m2, datos)
    assert m2.origen == "copia de seguridad"
