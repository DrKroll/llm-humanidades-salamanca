"""Funciones de alto nivel para los cuadernos 02, 03 y 04 del taller.

En los cuadernos solo se ven llamadas cortas como ``taller.entrenar_gpt(gpt)``. Todo el trabajo
(cargar textos, crear el modelo, el bucle de entrenamiento, guardar, cargar copias de seguridad...)
está aquí y en los demás módulos de ``dhllm``. Si tienes curiosidad, lee este archivo: cada
función explica qué hace y llama a funciones más pequeñas con nombres descriptivos.

Los ajustes de los cuadernos usan nombres en español (``pasos``, ``epocas``...). Aquí se traducen
a las claves de ``config.yaml`` (``max_steps``, ``epochs``...).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import calderon as cal
from .config import load_config, resolve
from .utils import get_device, set_seed, smoke_test_mode

# ---------------------------------------------------------------------------------------------
# 0. Utilidades comunes
# ---------------------------------------------------------------------------------------------
# Nombres en español (cuadernos) -> claves de config.yaml
_TRADUCCION = {
    "pasos": "max_steps", "epocas": "epochs", "ritmo_aprendizaje": "learning_rate", "lote": "batch_size",
    "capas": "n_layer", "cabezas": "n_head", "dimension": "n_embd", "contexto": "block_size",
    "modelo_base": "base_model", "metodo": "method", "pares": "max_train_pairs", "max_minutos": "max_minutes",
    "fragmentos_rag": "rag_k", "vocabulario": "vocab_size",
}


def _cfg(seccion: str, ajustes: dict | None):
    """Carga config.yaml y aplica los ajustes del cuaderno (con nombres en español) a una sección."""
    cambios = {_TRADUCCION.get(k, k): v for k, v in (ajustes or {}).items()}
    cfg = load_config(overrides={seccion: cambios})
    set_seed(cfg.seed)
    return cfg


def hay_gpu() -> bool:
    return get_device() == "cuda"


def comprobar_gpu() -> bool:
    """Dice si Colab nos ha dado una GPU y, si no, cómo conseguirla."""
    if hay_gpu():
        import torch
        nombre = torch.cuda.get_device_name(0)
        memoria = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU disponible: {nombre} ({memoria:.0f} GB). ¡Todo listo para entrenar!")
        return True
    print("⚠️ No hay GPU. Entrenar así sería MUY lento.\n"
          "   En Colab: menú «Entorno de ejecución» → «Cambiar tipo de entorno de ejecución» →\n"
          "   «Acelerador por hardware: GPU T4» → Guardar. Después vuelve a ejecutar el cuaderno desde arriba\n"
          "   (también la celda ⚙️ Preparación).\n"
          "   Si Colab no te da GPU (a veces se agota el cupo): trabaja con tu vecino/a o usa la copia de\n"
          "   seguridad (USAR_COPIA = True).")
    return False


@dataclass
class Modelo:
    """Un modelo de Hugging Face junto con su tokenizador (así los cuadernos solo manejan una variable)."""
    modelo: object
    tokenizer: object
    nombre: str = ""
    ajustado: bool = False          # True cuando ya lleva un adaptador LoRA entrenado o cargado
    origen: str = "modelo base"     # «entrenado por ti», «copia de seguridad»...

    def __repr__(self):
        return f"<Modelo {self.nombre} · {self.origen}>"


def _cargar_hf(nombre: str) -> Modelo:
    from .finetune import load_base_model
    modelo, tok = load_base_model(nombre)
    return Modelo(modelo, tok, nombre=nombre)


def _quitar_lora(modelo):
    """Tras un fallo: quita los adaptadores LoRA a medio entrenar y devuelve el modelo base limpio."""
    return modelo.unload() if hasattr(modelo, "unload") else modelo


def _es_falta_de_memoria(e: Exception) -> bool:
    return "out of memory" in str(e).lower()


# ---------------------------------------------------------------------------------------------
# 1. Cuaderno 02 · Tokenizador, GPT diminuto y muestreo
# ---------------------------------------------------------------------------------------------
def entrenar_tokenizador(vocabulario: int | None = None, frase: str | None = None):
    """Carga los textos de config.yaml, entrena un tokenizador BPE, lo guarda y lo prueba con una frase."""
    from .data import load_texts
    from .tokenizer import show_tokens, train_bpe
    cfg = _cfg("tokenizer", {"vocab_size": vocabulario} if vocabulario else None)
    textos = load_texts(cfg.data.folders)
    tok = train_bpe(textos, vocab_size=cfg.tokenizer.vocab_size, min_frequency=cfg.tokenizer.min_frequency)
    tok.save(cfg.tokenizer.save_dir)
    if frase:
        show_tokens(tok, frase)
    return tok


def comparar_tokenizadores(tok, frase: str):
    """Compara nuestro BPE con los tokenizadores de GPT-2 (inglés) y de Qwen2.5 (multilingüe)."""
    from .tokenizer import compare_tokenizers, show_tokens
    otros = {"nuestro BPE": tok}
    try:
        from transformers import AutoTokenizer
        otros["GPT-2 (inglés)"] = AutoTokenizer.from_pretrained("gpt2")
        otros["Qwen2.5 (multilingüe)"] = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception as e:  # sin internet
        print("No se pudieron descargar otros tokenizadores:", e)
    compare_tokenizers(frase, otros)
    if "GPT-2 (inglés)" in otros:
        show_tokens(otros["GPT-2 (inglés)"], frase)


@dataclass
class GPTDiminuto:
    """Todo lo que necesita el entrenamiento desde cero: modelo, tokenizador, datos y ajustes."""
    modelo: object
    tok: object
    train_ids: object
    val_ids: object
    ajustes: dict
    historial: dict = field(default_factory=dict)


def preparar_gpt(ajustes: dict | None = None) -> GPTDiminuto:
    """Carga textos y tokenizador (el de la parte A), convierte el corpus en números y crea un GPT vacío."""
    from .data import join_texts, load_texts
    from .pretrain import build_model, encode_corpus
    from .tokenizer import SimpleTokenizer, train_bpe
    cfg = _cfg("tiny_gpt", ajustes)
    a = dict(cfg.tiny_gpt)
    if not hay_gpu() and not smoke_test_mode() and a["max_steps"] > 300:
        print(f"ℹ️ Sin GPU: reducimos los pasos de {a['max_steps']} a 300 para que no tarde demasiado.")
        a["max_steps"] = 300
    textos = load_texts(cfg.data.folders)
    ruta = Path(cfg.tokenizer.save_dir) / "tokenizer.json"
    if ruta.exists():
        tok = SimpleTokenizer.load(ruta)
        print("Tokenizador de la parte A cargado:", tok)
    else:
        tok = train_bpe(textos, vocab_size=cfg.tokenizer.vocab_size)
    train_ids, val_ids = encode_corpus(tok, join_texts(textos))
    return GPTDiminuto(build_model(tok, a), tok, train_ids, val_ids, a)


def escribir_gpt(gpt: GPTDiminuto, inicio: str, n: int = 1, tokens: int = 60):
    """Genera `n` textos que empiezan por `inicio`."""
    from .sampling import generate
    for semilla in range(n):
        print(generate(gpt.modelo, gpt.tok, inicio, max_new_tokens=tokens, temperature=0.8, top_k=50, seed=semilla))
        if n > 1:
            print("-" * 60)


def entrenar_gpt(gpt: GPTDiminuto, inicio: str = "En un lugar de la Mancha") -> dict:
    """Entrena el GPT diminuto, dibuja la curva de pérdida, enseña cómo mejora el texto y lo guarda."""
    from .pretrain import plot_loss, save_tiny_gpt, train_tiny_gpt
    print(f"🏋️ Entrenando {gpt.ajustes['max_steps']} pasos: el modelo intenta predecir el siguiente token...")
    gpt.historial = train_tiny_gpt(gpt.modelo, gpt.tok, gpt.train_ids, gpt.val_ids, gpt.ajustes, sample_prompt=inicio)
    plot_loss(gpt.historial)
    print("\n¿Cómo ha cambiado el texto generado durante el entrenamiento?")
    for paso, texto in gpt.historial["samples"]:
        print(f"--- paso {paso} ---\n{texto}\n")
    save_tiny_gpt(gpt.modelo, gpt.tok, gpt.ajustes["save_dir"])
    return gpt.historial


def cargar_para_muestreo(usar: str = "auto"):
    """Devuelve (modelo, tokenizador): el GPT diminuto de la parte B o, si no existe, un modelo de Hugging Face."""
    from .finetune import load_base_model
    from .pretrain import load_tiny_gpt
    cfg = load_config()
    ruta = Path(cfg.tiny_gpt.save_dir)
    if usar in ("auto", "tiny_gpt") and (ruta / "model.pt").exists():
        print("Usando el GPT diminuto de la parte B")
        return load_tiny_gpt(ruta)
    return load_base_model("HuggingFaceTB/SmolLM2-135M" if usar in ("auto", "tiny_gpt") else usar)


# ---------------------------------------------------------------------------------------------
# 2. Cuaderno 03 · Ajuste fino (LoRA) con tus textos
# ---------------------------------------------------------------------------------------------
@dataclass
class DatosTexto:
    train: object
    val: object
    ajustes: dict


def preparar_ajuste_fino(ajustes: dict | None = None):
    """Carga el modelo base y tus textos, y corta los textos en trozos para entrenar. Devuelve (modelo, datos)."""
    from .data import load_texts
    from .finetune import make_chunks
    cfg = _cfg("finetune", ajustes)
    a = dict(cfg.finetune)
    m = _cargar_hf(a["base_model"])
    textos = load_texts(cfg.data.folders)
    train, val = make_chunks(textos, m.tokenizer, block_size=a["block_size"])
    return m, DatosTexto(train, val, a)


def evaluar(m: Modelo, datos: DatosTexto) -> dict:
    """Pérdida y perplejidad en los trozos de validación (texto que el modelo no usa para entrenar)."""
    from .finetune import evaluate
    r = evaluate(m.modelo, datos.val, batch_size=datos.ajustes["batch_size"], max_batches=datos.ajustes["eval_batches"])
    print(f"Validación: pérdida {r['loss']} · perplejidad {r['perplejidad']} (más bajo = mejor)")
    return r


def escribir(m: Modelo, inicios: list[str], tokens: int = 60) -> dict:
    """Continúa cada frase de `inicios` y muestra el resultado."""
    from .finetune import generations_table
    textos = generations_table(m.modelo, m.tokenizer, inicios, max_new_tokens=tokens)
    for t in textos.values():
        print(t, "\n" + "-" * 60)
    return textos


def comparar(antes: dict, despues: dict):
    from .finetune import compare_before_after
    compare_before_after(antes, despues)


def _usar_copia(m: Modelo, carpeta, motivo: str) -> Modelo:
    from .finetune import load_adapter
    print(f"🛟 {motivo}\n   → Usamos la copia de seguridad ya entrenada ({carpeta}).")
    m.modelo = load_adapter(m.modelo, resolve(load_config(), str(carpeta)))
    m.ajustado, m.origen = True, "copia de seguridad"
    return m


def _plot_historial(h: dict, titulo: str):
    import matplotlib.pyplot as plt
    if not h or not h.get("step"):
        return
    plt.plot(h["step"], h["train"], marker="o", label="entrenamiento")
    if h.get("val"):
        plt.plot(h["step"], h["val"], marker="o", label="validación")
    plt.xlabel("paso"); plt.ylabel("pérdida"); plt.legend(); plt.title(titulo); plt.show()


def ajustar_con_mis_textos(m: Modelo, datos: DatosTexto, usar_copia: bool = False) -> Modelo:
    """Ajuste fino con LoRA: añade adaptadores, entrena, dibuja la curva y guarda el resultado.

    Sin GPU (o con ``usar_copia=True``) carga el adaptador ya entrenado de ``backups/quijote_lora``.
    """
    from .finetune import finetune_on_texts, prepare_model, save_model
    a = datos.ajustes
    copia = "backups/quijote_lora"
    if usar_copia:
        return _usar_copia(m, copia, "Has elegido USAR_COPIA = True.")
    if not hay_gpu() and not smoke_test_mode():
        return _usar_copia(m, copia, "No hay GPU: en CPU este entrenamiento tardaría más de una hora.")
    try:
        m.modelo = prepare_model(m.modelo, a["method"], a["lora_r"], a["lora_alpha"])
        print(f"🏋️ Entrenando {a['max_steps']} pasos con tus textos (en una T4: 5–8 min)...")
        historial = finetune_on_texts(m.modelo, datos.train, datos.val, a)
    except Exception as e:  # noqa: BLE001
        if _es_falta_de_memoria(e):
            print("La GPU se ha quedado sin memoria. Prueba con AJUSTES = {'lote': 4} o reinicia el entorno.")
        m.modelo = _quitar_lora(m.modelo)
        return _usar_copia(m, copia, f"El entrenamiento ha fallado ({type(e).__name__}: {str(e)[:120]}).")
    _plot_historial(historial, "Ajuste fino con tus textos")
    save_model(m.modelo, m.tokenizer, a["save_dir"])
    m.ajustado, m.origen = True, "entrenado por ti"
    print("✅ ¡Listo! Tu modelo ajustado está guardado en", a["save_dir"])
    return m


# ---------------------------------------------------------------------------------------------
# 3. Cuaderno 04 · Calderón: modelo base, RAG, ajuste fino (SFT), grafo
# ---------------------------------------------------------------------------------------------
PREGUNTAS_CALDERON = [
    ("¿Quién es Cosme en «La dama duende»?", "El gracioso (criado de don Manuel)."),
    ("¿Qué personajes hablan más en «La vida es sueño»?", "Segismundo, Rosaura y Clotaldo."),
    ("¿Quién dice estos versos en «La vida es sueño»?: «Hipogrifo violento, / que corriste parejas con el viento»",
     "Rosaura, al comienzo de la jornada 1 (¡en esta edición: «Hipogrifo triste»!)."),
    ("¿Quién es Pedro Crespo en «El alcalde de Zalamea»?", "Un labrador (rico) de Zalamea, padre de Isabel."),
    ("¿Cuántas jornadas tiene «Eco y Narciso»?", "Tres."),
    ("¿Por qué está encerrado Segismundo en una torre?",
     "Su padre, el rey Basilio, lo encierra por un horóscopo que anunciaba que sería un rey tirano."),
]

PREGUNTAS_GRAFO = [
    "¿Quién es el Capitán en «El alcalde de Zalamea»?",
    "¿Quién es Cosme en «La dama duende»?",
    "¿Con quién habla más Segismundo en «La vida es sueño»?",
    "¿Quién dice «Al rey la hacienda y la vida se ha de dar» en «El alcalde de Zalamea»?",
]


@dataclass
class DatosCalderon:
    """Las obras, el buscador (índice BM25) y los pares pregunta-respuesta para entrenar y evaluar."""
    obras: list
    indice: object
    pares_train: list
    pares_test: list
    ajustes: dict
    grafos: dict = field(default_factory=dict)

    def obra(self, titulo: str) -> dict:
        t = cal._fold(titulo)
        for o in self.obras:
            if t in cal._fold(o["title"]):
                return o
        raise KeyError(f"No encuentro la obra «{titulo}». Obras: " + ", ".join(o["title"] for o in self.obras))


def _filtrar_obras(obras: list, cuales) -> list:
    if cuales in (None, "todas", "all"):
        return obras
    cuales = [cuales] if isinstance(cuales, str) else list(cuales)
    elegidas = [o for o in obras if any(cal._fold(c) in cal._fold(o["title"]) for c in cuales)]
    if not elegidas:
        raise ValueError("Ninguna obra coincide con " + repr(cuales))
    return elegidas


def preparar_datos(ajustes: dict | None = None) -> DatosCalderon:
    """Lee las obras (TEI), construye el buscador y genera los pares pregunta-respuesta.

    `ajustes["obras"]` puede ser "todas" o una lista de títulos (o trozos de título).
    """
    ajustes = dict(ajustes or {})
    cuales = ajustes.pop("obras", "todas")
    cfg = _cfg("calderon", ajustes)
    a = dict(cfg.calderon)
    obras = _filtrar_obras(cal.load_plays(resolve(cfg, a["tei_folder"])), cuales)
    print("\n🔎 Construyendo el buscador (índice BM25)...")
    indice = cal.BM25Index(cal.make_fragments(obras))
    print(f"   {len(indice.fragments)} fragmentos de ~12 versos + una ficha por obra.")
    print("\n📝 Generando preguntas y respuestas automáticas a partir del TEI...")
    train, test = cal.prepare_training_pairs(obras, indice, a, seed=cfg.seed)
    print("\nEjemplo de par de entrenamiento:\n   ❓", train[0]["pregunta"], "\n   💬", train[0]["respuesta"])
    return DatosCalderon(obras, indice, train, test, a)


def ver_obra(datos: DatosCalderon, titulo: str = "La vida es sueño", caracteres: int = 800):
    """Muestra la ficha de una obra y su comienzo (tal como sale del TEI)."""
    o = datos.obra(titulo)
    print(cal.play_card(o), "\n")
    print(cal.play_to_text(o)[:caracteres])


def cargar_modelo(datos: DatosCalderon | None = None, nombre: str | None = None) -> Modelo:
    """Carga el modelo base (por defecto Qwen2.5-0.5B-Instruct, ver config.yaml)."""
    nombre = nombre or (datos.ajustes["base_model"] if datos else load_config().calderon.base_model)
    return _cargar_hf(nombre)


def mostrar_busqueda(datos: DatosCalderon, pregunta: str, k: int | None = None):
    """Enseña qué fragmentos encuentra el buscador para una pregunta."""
    cal.show_fragments(datos.indice.search(pregunta, k=k or datos.ajustes["rag_k"]))


def preguntar(m: Modelo, pregunta: str, datos: DatosCalderon | None = None, sin_ajuste: bool = False,
              mostrar: bool = True, tokens: int = 80) -> str:
    """Hace una pregunta. Con `datos`, primero busca versos (RAG). Con `sin_ajuste=True` usa el modelo base."""
    import contextlib
    from .sft import ask
    apagar = m.modelo.disable_adapter() if (sin_ajuste and m.ajustado) else contextlib.nullcontext()
    with apagar:
        if datos is None:
            r = ask(m.modelo, m.tokenizer, pregunta, system=cal.SYSTEM_CALDERON, max_new_tokens=tokens, temperature=0)
        else:
            r = cal.ask_with_rag(m.modelo, m.tokenizer, pregunta, datos.indice, k=datos.ajustes["rag_k"],
                                 max_new_tokens=tokens)[0]
    if mostrar:
        print(f"❓ {pregunta}\n💬 {r}\n")
    return r


def responder(m: Modelo, preguntas, datos: DatosCalderon | None = None, sin_ajuste: bool = False,
              mostrar: bool = True) -> dict:
    """Responde a una lista de preguntas (o de pares (pregunta, respuesta esperada))."""
    textos = [p[0] if isinstance(p, (tuple, list)) else p for p in preguntas]
    return {p: preguntar(m, p, datos, sin_ajuste=sin_ajuste, mostrar=mostrar) for p in textos}


def tabla_comparativa(preguntas, **columnas):
    """Tabla: una fila por pregunta; columnas = respuesta esperada + cada estrategia (A=..., B=...)."""
    nombres = {"A": "A · base", "B": "B · RAG", "C": "C · SFT", "D": "D · SFT+RAG", "E": "E · grafo + versos"}
    filas = []
    for p in preguntas:
        q, esperada = (p[0], p[1]) if isinstance(p, (tuple, list)) else (p, "")
        fila = {"pregunta": q, "esperada": esperada}
        for k, col in columnas.items():
            fila[nombres.get(k, k)] = col.get(q, "")
        filas.append(fila)
    cal.comparison_table(filas)


def entrenar_calderon(m: Modelo, datos: DatosCalderon, usar_copia: bool = False) -> Modelo:
    """Ajuste fino (SFT con LoRA) con los pares pregunta-respuesta de las obras.

    - Muestra una barra de progreso y para solo al llegar a ``max_minutos`` (por defecto 10).
    - Si no hay GPU, si falla o si ``usar_copia=True``: carga la copia de seguridad de ``backups/``.
    - Si la GPU se queda sin memoria, vuelve a intentarlo con la mitad de lote.
    """
    from .finetune import prepare_model, save_adapter
    from .sft import train_sft
    a = dict(datos.ajustes)
    copia = a["backup_adapter"]
    if usar_copia:
        return _usar_copia(m, copia, "Has elegido USAR_COPIA = True.")
    if not hay_gpu() and not smoke_test_mode():
        return _usar_copia(m, copia, "No hay GPU: en CPU este entrenamiento tardaría horas.")
    pasos = max(1, a["epochs"] * len(datos.pares_train) // a["batch_size"])
    print(f"🏋️ Ajuste fino con {len(datos.pares_train)} pares de {len(datos.obras)} obras · "
          f"{a['epochs']} época(s) · {pasos} pasos · máximo {a.get('max_minutes', '—')} min.\n"
          "   El modelo ve cada pregunta y aprende a escribir la respuesta (la pérdida debe bajar).")
    m.modelo = prepare_model(m.modelo, a["method"], a["lora_r"], a["lora_alpha"])
    t0 = time.time()
    for intento in range(2):
        try:
            historial = train_sft(m.modelo, m.tokenizer, datos.pares_train, a)
            break
        except Exception as e:  # noqa: BLE001
            if _es_falta_de_memoria(e) and intento == 0 and a["batch_size"] > 1:
                import torch
                torch.cuda.empty_cache()
                a["batch_size"] //= 2
                print(f"La GPU se ha quedado sin memoria: repetimos con lote {a['batch_size']}.")
                continue
            m.modelo = _quitar_lora(m.modelo)
            return _usar_copia(m, copia, f"El entrenamiento ha fallado ({type(e).__name__}: {str(e)[:120]}).")
    _plot_historial(historial, "Ajuste fino con preguntas sobre Calderón")
    save_adapter(m.modelo, a["save_dir"])
    m.ajustado, m.origen = True, "entrenado por ti"
    print(f"✅ ¡Listo en {(time.time() - t0) / 60:.1f} min! Ahora el modelo conoce el formato de las respuestas "
          f"(y algunos datos de las {len(datos.obras)} obras).")
    return m


def evaluar_preguntas_nuevas(m: Modelo, datos: DatosCalderon, n: int = 6):
    """Compara base / SFT / SFT+RAG con pares que el modelo NO vio al entrenar."""
    filas = []
    for par in datos.pares_test[:n]:
        q = par["pregunta"]
        filas.append({"pregunta": q, "esperada": par["respuesta"],
                      "A · base": preguntar(m, q, sin_ajuste=True, mostrar=False, tokens=50),
                      "C · SFT": preguntar(m, q, mostrar=False, tokens=50),
                      "D · SFT+RAG": preguntar(m, q, datos, mostrar=False, tokens=60)})
    cal.comparison_table(filas)


# --- ⭐ grafo de personajes -------------------------------------------------------------------
def grafo(datos: DatosCalderon, titulo: str = "La vida es sueño", mostrar: bool = True):
    """Red de personajes de una obra: papel (reparto), versos y con quién dialoga."""
    if not datos.grafos:
        datos.grafos = cal.character_graphs(datos.obras)
    G = datos.grafos[datos.obra(titulo)["title"]]
    if mostrar:
        for personaje, d in sorted(G.nodes(data=True), key=lambda x: -x[1]["versos"]):
            print(f"{personaje:<14} papel: {d['rol'] or '—':<12} versos: {d['versos']}")
    return G


def dibujar_grafo(G):
    cal.draw_graph(G)


def personajes_centrales(G, n: int = 5):
    tabla = cal.central_characters(G, n=n)
    print("\nMás central por réplicas:", tabla[0]["personaje"],
          "· más versos:", max(G.nodes, key=lambda p: G.nodes[p]["versos"]))
    return tabla


def comparar_con_grafo(m: Modelo, datos: DatosCalderon, preguntas=None):
    """Columna D (solo versos) frente a E (datos del grafo + versos)."""
    if not datos.grafos:
        datos.grafos = cal.character_graphs(datos.obras)
    filas = []
    for q in preguntas or PREGUNTAS_GRAFO:
        k = datos.ajustes["rag_k"]
        solo_versos = cal.ask_with_rag(m.modelo, m.tokenizer, q, datos.indice, k=k, max_new_tokens=60)[0]
        r, hechos, frags = cal.ask_with_graph(m.modelo, m.tokenizer, q, datos.grafos, datos.indice, k=k, max_new_tokens=60)
        citas = "; ".join("ficha" if f["verso"] == "ficha" else f"j. {f['jornada']}, v. {f['verso']}" for f in frags)
        filas.append({"pregunta": q, "D · versos": solo_versos, "E · grafo + versos": r,
                      "datos del grafo": hechos, "versos citados": citas})
    cal.comparison_table(filas)
