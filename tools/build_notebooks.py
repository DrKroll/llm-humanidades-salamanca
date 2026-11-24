"""Genera todos los cuadernos del taller (día 2), los extras y los de Python (día 1).

Los bloques N00…N07 son las secciones de los antiguos cuadernos; al final se combinan en
4 cuadernos obligatorios (01–04) y los extras (notebooks/extras/).
"""

from nbtools import SETUP_CODE, code, md, opcional, part, write_notebook

import build_python_notebooks

TALLER = "notebooks/taller_llm"

# ---------------------------------------------------------------- 00
N00 = [
    md("""
# 00 · Preparación del entorno

**Objetivo:** comprobar que todo funciona antes de empezar a entrenar modelos.

En este taller (4 cuadernos) vas a:
1. **01** · preparar el entorno y explorar tus propios textos,
2. **02** · crear un *tokenizador*, entrenar un GPT diminuto **desde cero** y jugar con la generación (temperatura, top-k, top-p),
3. **03** · ajustar (*fine-tuning*) un modelo pequeño ya existente con tus textos,
4. **04** · hacer preguntas sobre el teatro de Calderón: modelo base, RAG y ajuste fino.

Para quien vaya rápido o quiera seguir en casa: carpeta `notebooks/extras/`.

### Antes de empezar en Google Colab
1. Menú **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4** (gratuita).
2. Ejecuta las celdas en orden con el botón ▶ o con **Mayús + Intro**.
3. Si algo falla: **Entorno de ejecución → Reiniciar sesión** y vuelve a ejecutar desde arriba.

> 💡 No hace falta entender todo el código. Las celdas llaman a funciones del paquete `dhllm`
> (carpeta `src/dhllm/`). Lo importante son los **ajustes** y los **resultados**.
"""),
    code(SETUP_CODE),
    md("## 1. ¿Qué ordenador tenemos?\nLa siguiente celda muestra las versiones instaladas y si hay una GPU disponible."),
    code("""
from dhllm import describe_environment
info = describe_environment()
"""),
    md("""
## 2. La configuración

Todos los ajustes del taller están en el archivo **`config.yaml`** (en Colab: icono de carpeta 📁 a la izquierda).
Además, cada cuaderno tiene arriba una celda de **Ajustes** donde puedes cambiar valores sin tocar ningún archivo.
"""),
    code("""
from dhllm import load_config
cfg = load_config()
print("Carpetas de textos:", cfg.data.folders)
print("Tamaño del vocabulario:", cfg.tokenizer.vocab_size)
print("Modelo base para el ajuste fino:", cfg.finetune.base_model)
"""),
    md("## 3. Una prueba rápida\nSi esta celda muestra un número y un mensaje, ¡todo está listo!"),
    code("""
import torch
x = torch.rand(3, 3)
print("Suma de una matriz aleatoria:", round(x.sum().item(), 3))
print("✅ Todo listo. Continúa con la parte B.")
"""),
    md("""
### ✅ Comprobación
Si estás en Colab sin GPU, cambia el tipo de entorno a **T4** y vuelve a ejecutar la celda del punto 1.
"""),
]

# ---------------------------------------------------------------- 01
N01 = [
    md("""
# 01 · Explora tus textos

**Objetivo:** cargar un corpus (colección de textos) y conocerlo un poco antes de entrenar.

Un modelo de lenguaje solo puede aprender lo que hay en sus datos. Por eso, en Humanidades Digitales
lo primero es siempre **mirar el corpus**: ¿cuánto texto hay?, ¿qué palabras aparecen?, ¿hay «ruido»
(números de página, cabeceras, errores de OCR)?

Usaremos como ejemplo la *Primera parte del Quijote* y el *Lazarillo de Tormes* (dominio público,
Project Gutenberg). Puedes usar tus propios textos `.txt`.
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes\nCambia aquí la carpeta con tus textos. `data/sample` contiene los textos de ejemplo."),
    code("""
CARPETAS = ["data/sample"]     # p. ej. ["data/mis_textos"]
PALABRA_CLAVE = "Dulcinea"     # para buscar en contexto (KWIC)
"""),
    md("""
### (Opcional) Sube tus propios textos
En Colab, ejecuta la celda siguiente con `SUBIR = True` para elegir archivos `.txt` de tu ordenador.
Se guardarán en `data/mis_textos/`. Después cambia `CARPETAS = ["data/mis_textos"]` arriba.
"""),
    code("""
SUBIR = False
if SUBIR:
    from dhllm.data import upload_texts
    upload_texts("data/mis_textos")
"""),
    md("## 1. Cargar los textos"),
    code("""
from dhllm.data import load_texts
textos = load_texts(CARPETAS)
textos
"""),
    md("Veamos el principio del primer texto:"),
    code("""
print(textos[0].content[:800])
"""),
    md("""
## 2. Estadísticas básicas
- **riqueza léxica** = palabras distintas ÷ palabras totales (*type-token ratio*). Depende mucho de la longitud del texto.
"""),
    code("""
from dhllm.data import corpus_stats
tabla = corpus_stats(textos)
"""),
    md("## 3. Palabras más frecuentes\nQuitamos las *palabras vacías* (de, la, que, y...) para ver las palabras con contenido."),
    code("""
from dhllm.data import top_words
top_words(textos, n=25);
"""),
    md("## 4. Palabra clave en contexto (KWIC)\nUna herramienta clásica de la lingüística de corpus."),
    code("""
from dhllm.data import kwic
kwic(textos, PALABRA_CLAVE, width=45, max_results=8);
"""),
    md("""
## 5. ¿Cuánto texto necesita un modelo?
Para comparar: GPT-3 se entrenó con unos 300 000 millones de tokens; los modelos actuales con más de
10 billones. Nuestro corpus de ejemplo tiene ~1 millón de caracteres. Por eso nuestros modelos serán
**miniaturas**: aprenderán algo del estilo, pero no «sabrán» cosas.
"""),
    code("""
from dhllm.data import join_texts
corpus = join_texts(textos)
print(f"Caracteres: {len(corpus):,}".replace(",", "."))
print(f"Palabras aprox.: {len(corpus.split()):,}".replace(",", "."))
"""),
    md("""
## 🖐️ Tu turno
1. Busca otras palabras con `kwic` (por ejemplo *molinos*, *honra*, *pan*). ¿Qué te dice el contexto?
2. Cambia `n=25` por `n=50` en `top_words`. ¿Aparecen nombres propios?
3. Sube un texto propio y compara sus estadísticas con las del Quijote.
4. **Para pensar:** ¿qué «ruido» ves en tus textos que un modelo aprendería a imitar?
"""),
]

# ---------------------------------------------------------------- 02
N02 = [
    md("""
# 02 · El tokenizador: de texto a números

**Objetivo:** entrenar un tokenizador **BPE** (*Byte Pair Encoding*) con tus textos y ver cómo trocea el lenguaje.

Un ordenador no entiende letras, solo números. El **tokenizador** divide el texto en trozos
(**tokens**) y asigna un número a cada uno. Opciones:

| Unidad | Ejemplo con «caballero» | Problema |
|---|---|---|
| letras | c·a·b·a·l·l·e·r·o | secuencias muy largas |
| palabras | caballero | vocabulario enorme, palabras nuevas desconocidas |
| **subpalabras (BPE)** | caball·ero | ✅ el compromiso que usan GPT, Llama, etc. |

**Cómo funciona BPE:** empieza con caracteres sueltos y une una y otra vez el par más frecuente
(«q»+«u» → «qu», «qu»+«e» → «que»...) hasta alcanzar el tamaño de vocabulario deseado.
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config
cfg = load_config()

CARPETAS = cfg.data.folders
VOCABULARIO = cfg.tokenizer.vocab_size   # prueba 500, 2000, 8000
FRASE = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo."
"""),
    md("## 1. Cargar textos y entrenar el tokenizador\nTarda unos segundos."),
    code("""
from dhllm.data import load_texts
from dhllm.tokenizer import train_bpe
textos = load_texts(CARPETAS)
tok = train_bpe(textos, vocab_size=VOCABULARIO, min_frequency=cfg.tokenizer.min_frequency)
"""),
    md("## 2. Ver los tokens\nCada color es un token. El espacio suele ir pegado al principio de la palabra."),
    code("""
from dhllm.tokenizer import show_tokens
show_tokens(tok, FRASE);
"""),
    md("Y estos son los números que «ve» el modelo:"),
    code("""
ids = tok.encode(FRASE)
print(ids)
print("De vuelta a texto:", tok.decode(ids))
"""),
    md("## 3. ¿Qué ha aprendido el tokenizador?\nLos tokens más largos del vocabulario suelen ser palabras frecuentes de tu corpus."),
    code("""
from dhllm.tokenizer import show_vocabulary_sample
show_vocabulary_sample(tok, n=40);
"""),
    opcional("""
## 4. Comparar con un tokenizador «de verdad»
Comparamos con el tokenizador de GPT-2 (entrenado sobre todo con inglés) y con el de SmolLM2.
**Menos tokens por palabra = más eficiente** para ese idioma. (Necesita conexión a internet.)
"""),
    code("""
from dhllm.tokenizer import compare_tokenizers
otros = {"nuestro BPE": tok}
try:
    from transformers import AutoTokenizer
    otros["GPT-2 (inglés)"] = AutoTokenizer.from_pretrained("gpt2")
    otros["SmolLM2"] = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
except Exception as e:
    print("No se pudieron descargar otros tokenizadores:", e)
compare_tokenizers(FRASE, otros);
"""),
    code("""
if "GPT-2 (inglés)" in otros:
    show_tokens(otros["GPT-2 (inglés)"], FRASE);
"""),
    md("## 5. Guardar el tokenizador\nLo usaremos en la parte B."),
    code("""
tok.save(cfg.tokenizer.save_dir)
"""),
    md("""
## 🖐️ Tu turno
1. Prueba `VOCABULARIO = 300` y `VOCABULARIO = 10000`. ¿Qué cambia en los colores de la frase?
2. Tokeniza una frase en otro idioma (latín, inglés, catalán...) o con ortografía antigua. ¿Qué pasa?
3. **Para pensar:** si un idioma necesita el doble de tokens que el inglés, ¿qué consecuencias tiene
   en el coste y la calidad de los grandes modelos comerciales?
"""),
]

# ---------------------------------------------------------------- 03
N03 = [
    md("""
# 03 · Preentrenar un GPT diminuto desde cero

**Objetivo:** entrenar tu propio modelo de lenguaje (¡con pesos aleatorios al principio!) usando solo tus textos.

La tarea es siempre la misma: **predecir el siguiente token**. El modelo ve un trozo de texto y
tiene que adivinar qué viene después. Al principio adivina al azar; poco a poco mejora.

- La **pérdida** (*loss*) mide cuánto se equivoca. Debe **bajar** durante el entrenamiento.
- **Entrenamiento** vs. **validación**: la validación es texto que el modelo no ve al entrenar.
  Si la pérdida de entrenamiento baja pero la de validación sube, el modelo está **memorizando** (*sobreajuste*).

Nuestro modelo tiene ~2,6 millones de parámetros. GPT-3 tiene 175 000 millones. 🙂
Inspirado en [nanoGPT](https://github.com/karpathy/nanoGPT) y [minimind](https://github.com/jingyaogong/minimind).

⏱️ Con GPU T4: unos 2–4 minutos. En CPU: usa `"max_steps": 300` en los ajustes (≈ 1,5 min, resultados más pobres).
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config, set_seed, get_device
cfg = load_config(overrides={
    "tiny_gpt": {
        # "max_steps": 300,   # ← quita el «#» para entrenar menos pasos (p. ej. en CPU)
        # "n_layer": 6,       # más capas = modelo más grande
    }
})
ajustes = cfg.tiny_gpt
set_seed(cfg.seed)
print("Dispositivo:", get_device())
print(dict(ajustes))
INICIO = "En un lugar de la Mancha"    # texto con el que el modelo empezará a escribir
"""),
    md("## 1. Textos y tokenizador\nSe carga el tokenizador de la parte A (si no existe, se entrena ahora)."),
    code("""
from pathlib import Path
from dhllm.data import load_texts, join_texts
from dhllm.tokenizer import SimpleTokenizer, train_bpe

textos = load_texts(cfg.data.folders)
ruta_tok = Path(cfg.tokenizer.save_dir) / "tokenizer.json"
if ruta_tok.exists():
    tok = SimpleTokenizer.load(ruta_tok)
    print("Tokenizador cargado:", tok)
else:
    tok = train_bpe(textos, vocab_size=cfg.tokenizer.vocab_size)
"""),
    md("## 2. Convertir el corpus en números"),
    code("""
from dhllm.pretrain import encode_corpus
train_ids, val_ids = encode_corpus(tok, join_texts(textos))
"""),
    md("## 3. Crear el modelo (vacío)"),
    code("""
from dhllm.pretrain import build_model
modelo = build_model(tok, ajustes)
"""),
    md("Antes de entrenar, el modelo escribe sin sentido:"),
    code("""
from dhllm.sampling import generate
print(generate(modelo, tok, INICIO, max_new_tokens=40, seed=0))
"""),
    md("## 4. ¡Entrenar!\nVerás la pérdida cada cierto número de pasos y, de vez en cuando, una muestra de texto."),
    code("""
from dhllm.pretrain import train_tiny_gpt, plot_loss
historial = train_tiny_gpt(modelo, tok, train_ids, val_ids, ajustes, sample_prompt=INICIO)
plot_loss(historial)
"""),
    md("""
## 5. ¿Cómo ha evolucionado el texto generado?
**Interpretación de la pérdida:** una pérdida de ~8,3 significa «adivinar al azar» entre 4000 tokens
(ln 4000 ≈ 8,3). Si baja a ~4, el modelo duda en promedio entre unas e⁴ ≈ 55 opciones.
"""),
    code("""
for paso, texto in historial["samples"]:
    print(f"--- paso {paso} ---\\n{texto}\\n")
"""),
    md("## 6. Generar texto nuevo"),
    code("""
for semilla in range(3):
    print(generate(modelo, tok, INICIO, max_new_tokens=80, temperature=0.8, top_k=50, seed=semilla))
    print("-" * 60)
"""),
    md("## 7. Guardar el modelo\nLo usaremos en la parte C. (En Colab, descarga la carpeta `outputs/` si quieres conservarlo.)"),
    code("""
from dhllm.pretrain import save_tiny_gpt
save_tiny_gpt(modelo, tok, ajustes.save_dir)
"""),
    md("""
## 🖐️ Tu turno
1. Cambia `INICIO` por otra frase («Sancho dijo», «Yo por bien tengo»...).

⭐ **Opcional (en casa o si vas rápido; cada prueba vuelve a entrenar):**

2. Entrena con más pasos (`"max_steps": 3000`). ¿Mejora la validación o empieza a subir?
3. Haz el modelo más grande (`"n_layer": 6, "n_embd": 256`) o más pequeño (`"n_layer": 2`). Compara curvas.
4. Entrena **solo** con el Lazarillo: en los ajustes añade `"data": {"folders": ["data/sample/lazarillo_de_tormes.txt"]}` dentro de `overrides`. ¿Qué pasa con tan poco texto?
5. **Para pensar:** ¿«entiende» este modelo el Quijote? ¿Qué ha aprendido exactamente?
"""),
]

# ---------------------------------------------------------------- 04
N04 = [
    md("""
# 03 · Ajuste fino (*fine-tuning*) de un modelo preentrenado con tus textos

**Objetivo:** tomar un modelo pequeño que ya «sabe» lenguaje y adaptarlo al estilo de tus textos.

Entrenar desde cero (cuaderno 02) necesita muchísimo texto. Es mucho más eficaz partir de un modelo
**preentrenado** (que ya ha leído miles de millones de palabras) y **ajustarlo** con nuestro corpus.

- **Modelo base:** por defecto `Qwen/Qwen2.5-0.5B-Instruct` (494 millones de parámetros, buen español).
  `HuggingFaceTB/SmolLM2-135M` es más rápido, pero apenas sabe español.
- **LoRA** (*Low-Rank Adaptation*): en lugar de cambiar todos los parámetros, añadimos unas pocas
  matrices pequeñas y solo entrenamos esas (≈1–5 % del total). Ahorra memoria y tiempo y el resultado
  ocupa solo unos MB.

⏱️ **≈ 25 min** (martes 15:50–16:15). Con GPU T4: 300 pasos ≈ 5–8 min (Qwen2.5-0.5B). **En CPU es muy lento** (varios segundos por paso): usa la GPU de Colab.
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config, set_seed
cfg = load_config(overrides={
    "finetune": {
        # "base_model": "HuggingFaceTB/SmolLM2-135M",   # más rápido, casi sin español
        # "method": "full",                     # entrenar todos los parámetros
        # "max_steps": 50,                      # menos pasos (p. ej. en CPU)
    }
})
ajustes = cfg.finetune
set_seed(cfg.seed)
print(dict(ajustes))

ENTRENAR = True   # False = usar el adaptador ya entrenado de backups/quijote_lora (plan B, solo con Qwen2.5-0.5B-Instruct)

INICIOS = [                     # frases para comparar antes/después
    "En un lugar de la Mancha",
    "Sancho Panza le dijo a su amo",
    "Yo, señor, nací en",
]
"""),
    md("## 1. Cargar el modelo base y los textos"),
    code("""
from dhllm.finetune import load_base_model, make_chunks
from dhllm.data import load_texts
modelo, tokenizer = load_base_model(ajustes.base_model)
textos = load_texts(cfg.data.folders)
train_chunks, val_chunks = make_chunks(textos, tokenizer, block_size=ajustes.block_size)
"""),
    md("## 2. ¿Qué escribe el modelo ANTES del ajuste?"),
    code("""
from dhllm.finetune import generations_table, evaluate
antes = generations_table(modelo, tokenizer, INICIOS, max_new_tokens=60)
for p, t in antes.items():
    print(t, "\\n" + "-" * 60)
print("Validación antes:", evaluate(modelo, val_chunks, batch_size=ajustes.batch_size, max_batches=ajustes.eval_batches))
"""),
    md("""
## 3. Preparar LoRA y entrenar
La **perplejidad** indica, más o menos, entre cuántas palabras «duda» el modelo. Debería bajar.
"""),
    code("""
from dhllm.finetune import prepare_model, finetune_on_texts, load_adapter
if ENTRENAR:
    modelo = prepare_model(modelo, ajustes.method, ajustes.lora_r, ajustes.lora_alpha)
    historial = finetune_on_texts(modelo, train_chunks, val_chunks, ajustes)
else:
    modelo = load_adapter(modelo, "backups/quijote_lora")
    historial = None
"""),
    code("""
import matplotlib.pyplot as plt
if historial:
    plt.plot(historial["step"], historial["train"], marker="o", label="entrenamiento")
    plt.plot(historial["step"], historial["val"], marker="o", label="validación")
    plt.xlabel("paso"); plt.ylabel("pérdida"); plt.legend(); plt.title("Ajuste fino"); plt.show()
print("Validación después:", evaluate(modelo, val_chunks, batch_size=ajustes.batch_size, max_batches=ajustes.eval_batches))
"""),
    md("## 4. Comparar ANTES y DESPUÉS"),
    code("""
from dhllm.finetune import compare_before_after
despues = generations_table(modelo, tokenizer, INICIOS, max_new_tokens=60)
compare_before_after(antes, despues)
"""),
    md("## 5. Guardar\nCon LoRA solo se guardan los adaptadores (unos pocos MB)."),
    code("""
from dhllm.finetune import save_model
save_model(modelo, tokenizer, ajustes.save_dir)
"""),
    md("""
## 🖐️ Tu turno
1. Añade tus propios `INICIOS`.
2. Cambia el modelo base a `HuggingFaceTB/SmolLM2-135M`. ¿Qué diferencia notas en el español?
3. Entrena con tus propios textos (carpeta `data/mis_textos`). ¿Reconoces «tu» estilo?
4. ⭐ (Opcional, en casa) Compara `"method": "lora"` y `"method": "full"`: tiempo, memoria y resultado.
5. **Para pensar:** si ajustas un modelo con las cartas de una persona, ¿de quién es el texto que genera?

> 🚀 **Para ir más lejos:** el extra `notebooks/extras/E1_instrucciones_sft.ipynb` enseña a un modelo a
> responder preguntas (SFT). [Unsloth](https://github.com/unslothai/unsloth) ofrece cuadernos de Colab
> muy optimizados para ajustar modelos más grandes (1–8 mil millones de parámetros) con QLoRA en una T4.
"""),
]

# ---------------------------------------------------------------- 05
N05 = [
    md("""
# E1 · (Extra) Ajuste por instrucciones: de «modelo base» a «asistente»

**Objetivo:** ver cómo un modelo aprende a **responder preguntas** en lugar de solo continuar texto.

Un modelo base solo continúa el texto. Si le preguntas «¿Quién es Sancho?», puede que siga con
otra pregunta. Para crear asistentes como ChatGPT se hace un **ajuste supervisado (SFT)** con miles de
pares *instrucción → respuesta*. Aquí lo hacemos en miniatura con 30 pares sobre el Quijote y el Lazarillo
(archivo `data/qa/ejemplo_preguntas_respuestas.jsonl`).

La **pérdida solo se calcula sobre la respuesta**: el modelo no tiene que aprender a escribir preguntas.

> ⚠️ Con tan pocos ejemplos, el modelo aprende sobre todo el **formato** y el **tono**, no hechos fiables.
> Es un buen momento para hablar de **alucinaciones**. El cuaderno 04 (Calderón) hace lo mismo a mayor
> escala, con ~2000 pares generados automáticamente y RAG.

⏱️ Con GPU T4: unos 5 min.
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config, set_seed
cfg = load_config(overrides={
    "sft": {
        # "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",  # más rápido, pero apenas sabe español
        # "epochs": 5,
    }
})
ajustes = cfg.sft
set_seed(cfg.seed)
PREGUNTAS = [
    "¿Cómo se llama el caballo de don Quijote?",
    "¿Quién fue el primer amo de Lázaro?",
    "¿Qué es la novela picaresca?",
]
"""),
    md("## 1. Datos: pares pregunta-respuesta"),
    code("""
from dhllm.sft import load_qa
pares = load_qa(ajustes.qa_file)
pares[:3]
"""),
    md("## 2. Modelo y respuestas ANTES"),
    code("""
from dhllm.finetune import load_base_model, prepare_model
from dhllm.sft import ask, format_prompt
modelo, tokenizer = load_base_model(ajustes.base_model)
print("Así ve el modelo una pregunta (plantilla de chat):\\n")
print(format_prompt(tokenizer, PREGUNTAS[0]))
antes = {p: ask(modelo, tokenizer, p) for p in PREGUNTAS}
for p, r in antes.items():
    print(f"❓ {p}\\n💬 {r}\\n")
"""),
    md("## 3. Entrenar"),
    code("""
from dhllm.sft import train_sft
modelo = prepare_model(modelo, ajustes.method, ajustes.lora_r, ajustes.lora_alpha)
historial = train_sft(modelo, tokenizer, pares, ajustes)
"""),
    md("## 4. Respuestas DESPUÉS"),
    code("""
for p in PREGUNTAS:
    print(f"❓ {p}\\n   ANTES:   {antes[p]}\\n   DESPUÉS: {ask(modelo, tokenizer, p)}\\n")
"""),
    md("Prueba también una pregunta que **no** está en los datos:"),
    code("""
print(ask(modelo, tokenizer, "¿Quién escribió La Celestina?"))
"""),
    md("""
## 🖐️ Tu turno
1. Escribe 10–20 pares pregunta-respuesta sobre **tu** corpus (mismo formato `.jsonl`) y entrena con ellos.
2. ¿Responde bien a preguntas nuevas? ¿Inventa? ¿Por qué?
3. **Para pensar:** en los grandes modelos, miles de personas escriben y valoran respuestas (RLHF/DPO).
   ¿Qué valores y qué visión del mundo entran así en el modelo?
"""),
]

# ---------------------------------------------------------------- 06
N06 = [
    md("""
# 06 · Laboratorio de muestreo: temperatura, top-k y top-p

**Objetivo:** entender cómo un modelo **elige** la siguiente palabra.

En cada paso, el modelo calcula una probabilidad para **cada** token del vocabulario. Después hay que
elegir uno:

- **Greedy** (temperatura 0): siempre el más probable → repetitivo.
- **Temperatura**: < 1 hace la distribución más «puntiaguda» (conservador); > 1 la aplana (creativo, caótico).
- **Top-k**: solo se sortea entre los *k* tokens más probables.
- **Top-p** (*nucleus sampling*): solo entre los tokens más probables que suman una probabilidad *p*.

Usamos el GPT diminuto de la parte B (si no existe, se carga SmolLM2-135M).

⏱️ ≈ 10 min. Si vas justo de tiempo, haz solo los puntos 1 y 3.
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config
cfg = load_config()
INICIO = "Dijo don Quijote"
USAR_MODELO = "auto"   # "auto", "tiny_gpt" o un nombre de Hugging Face, p. ej. "HuggingFaceTB/SmolLM2-135M"
"""),
    code("""
from pathlib import Path
from dhllm.pretrain import load_tiny_gpt
from dhllm.finetune import load_base_model

ruta = Path(cfg.tiny_gpt.save_dir)
if USAR_MODELO in ("auto", "tiny_gpt") and (ruta / "model.pt").exists():
    modelo, tok = load_tiny_gpt(ruta)
    print("Usando el GPT diminuto de la parte B")
else:
    nombre = "HuggingFaceTB/SmolLM2-135M" if USAR_MODELO in ("auto", "tiny_gpt") else USAR_MODELO
    modelo, tok = load_base_model(nombre)
"""),
    md("## 1. ¿Qué token viene después?\nEstas son las probabilidades del modelo para el siguiente token."),
    code("""
from dhllm.sampling import next_token_table
next_token_table(modelo, tok, INICIO, temperature=1.0, n=15);
"""),
    md("## 2. El efecto de la temperatura\nMisma frase, distinta temperatura. Fíjate en cómo cambian las barras."),
    code("""
for t in [0.3, 1.0, 2.0]:
    next_token_table(modelo, tok, INICIO, temperature=t, n=10)
"""),
    md("## 3. Generar con distintos ajustes"),
    code("""
from dhllm.sampling import compare_settings
compare_settings(modelo, tok, INICIO, [
    {"temperature": 0},                                # greedy
    {"temperature": 0.7, "top_k": 50, "top_p": 0.95},  # equilibrado
    {"temperature": 1.5, "top_k": None, "top_p": 1.0}, # caótico
    {"temperature": 1.0, "top_k": 5, "top_p": 1.0},    # solo 5 opciones
], max_new_tokens=cfg.sampling.max_new_tokens // 2);
"""),
    md("## 4. Tu propio experimento"),
    code("""
from dhllm.sampling import generate
print(generate(modelo, tok, INICIO,
               max_new_tokens=cfg.sampling.max_new_tokens,
               temperature=0.9,   # ← cambia
               top_k=40,          # ← cambia (None = sin límite)
               top_p=0.9,         # ← cambia (1.0 = sin límite)
               seed=1))
"""),
    md("""
## 🖐️ Tu turno
1. Encuentra los ajustes que producen el texto más «cervantino». ¿Y el más absurdo?
2. Con `temperature=0`, ¿se repite el texto? ¿Por qué?
3. Ejecuta dos veces con `seed=None`. ¿Por qué cambia el resultado?
4. **Para pensar:** un chatbot responde distinto a la misma pregunta. ¿Qué implica esto para citar
   o usar textos generados en investigación?
"""),
]


# ---------------------------------------------------------------- 07
N07 = [
    md("""
# 04 · Preguntas sobre el teatro de Calderón: modelo base, RAG y ajuste fino

**Objetivo:** hacer preguntas sobre obras de Calderón y comparar cuatro estrategias:

| | Estrategia | Qué hace |
|---|---|---|
| A | **Modelo base** | Responde con lo que «recuerda» de su entrenamiento general. |
| B | **RAG** (*Retrieval-Augmented Generation*) | Primero **busca** versos relevantes en el corpus y se los da al modelo junto con la pregunta. |
| C | **SFT** (ajuste fino con LoRA) | Entrenamos el modelo con ~2000 preguntas y respuestas generadas automáticamente a partir del TEI. |
| D | **SFT + RAG** | El modelo ajustado recibe además los fragmentos encontrados. |

**Corpus:** 14 obras en TEI-XML (8 comedias/dramas de EMOTHE/ARTELOPE, 3 autos sacramentales en CC0 y
3 comedias en ediciones de Simon Kroll: *Los empeños de un acaso*, *Eco y Narciso* y *El secreto a voces*). Fuentes y licencias: `data/FUENTES.md`.

⏱️ **≈ 30 min** (martes 16:30–17:00). Por defecto usamos la **copia de seguridad** ya entrenada
(`ENTRENAR = False`, ≈ 15 min en total). Con GPU T4 y tiempo, pon `ENTRENAR = True` (+ ~10 min de entrenamiento).
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config, set_seed
cfg = load_config(overrides={
    "calderon": {
        # "epochs": 1,              # menos tiempo de entrenamiento
        # "max_train_pairs": 1000,
    }
})
ajustes = cfg.calderon
set_seed(cfg.seed)

ENTRENAR = False       # True = entrenar tú (GPU T4, ~10 min); False = usar el adaptador ya entrenado de backups/

# Preguntas de prueba: (pregunta, respuesta esperada según el corpus)
PREGUNTAS = [
    ("¿Quién es Cosme en «La dama duende»?", "El gracioso (criado de don Manuel)."),
    ("¿Qué personajes hablan más en «La vida es sueño»?", "Segismundo, Rosaura y Clotaldo."),
    ("¿Quién dice estos versos en «La vida es sueño»?: «Hipogrifo violento, / que corriste parejas con el viento»",
     "Rosaura, al comienzo de la jornada 1 (¡en esta edición: «Hipogrifo triste»!)."),
    ("¿Quién es Pedro Crespo en «El alcalde de Zalamea»?", "Un labrador (rico) de Zalamea, padre de Isabel."),
    ("¿Cuántas jornadas tiene «El médico de su honra»?", "Tres."),
    ("¿Por qué está encerrado Segismundo en una torre?",
     "Su padre, el rey Basilio, lo encierra por un horóscopo que anunciaba que sería un rey tirano."),
]
"""),
    md("## 1. Leer las obras (TEI-XML)\nEl TEI marca jornadas, personajes (`<sp>`, `<speaker>`), versos (`<l>`) y acotaciones (`<stage>`)."),
    code("""
from dhllm import calderon as cal
obras = cal.load_plays(ajustes.tei_folder)
"""),
    code("""
obra = obras[0]
print(cal.play_card(obra), "\\n")
print(cal.play_to_text(obra)[:1200])
"""),
    md("## 2. Cargar el modelo"),
    code("""
from dhllm.finetune import load_base_model
modelo, tokenizer = load_base_model(ajustes.base_model)
"""),
    md("""
## 3. (A) Preguntar al modelo base
Fíjate en si las respuestas son **correctas** o solo **suenan bien**. Un modelo de lenguaje siempre
produce un texto plausible, aunque no «sepa» la respuesta: eso es una **alucinación**.
"""),
    code("""
from dhllm.sft import ask
resultados = {p: {"pregunta": p, "esperada": e} for p, e in PREGUNTAS}
for p, _ in PREGUNTAS:
    resultados[p]["A · base"] = ask(modelo, tokenizer, p, system=cal.SYSTEM_CALDERON, max_new_tokens=60, temperature=0)
    print(f"❓ {p}\\n💬 {resultados[p]['A · base']}\\n")
"""),
    md("""
## 4. (B) RAG: buscar primero, responder después
Construimos un **índice de búsqueda** (BM25, la técnica clásica de los buscadores) con fragmentos de
~12 versos y una «ficha» por obra (reparto, jornadas...). Para cada pregunta buscamos los fragmentos más
parecidos y se los damos al modelo.
"""),
    code("""
from pathlib import Path
indice = cal.BM25Index(cal.make_fragments(obras))
print(len(indice.fragments), "fragmentos en el índice")
# Plan B: índice precalculado → indice = cal.BM25Index.load(ajustes.backup_index)
"""),
    md("Así ve el buscador una pregunta:"),
    code("""
cal.show_fragments(indice.search(PREGUNTAS[0][0], k=ajustes.rag_k))
"""),
    code("""
for p, _ in PREGUNTAS:
    respuesta, _frags = cal.ask_with_rag(modelo, tokenizer, p, indice, k=ajustes.rag_k, max_new_tokens=80)
    resultados[p]["B · RAG"] = respuesta
    print(f"❓ {p}\\n💬 {respuesta}\\n")
"""),
    md("""
## 5. (C) Ajuste fino (SFT) con preguntas generadas automáticamente
A partir del TEI generamos preguntas con respuesta **verificable**: reparto, papel de cada personaje,
quién habla más, quién dice unos versos, de qué obra son, cómo continúa un pasaje... Algunas llevan
fragmentos de RAG para que el modelo aprenda a usarlos. Reservamos unos pares para evaluar.
"""),
    code("""
pares_train, pares_test = cal.prepare_training_pairs(obras, indice, ajustes, seed=cfg.seed)
pares_train[:3]
"""),
    code("""
from dhllm.finetune import prepare_model, load_adapter, save_adapter
from dhllm.sft import train_sft

if ENTRENAR:
    modelo = prepare_model(modelo, ajustes.method, ajustes.lora_r, ajustes.lora_alpha)
    historial = train_sft(modelo, tokenizer, pares_train, ajustes)
    save_adapter(modelo, ajustes.save_dir)
else:
    modelo = load_adapter(modelo, ajustes.backup_adapter)
    print("Nota: la copia de seguridad se entrenó en CPU con ajustes reducidos y sin «Eco y Narciso» ni",
          "«El secreto a voces» (ver backups/LEEME.md). Con RAG sí se encuentran sus versos.")
"""),
    md("## 6. (C) y (D): preguntar al modelo ajustado, sin y con RAG"),
    code("""
for p, _ in PREGUNTAS:
    resultados[p]["C · SFT"] = ask(modelo, tokenizer, p, system=cal.SYSTEM_CALDERON, max_new_tokens=60, temperature=0)
    resultados[p]["D · SFT+RAG"] = cal.ask_with_rag(modelo, tokenizer, p, indice, k=ajustes.rag_k, max_new_tokens=80)[0]
"""),
    md("## 7. Comparación\nLee la tabla con ojos de filólogo/a: ¿qué columna acierta?, ¿cuál inventa?, ¿cuál cita?"),
    code("""
cal.comparison_table(list(resultados.values()))
"""),
    opcional("""
### ¿Y con preguntas que el modelo **no** vio al entrenar?
Estos pares se generaron igual que los de entrenamiento, pero los reservamos. Comparamos el modelo
ajustado sin y con RAG (con `modelo.disable_adapter()` podemos ver también el modelo base).
"""),
    code("""
filas = []
for par in pares_test[:6]:
    with modelo.disable_adapter():
        base = ask(modelo, tokenizer, par["pregunta"], system=cal.SYSTEM_CALDERON, max_new_tokens=50, temperature=0)
    filas.append({
        "pregunta": par["pregunta"], "esperada": par["respuesta"], "A · base": base,
        "C · SFT": ask(modelo, tokenizer, par["pregunta"], system=cal.SYSTEM_CALDERON, max_new_tokens=50, temperature=0),
        "D · SFT+RAG": cal.ask_with_rag(modelo, tokenizer, par["pregunta"], indice, k=ajustes.rag_k, max_new_tokens=60)[0],
    })
cal.comparison_table(filas)
"""),
    md("""
## 💬 Para discutir: *el ajuste fino cambia el comportamiento y el estilo; el RAG aporta los hechos*

- El **modelo base** inventa con seguridad: nombres, tramas, fechas.
- El **RAG** encuentra los versos reales; el modelo los resume y puede **citar** el fragmento. Si el buscador
  no encuentra el pasaje correcto (pregunta sobre la *trama*, p. ej. Segismundo en la torre), el modelo
  tampoco acierta: la calidad depende de la búsqueda.
- El **SFT** enseña el **formato** («Los dice X, en la jornada N») y algunos datos que se repitieron mucho, pero
  no puede memorizar 30 000 versos: con citas nuevas suele adivinar un personaje plausible.
- **SFT + RAG** suele ser lo mejor: el formato aprendido + los hechos del corpus.
- ⚠️ El modelo solo «conoce» **esta edición**: en la de EMOTHE, *La vida es sueño* empieza con «Hipogrifo
  *triste*», no «violento». ¿Qué implica esto para la crítica textual y para usar LLM como fuente?

## 🖐️ Tu turno
1. Añade tus propias preguntas a `PREGUNTAS` (sobre el reparto, sobre citas, sobre la trama).
2. Cambia `rag_k` (1, 3, 6). ¿Mejoran las respuestas con más fragmentos o se confunde el modelo?
3. Busca con `indice.search("honor", k=5)` o con otra palabra clave: ¿en qué obras aparece?
4. **Para pensar:** ¿podríamos publicar este modelo ajustado? ¿Quién tiene derechos sobre una edición crítica?
"""),

]



# ================================================================ extras
E2 = [
    md("""
# E2 · (Extra) Ajuste de **estilo**: escribir «como Calderón»

**Objetivo:** ver la otra cara del ajuste fino. En el cuaderno 04 enseñamos al modelo a **responder**;
aquí solo le damos **versos** (sin preguntas) para que imite el estilo.

Entrenamos unos minutos con el texto de 2–3 obras y pedimos al modelo que continúe un parlamento.
Compara el resultado con el del modelo sin ajustar: ¿cambia el **estilo** (verso, léxico, ritmo)? ¿Sabe más **hechos**?

⏱️ Con GPU T4: unos 5 min. En CPU pon `"max_steps": 20` en los ajustes.
"""),
    code(SETUP_CODE),
    md("## ⚙️ Ajustes"),
    code("""
from dhllm import load_config, set_seed
cfg = load_config(overrides={
    "calderon_estilo": {
        # "max_steps": 20,   # ← quita el «#» en CPU
    }
})
ajustes = cfg.calderon_estilo
set_seed(cfg.seed)
INICIO = "Segismundo:\\n    ¡Ay mísero de mí, ay infelice!"
"""),
    md("## 1. Textos y modelo"),
    code("""
from dhllm import calderon as cal
from dhllm.finetune import load_base_model, make_chunks, prepare_model, finetune_on_texts, generate_hf
obras = cal.load_plays(cfg.calderon.tei_folder)
textos_estilo = [cal.play_to_text(o) for o in obras[: ajustes.n_plays]]
modelo, tokenizer = load_base_model(cfg.calderon.base_model)
print("ANTES:\\n", generate_hf(modelo, tokenizer, INICIO, max_new_tokens=80))
"""),
    md("## 2. Entrenar y comparar"),
    code("""
tr, va = make_chunks(textos_estilo, tokenizer, block_size=ajustes.block_size)
modelo = prepare_model(modelo, "lora", ajustes.lora_r, ajustes.lora_alpha)
finetune_on_texts(modelo, tr, va, ajustes)
print("DESPUÉS:\\n", generate_hf(modelo, tokenizer, INICIO, max_new_tokens=80))
"""),
    md("""
## 🖐️ Tu turno
1. Cambia `INICIO` por el comienzo de otro parlamento o por una frase en prosa.
2. Usa solo los autos sacramentales (`obras[8:11]`). ¿Cambia el estilo?
3. **Para pensar:** el modelo «escribe como Calderón», pero ¿podríamos atribuirle esos versos? ¿Qué diferencia hay con la estilometría?
"""),
]

# ================================================================ cuadernos finales
write_notebook(f"{TALLER}/01_preparar_y_explorar_tus_textos.ipynb", [
    md("""
# 01 · Preparar el entorno y explorar tus textos

⏱️ **≈ 30 min** (martes 13:00–14:00, después de presentar el repositorio)

| Parte | Contenido | Tiempo |
|---|---|---|
| A | Preparación: GPU, configuración, prueba rápida | ≈ 10 min |
| B | Explorar tus textos: estadísticas, palabras frecuentes, KWIC; subir tu propio corpus | ≈ 20 min |
"""),
    *part("Parte A", N00),
    *part("Parte B", N01, drop_sources=(SETUP_CODE,)),
])

write_notebook(f"{TALLER}/02_del_texto_al_gpt_diminuto.ipynb", [
    md("""
# 02 · Del texto al GPT diminuto: tokenizador, entrenamiento desde cero y muestreo

⏱️ **≈ 50 min** (martes 15:00–15:50)

| Parte | Contenido | Tiempo |
|---|---|---|
| A | El tokenizador: de texto a números (BPE) | ≈ 15 min |
| B | Preentrenar un GPT diminuto desde cero (2–4 min de entrenamiento en T4) | ≈ 25 min |
| C | Laboratorio de muestreo: temperatura, top-k, top-p | ≈ 10 min |

Las secciones marcadas con ⭐ **Opcional** se pueden saltar.
"""),
    code(SETUP_CODE),
    *part("Parte A", N02, drop_sources=(SETUP_CODE,)),
    *part("Parte B", N03, drop_sources=(SETUP_CODE,)),
    *part("Parte C", N06, drop_sources=(SETUP_CODE,)),
])

write_notebook(f"{TALLER}/03_ajuste_fino_con_tus_textos.ipynb", N04)

write_notebook(f"{TALLER}/04_calderon_preguntas.ipynb", N07)

write_notebook("notebooks/extras/E1_instrucciones_sft.ipynb", N05)
write_notebook("notebooks/extras/E2_calderon_estilo.ipynb", E2)

build_python_notebooks.main()
