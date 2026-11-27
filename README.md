# LLM para las Humanidades Digitales · Taller (Universidad de Salamanca)

**Entrena y ajusta pequeños modelos de lenguaje con tus propios textos — sin ser programador/a.**

Material del curso de Humanidades Digitales del Dr. Simon Kroll, Universidad de Salamanca,
14 y 15 de diciembre de 2026. Todo funciona en **Google Colab** (gratis, con GPU T4) y, más despacio,
en cualquier ordenador sin GPU.

## 🎯 Objetivos

Al terminar el taller sabrás:
- cómo un modelo de lenguaje convierte texto en números (**tokenización**);
- cómo aprende a **predecir la siguiente palabra** (preentrenamiento) y qué significa la curva de pérdida;
- cómo **adaptar** un modelo existente a un corpus propio (**ajuste fino / LoRA**);
- cómo se convierte un modelo base en un «asistente» (**ajuste por instrucciones**, SFT; cuaderno 04 y extra E1);
- cómo influyen **temperatura, top-k y top-p** en el texto generado;
- por qué **el ajuste fino cambia el comportamiento y el estilo, pero el RAG aporta los hechos** (con el teatro de Calderón);
- y, sobre todo, qué pueden y qué **no** pueden hacer estos modelos con nuestros textos.

No hace falta saber programar: los cuadernos llaman a funciones sencillas (en [`src/dhllm/taller.py`](src/dhllm/taller.py))
y todo se controla con una celda de **Ajustes** o con el archivo [`config.yaml`](config.yaml).
Cada participante entrena en **su propia cuenta de Google Colab** (GPU T4 gratuita).

## 🚀 Inicio rápido

1. Haz clic en un botón **«Open in Colab»** de las tablas de abajo.
2. En Colab: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**.
3. Ejecuta las celdas en orden (**Mayús + Intro**). La primera celda descarga este repositorio e instala lo necesario.
4. *Archivo → Guardar una copia en Drive* para conservar tus cambios.

**En tu ordenador** (opcional):
```bash
git clone https://github.com/DrKroll/llm-humanidades-salamanca.git
cd llm-humanidades-salamanca
pip install -r requirements.txt
pip install -e .
jupyter lab
```

## 📚 Día 1 · Python para humanistas

Tres cuadernos; cada uno tiene dos partes (A y B).

| Cuaderno | Contenido | Tiempo | Abrir |
|---|---|---|---|
| P1 | **Colab y textos:** celdas, `print`, variables, errores · cadenas de texto (`len`, *slicing*, `split`/`join`, f-strings, Unicode) | ≈ 75 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P1_colab_y_textos.ipynb) |
| P2 | **Listas, bucles y funciones:** listas y diccionarios, contar palabras · `if`, `for`, `while`, funciones | ≈ 90 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P2_listas_bucles_funciones.ipynb) |
| P3 | **Leer y analizar el *Quijote*:** ficheros, descarga de Gutenberg (con copia local) · frecuencias, KWIC, gráficos y un modelo de bigramas | ≈ 105 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P3_leer_y_analizar_el_quijote.ipynb) |

## 🤖 Día 2 · Taller: tu propio modelo de lenguaje

| N.º | Contenido | Tiempo (T4) | Abrir |
|---|---|---|---|
| 01 | **Preparar y explorar tus textos:** comprobar GPU y configuración · cargar tu corpus, estadísticas, palabras frecuentes, KWIC | ≈ 30 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/01_preparar_y_explorar_tus_textos.ipynb) |
| 02 | **Del texto al GPT diminuto:** tokenizador BPE · GPT diminuto **desde cero** (curva de pérdida, muestras) · laboratorio de muestreo (temperatura, top-k, top-p) | ≈ 50 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/02_del_texto_al_gpt_diminuto.ipynb) |
| 03 | **Ajuste fino con tus textos:** LoRA sobre Qwen2.5-0.5B-Instruct, antes/después | ≈ 25 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/03_ajuste_fino_con_tus_textos.ipynb) |
| 04 | **Calderón:** preguntas sobre 14 obras en TEI — modelo base vs. RAG vs. ajuste fino (SFT) vs. SFT+RAG; ⭐ opcional: grafo de personajes (networkx) | ≈ 30 min (+5–10 opc.) | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/04_calderon_preguntas.ipynb) |

Las secciones marcadas con ⭐ **Opcional** se pueden saltar si no hay tiempo.

## ➕ Extras (para ir más lejos o seguir en casa)

| N.º | Contenido | Tiempo (T4) | Abrir |
|---|---|---|---|
| E1 | Ajuste por instrucciones (SFT) en miniatura: de «modelo base» a «asistente» con 30 pares sobre el Quijote y el Lazarillo | ≈ 10 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/extras/E1_instrucciones_sft.ipynb) |
| E2 | Ajuste de **estilo**: escribir «como Calderón» (LoRA con versos, sin preguntas) | ≈ 10 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/extras/E2_calderon_estilo.ipynb) |

## 📝 Usar tus propios textos

1. Prepara uno o varios archivos `.txt` en **UTF-8** (o TEI `.xml`) (idealmente 0,5–5 MB en total).
2. Cópialos a `data/mis_textos/` (en Colab: cuaderno 01, parte B, celda «Sube tus propios textos»).
3. En `config.yaml` cambia `folders: ["data/sample"]` por `folders: ["data/mis_textos"]`.

⚖️ Usa solo textos que tengas derecho a usar (derechos de autor, datos personales).

## 🗂️ Estructura del repositorio

```
llm-humanidades-salamanca/
├── config.yaml              ← todos los ajustes del taller (en español, comentados)
├── data/
│   ├── sample/              ← corpus de ejemplo: Quijote (Parte I) y Lazarillo (dominio público)
│   ├── gutenberg/           ← Quijote completo tal como lo ofrece Gutenberg (para P3)
│   ├── mis_textos/          ← aquí van tus textos
│   ├── qa/                  ← pares pregunta-respuesta para el extra E1
│   ├── calderon/tei/        ← 14 obras de Calderón en TEI-XML (solo textos libres, ver FUENTES.md)
│   └── FUENTES.md           ← procedencia y licencias de los textos
├── notebooks/
│   ├── dia1_python/         ← P1–P3: introducción a Python
│   ├── taller_llm/          ← 01–04: el taller de LLM
│   └── extras/              ← E1–E2: opcionales (SFT, estilo)
├── src/dhllm/               ← el paquete con las funciones del taller (código comentado)
│   ├── data.py              ← cargar textos, estadísticas, KWIC
│   ├── tokenizer.py         ← tokenizador BPE y visualización
│   ├── model.py             ← el GPT diminuto (estilo nanoGPT)
│   ├── pretrain.py          ← entrenamiento desde cero
│   ├── finetune.py          ← ajuste fino con LoRA (Hugging Face + PEFT)
│   ├── sft.py               ← ajuste por instrucciones
│   ├── sampling.py          ← generación: temperatura, top-k, top-p
│   ├── taller.py            ← funciones de alto nivel que usan los cuadernos 02–04
│   └── calderon.py          ← leer TEI, buscador BM25 (RAG), preguntas automáticas, grafo de personajes
├── backups/                 ← plan B: adaptadores LoRA ya entrenados e índice RAG precalculado
├── tools/                   ← scripts que generan los cuadernos (para docentes)
├── tests/                   ← pruebas automáticas
├── requirements.txt
└── LICENSE                  ← MIT
```

## ⚙️ Configuración

Ejemplo de [`config.yaml`](config.yaml):

```yaml
tiny_gpt:
  n_layer: 4        # capas del GPT diminuto
  max_steps: 1500   # pasos de entrenamiento
finetune:
  base_model: Qwen/Qwen2.5-0.5B-Instruct   # o HuggingFaceTB/SmolLM2-135M (más rápido)
  method: lora
```

En cada cuaderno, la celda **⚙️ Ajustes** permite cambiar valores solo para ese cuaderno:
```python
cfg = load_config(overrides={"tiny_gpt": {"max_steps": 300}})
```

## 🧑‍🏫 Para docentes

- Los cuadernos se generan con `python tools/build_notebooks.py` (editar allí, no en el `.ipynb`).
- Prueba rápida de todos los cuadernos (CPU, pocos pasos; 03 y 04 se prueban con y sin entrenamiento):
  `pip install nbconvert ipykernel && python tests/run_notebooks.py`
- Prueba de las funciones: `python -m pytest tests/`
- Si Colab no da GPU: reducir `max_steps` (02, parte B: 200–300) o trabajar en parejas. Los cuadernos 03 y 04
  tienen `USAR_COPIA = True` (y lo activan solos sin GPU o si falla el entrenamiento) para usar los adaptadores de `backups/` (entrenados con
  `python tools/entrenar_copias_de_seguridad.py`). **No subir los adaptadores a Hugging Face.**

## 🙏 Créditos e inspiración

- [**minimind**](https://github.com/jingyaogong/minimind) (jingyaogong): entrenar un LLM diminuto desde cero
  (tokenizador, preentrenamiento, SFT, DPO) con muy pocos recursos. Inspiró el recorrido completo del taller.
- [**nanoGPT**](https://github.com/karpathy/nanoGPT) y [**nanochat**](https://github.com/karpathy/nanochat) (Andrej Karpathy):
  el GPT diminuto de `src/dhllm/model.py` sigue su diseño.
- [**LitGPT**](https://github.com/Lightning-AI/litgpt) (Lightning AI): recetas de preentrenamiento y ajuste fino (p. ej. TinyLlama).
- [**Unsloth**](https://github.com/unslothai/unsloth): cuadernos de Colab para ajuste fino eficiente — recomendado para ir más lejos.
- [Hugging Face](https://huggingface.co) `transformers`, `tokenizers` y `peft`; modelos
  [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) y [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-0.5B).
- Textos: [Project Gutenberg](https://www.gutenberg.org); teatro de Calderón: [ARTELOPE/EMOTHE](https://emothe.uv.es)
  (Universitat de València), [DraCor](https://dracor.org)/TESO y las ediciones de Simon Kroll (ver [`data/FUENTES.md`](data/FUENTES.md)).

## 📄 Licencia

Código y materiales: [MIT](LICENSE) © 2026 Simon Kroll. Los textos de `data/sample` y `data/gutenberg` son de
dominio público; los de `data/calderon` tienen las licencias indicadas en [`data/FUENTES.md`](data/FUENTES.md)
(uso docente, sin ánimo de lucro, citando la procedencia).
Los modelos descargados de Hugging Face tienen sus propias licencias (Qwen2.5-0.5B-Instruct y SmolLM2: Apache 2.0).
