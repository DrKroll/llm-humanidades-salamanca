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
- cómo se convierte un modelo base en un «asistente» (**ajuste por instrucciones**, SFT);
- cómo influyen **temperatura, top-k y top-p** en el texto generado;
- por qué **el ajuste fino cambia el comportamiento y el estilo, pero el RAG aporta los hechos** (con el teatro de Calderón);
- y, sobre todo, qué pueden y qué **no** pueden hacer estos modelos con nuestros textos.

No hace falta saber programar: los cuadernos llaman a funciones sencillas y todo se controla con
una celda de **Ajustes** o con el archivo [`config.yaml`](config.yaml).

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

| Módulo | Contenido | Abrir |
|---|---|---|
| P1 | Primeros pasos en Colab: celdas, `print`, variables, errores | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P1_primeros_pasos_en_colab.ipynb) |
| P2 | Cadenas de texto: `len`, *slicing*, `split`/`join`, f-strings, Unicode | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P2_cadenas_de_texto.ipynb) |
| P3 | Listas y diccionarios; contar palabras | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P3_listas_y_diccionarios.ipynb) |
| P4 | `if`, `for`, `while`, funciones | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P4_bucles_condiciones_funciones.ipynb) |
| P5 | Ficheros: leer el *Quijote*, quitar la cabecera de Gutenberg | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P5_ficheros_y_don_quijote.ipynb) |
| P6 | Frecuencias, KWIC, gráficos y un modelo de bigramas | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/dia1_python/P6_analisis_de_texto.ipynb) |

## 🤖 Día 2 · Taller: tu propio modelo de lenguaje

| N.º | Contenido | Tiempo en T4 | Abrir |
|---|---|---|---|
| 00 | Comprobar el entorno (GPU, versiones, configuración) | 1 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/00_preparacion.ipynb) |
| 01 | Cargar tu corpus, estadísticas, palabras frecuentes, KWIC | 5 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/01_explorar_tus_textos.ipynb) |
| 02 | Entrenar un tokenizador BPE y visualizar los tokens | 2 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/02_tokenizador.ipynb) |
| 03 | Entrenar un GPT diminuto **desde cero** (curva de pérdida, muestras) | 3–5 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/03_preentrenar_gpt_diminuto.ipynb) |
| 04 | Ajuste fino (LoRA) de Qwen2.5-0.5B-Instruct con tus textos, antes/después | 5–10 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/04_ajuste_fino_modelo_preentrenado.ipynb) |
| 05 | (Opcional) SFT: enseñar a responder preguntas | 5 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/05_instrucciones_opcional.ipynb) |
| 06 | Temperatura, top-k, top-p: cómo elige el modelo la siguiente palabra | 2 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/06_laboratorio_de_muestreo.ipynb) |
| 07 | **Calderón:** preguntas sobre 12 obras en TEI — modelo base vs. RAG vs. ajuste fino (SFT) vs. SFT+RAG | 15–20 min | [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DrKroll/llm-humanidades-salamanca/blob/main/notebooks/taller_llm/07_calderon_preguntas.ipynb) |

## 📝 Usar tus propios textos

1. Prepara uno o varios archivos `.txt` en **UTF-8** (idealmente 0,5–5 MB en total).
2. Cópialos a `data/mis_textos/` (en Colab: cuaderno 01, celda «Sube tus propios textos»).
3. En `config.yaml` cambia `folders: ["data/sample"]` por `folders: ["data/mis_textos"]`.

⚖️ Usa solo textos que tengas derecho a usar (derechos de autor, datos personales).

## 🗂️ Estructura del repositorio

```
llm-humanidades-salamanca/
├── config.yaml              ← todos los ajustes del taller (en español, comentados)
├── data/
│   ├── sample/              ← corpus de ejemplo: Quijote (Parte I) y Lazarillo (dominio público)
│   ├── gutenberg/           ← Quijote completo tal como lo ofrece Gutenberg (para P5)
│   ├── mis_textos/          ← aquí van tus textos
│   ├── qa/                  ← pares pregunta-respuesta para el cuaderno 05
│   ├── calderon/tei/        ← 12 obras de Calderón en TEI-XML (solo textos libres, ver FUENTES.md)
│   └── FUENTES.md           ← procedencia y licencias de los textos
├── notebooks/
│   ├── dia1_python/         ← P1–P6: introducción a Python
│   └── taller_llm/          ← 00–06: el taller de LLM
├── src/dhllm/               ← el paquete con las funciones del taller (código comentado)
│   ├── data.py              ← cargar textos, estadísticas, KWIC
│   ├── tokenizer.py         ← tokenizador BPE y visualización
│   ├── model.py             ← el GPT diminuto (estilo nanoGPT)
│   ├── pretrain.py          ← entrenamiento desde cero
│   ├── finetune.py          ← ajuste fino con LoRA (Hugging Face + PEFT)
│   ├── sft.py               ← ajuste por instrucciones
│   ├── sampling.py          ← generación: temperatura, top-k, top-p
│   └── calderon.py          ← leer TEI, buscador BM25 (RAG), preguntas automáticas
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
- Prueba rápida de todos los cuadernos (CPU, pocos pasos):
  `pip install nbconvert ipykernel && python tests/run_notebooks.py`
- Prueba de las funciones: `python -m pytest tests/`
- Si Colab no da GPU: reducir `max_steps` (03: 200–300) o trabajar en parejas. Los cuadernos 04 y 07
  tienen `ENTRENAR = False` para usar los adaptadores de `backups/` (entrenados con
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
  (Universitat de València), [DraCor](https://dracor.org)/TESO y la edición de Simon Kroll (ver [`data/FUENTES.md`](data/FUENTES.md)).

## 📄 Licencia

Código y materiales: [MIT](LICENSE) © 2026 Simon Kroll. Los textos de `data/sample` y `data/gutenberg` son de
dominio público; los de `data/calderon` tienen las licencias indicadas en [`data/FUENTES.md`](data/FUENTES.md)
(uso docente, sin ánimo de lucro, citando la procedencia).
Los modelos descargados de Hugging Face tienen sus propias licencias (Qwen2.5-0.5B-Instruct y SmolLM2: Apache 2.0).
