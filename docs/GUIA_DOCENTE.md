# Guía para docentes

## Calendario sugerido (14–15 de diciembre de 2026)

| Día | Bloque | Cuadernos |
|---|---|---|
| Día 1, 10:00–14:00 | Python para humanistas | P1–P5 |
| Día 1, 15:00–16:00 | Análisis de texto y bigramas (puente a la clase) | P6 |
| Día 2, 13:00–14:00 | Presentación del repositorio, preparar corpus | 00, 01 |
| Día 2, 15:00–16:15 | Tokenizador y GPT diminuto | 02, 03 (y 04 si hay tiempo) |
| Día 2, 16:30–17:00 | Muestreo; SFT como demostración | 06, (05) |
| Día 2 (demo o extensión) | Calderón: base vs. RAG vs. SFT vs. SFT+RAG | 07 |

## Antes del curso
- [ ] Hacer público el repositorio (Colab solo abre repos públicos sin iniciar sesión en GitHub).
- [ ] Ejecutar todos los cuadernos en una cuenta de Colab nueva con GPU T4.
- [ ] Preparar copias en USB (repositorio + modelos descargados) por si falla la red.

## Plan B: copias de seguridad
- `backups/calderon_indice_rag.json.gz`: índice RAG precalculado (cuaderno 07; se crea en segundos, pero por si acaso).
- `backups/calderon_lora/` y `backups/quijote_lora/`: adaptadores LoRA para Qwen2.5-0.5B-Instruct (cuadernos 07 y 04,
  `ENTRENAR = False`). Datos del entrenamiento en `entrenamiento.json` de cada carpeta.
- Volver a entrenarlos (mejor con GPU): `python tools/entrenar_copias_de_seguridad.py todo --epocas-calderon 2 --pasos-quijote 300`.
- No publicar los adaptadores (Hugging Face, etc.): están entrenados con ediciones con licencia de uso docente.

## Problemas frecuentes
| Problema | Solución |
|---|---|
| «No hay GPU» | *Entorno de ejecución → Cambiar tipo* → T4. Si Colab no la da: trabajar en parejas, reducir `max_steps`. |
| `CUDA out of memory` | Reducir `batch_size`, `block_size` o `max_length`; usar SmolLM2-135M en lugar de Qwen2.5-0.5B. |
| `ModuleNotFoundError: dhllm` | Ejecutar la primera celda (⚙️ Preparación) otra vez. |
| La descarga de Gutenberg falla | P5 usa automáticamente la copia local de `data/gutenberg/`. |
| El texto propio da error de codificación | Guardarlo como «Texto UTF-8» (en Word: *Guardar como → Texto sin formato → UTF-8*). |

## Mantenimiento
Los cuadernos se generan con `python tools/build_notebooks.py`. Edita los textos allí (no en los `.ipynb`) y vuelve a generar.
Prueba automática: `DHLLM_SMOKE_TEST=1 python tests/run_notebooks.py`.
