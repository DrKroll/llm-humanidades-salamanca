# Copias de seguridad (plan B)

Material precalculado por si Colab no da GPU o el tiempo no alcanza. **No se sube a Hugging Face**: solo vive en este repositorio privado.

| Archivo | Cuaderno | Contenido |
|---|---|---|
| `calderon_indice_rag.json.gz` | 04 | Índice BM25 con 3507 fragmentos (14 obras + fichas) |
| `calderon_lora/` | 04 (`USAR_COPIA = True` o automático) | Adaptador LoRA (fp16, 17,6 MB) para Qwen2.5-0.5B-Instruct |
| `quijote_lora/` | 03 (`USAR_COPIA = True` o automático) | Adaptador LoRA (fp16) para Qwen2.5-0.5B-Instruct sobre el *Quijote* |

**Atención:** los adaptadores actuales se entrenaron en **CPU con ajustes reducidos**
(Calderón: 400 pares de las 14 obras —incluidas *Eco y Narciso* y *El secreto a voces*—, 1 época, lote 2, ~10 min; Quijote: 60 pasos, lote 2). Detalles en
`*/entrenamiento.json`. Funcionan para la demostración, pero son más flojos que un
entrenamiento completo. Versión completa con GPU (T4, ~15-20 min):

```bash
python tools/entrenar_copias_de_seguridad.py todo --epocas-calderon 2 --pasos-quijote 300
```

Licencias: el adaptador de Calderón se ha entrenado con textos que incluyen *El príncipe constante*
(EMOTHE, CC BY-NC-ND 4.0) y los textos ARTELOPE/EMOTHE (cita obligatoria); ver `data/FUENTES.md`.
Uso solo docente y no comercial; no publicar los adaptadores por separado.
