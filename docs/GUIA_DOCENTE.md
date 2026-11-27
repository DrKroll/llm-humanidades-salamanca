# Guía para docentes

## Calendario sugerido (14–15 de diciembre de 2026)

7 cuadernos obligatorios (3 + 4). Cada uno se divide en partes A/B(/C); las secciones con ⭐ **Opcional**
se pueden saltar. Lo que no cabe está en `notebooks/extras/`.

### Día 1 · lunes 14 · Python para humanistas
| Hora | Cuaderno | Contenido | Tiempo |
|---|---|---|---|
| 9:30–10:00 | — | Bienvenida, comprobar Colab y cuenta de Google | 30′ |
| 10:00–11:15 | **P1** · Colab y textos | A: celdas, `print`, variables, errores (≈ 30′) · B: cadenas de texto (≈ 45′) | 75′ |
| 11:45–13:15 | **P2** · Listas, bucles y funciones | A: listas y diccionarios (≈ 40′) · B: `if`, `for`, funciones (≈ 50′) | 90′ |
| 13:15–14:00 | **P3**, parte A | Ficheros: descargar y limpiar el *Quijote* | 45′ |
| 15:00–16:00 | **P3**, parte B | Frecuencias, KWIC, gráficos, bigramas (puente a la clase). Tras la comida: volver a ejecutar la celda ⚙️ | 60′ |
| 16:15–17:00 | — | Clase LLM, parte I-a | 45′ |

### Día 2 · martes 15 · Taller
| Hora | Cuaderno | Contenido | Tiempo |
|---|---|---|---|
| 13:00–13:30 | — | Presentación del repositorio | 30′ |
| 13:30–14:00 | **01** · Preparar y explorar tus textos | A: GPU y configuración (≈ 10′) · B: corpus, estadísticas, KWIC, subir textos (≈ 20′) | 30′ |
| 15:00–15:50 | **02** · Del texto al GPT diminuto | 0: comprobar GPU · A: tokenizador (≈ 15′) · B: GPT desde cero, 1500 pasos ≈ 2–4 min en la T4 (≈ 25′) · C: muestreo (≈ 10′) | 50′ |
| 15:50–16:15 | **03** · Ajuste fino con tus textos | Cargar Qwen2.5-0.5B (≈ 2′, descarga de ~1 GB) · antes (≈ 3′) · LoRA, 300 pasos ≈ 4–6 min en la T4, tope 8′ · después y comparación (≈ 5′) | 25′ |
| 16:30–17:00 | **04** · Calderón | Datos + modelo (≈ 3′) · A y B (≈ 5′) · **cada participante entrena su modelo**: 2000 pares, 1 época, lote 8 = 250 pasos ≈ 4–7 min en la T4, tope automático 10′ (mientras entrena, comentar las columnas A y B) · C, D y tabla (≈ 5′) · discusión (≈ 8′). ⭐ Sección 8 opcional, *Grafo de personajes* (+5–10′, o como demostración del docente) | 30′ |

Todos los participantes tienen cuenta de Google Colab: **cada uno entrena en su propia GPU T4**. Cada cuaderno
con entrenamiento empieza con la celda «0. ¿Tenemos GPU?», que explica cómo cambiar a T4.

**Si falta tiempo o falla algo:** en 02 saltar las secciones ⭐ y hacer la parte C solo con los puntos 1 y 3;
en 03/04 bajar `"pasos"`/`"max_minutos"` o poner `USAR_COPIA = True` (adaptador ya entrenado de `backups/`).
Sin GPU, o si el entrenamiento falla, 03 y 04 cargan la copia de seguridad **automáticamente** y lo dicen en pantalla.
Si la GPU se queda sin memoria, 04 repite solo con la mitad de lote.

### Extras (`notebooks/extras/`)
| Cuaderno | Contenido | Tiempo (T4) |
|---|---|---|
| **E1** | Ajuste por instrucciones (SFT) con 30 pares sobre el Quijote y el Lazarillo (antes «05») | ≈ 10′ |
| **E2** | Ajuste de estilo con versos de Calderón (antes, sección opcional del cuaderno de Calderón) | ≈ 10′ |

### Correspondencia con la numeración anterior
| Antes | Ahora |
|---|---|
| P1 + P2 | P1 |
| P3 + P4 | P2 |
| P5 + P6 | P3 |
| 00 + 01 | 01 |
| 02 + 03 + 06 | 02 (partes A, B, C) |
| 04 | 03 |
| 05 | extras/E1 |
| 07 | 04 (la sección de estilo → extras/E2) |

## Antes del curso
- [ ] Hacer público el repositorio (Colab solo abre repos públicos sin iniciar sesión en GitHub).
- [ ] Ejecutar todos los cuadernos en una cuenta de Colab nueva con GPU T4.
- [ ] Preparar copias en USB (repositorio + modelos descargados) por si falla la red.

## Plan B: copias de seguridad
- `backups/calderon_indice_rag.json.gz`: índice RAG precalculado (cuaderno 04; se crea en segundos, pero por si acaso).
- `backups/calderon_lora/` y `backups/quijote_lora/`: adaptadores LoRA para Qwen2.5-0.5B-Instruct (cuadernos 04 y 03,
  `USAR_COPIA = True`, o automáticamente sin GPU / si falla el entrenamiento). Datos del entrenamiento en `entrenamiento.json` de cada carpeta.
- **Los adaptadores actuales están entrenados en CPU con ajustes reducidos** (Calderón: 400 pares de las 14 obras, 1 época; Quijote: 60 pasos).
  Antes del curso conviene reentrenarlos en una T4 con el comando siguiente (ver `backups/LEEME.md`).
- Volver a entrenarlos (mejor con GPU): `python tools/entrenar_copias_de_seguridad.py todo --epocas-calderon 2 --pasos-quijote 300`.
- No publicar los adaptadores (Hugging Face, etc.): están entrenados con ediciones con licencia de uso docente.

## Problemas frecuentes
| Problema | Solución |
|---|---|
| «No hay GPU» | *Entorno de ejecución → Cambiar tipo* → T4. Si Colab no la da: trabajar en parejas, reducir `max_steps`. |
| `CUDA out of memory` | Reducir `batch_size`, `block_size` o `max_length`; usar SmolLM2-135M en lugar de Qwen2.5-0.5B. |
| `ModuleNotFoundError: dhllm` | Ejecutar la primera celda (⚙️ Preparación) otra vez. |
| La descarga de Gutenberg falla | P3 usa automáticamente la copia local de `data/gutenberg/`. |
| El texto propio da error de codificación | Guardarlo como «Texto UTF-8» (en Word: *Guardar como → Texto sin formato → UTF-8*). |

## Mantenimiento
**Dónde está el código:** los cuadernos solo llaman a funciones cortas de `src/dhllm/taller.py`
(`entrenar_tokenizador`, `preparar_gpt`, `entrenar_gpt`, `preparar_ajuste_fino`, `ajustar_con_mis_textos`,
`preparar_datos`, `cargar_modelo`, `responder`, `entrenar_calderon`, `tabla_comparativa`, `grafo`...). Esas funciones
usan los módulos de siempre (`tokenizer`, `pretrain`, `finetune`, `sft`, `calderon`). Los ajustes de los
cuadernos usan nombres en español (`pasos`, `epocas`, `ritmo_aprendizaje`, `max_minutos`, `obras`) que
`taller.py` traduce a las claves de `config.yaml`.

Los cuadernos se generan con `python tools/build_notebooks.py`. Edita los textos allí (no en los `.ipynb`) y vuelve a generar.
Prueba automática: `DHLLM_SMOKE_TEST=1 python tests/run_notebooks.py` (03 y 04 se ejecutan con `USAR_COPIA = False` y `True`).
Las «partes» de cada cuaderno se definen en `tools/build_notebooks.py` y `tools/build_python_notebooks.py` (función `part`).

## ⭐ Cuaderno 04, sección 8: grafo de personajes (opcional)

- **Qué es:** una red de personajes por obra, construida en menos de un segundo a partir del TEI con `networkx`
  (ya instalado en Colab; no necesita GPU ni descargas). Nodo = personaje (papel según `<castList>`, versos);
  arista = «dialoga con» (parlamentos seguidos en la misma jornada; peso = réplicas).
- **Qué muestra:** la red de *La vida es sueño*, tres medidas de centralidad (interlocutores, réplicas,
  intermediación) y la columna **E · grafo + versos** frente a **D · versos**.
- **Mensaje:** el grafo aporta la **estructura** (quién es quién, con quién habla), la búsqueda aporta las
  **citas**. En la prueba, con el adaptador de `backups/`, solo con grafo acierta «el Capitán = Don Álvaro de Ataide»
  y «Cosme = gracioso»; las citas («Al rey la hacienda…» → Crespo, j. 1) siguen viniendo del buscador.
- **Límites para comentar:** «hablar justo después» ≠ «hablar a»; muchos repartos del TEI no indican el papel;
  centralidad ≠ protagonismo (en *La vida es sueño* Segismundo tiene más versos, Clotaldo/Rosaura más réplicas).
- **Si falta tiempo:** saltar la sección; no afecta al resto del cuaderno.

