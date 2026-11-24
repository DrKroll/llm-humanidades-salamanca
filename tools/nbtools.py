"""Pequeñas utilidades para generar los cuadernos con nbformat.

Los cuadernos se generan desde estos scripts para que sea fácil mantenerlos
(y traducirlos). Después de editar, ejecuta:  python tools/build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

GITHUB_USER = "DrKroll"
REPO_NAME = "llm-humanidades-salamanca"
REPO_URL = f"https://github.com/{GITHUB_USER}/{REPO_NAME}"
ROOT = Path(__file__).resolve().parents[1]


def colab_badge(path: str) -> str:
    url = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO_NAME}/blob/main/{path}"
    return f'<a href="{url}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir en Colab"/></a>'


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    return nbf.v4.new_code_cell(text.strip("\n"))


SETUP_CODE = f'''
# ⚙️ Preparación: ejecuta esta celda primero (Mayús + Intro)
import os, sys
REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_NAME}"

if "google.colab" in sys.modules:
    # En Colab: descargamos el repositorio e instalamos el paquete del taller
    if not os.path.exists(REPO_DIR):
        !git clone -q {{REPO_URL}}
    os.chdir(REPO_DIR)
    !pip install -q -e .
else:
    # En tu ordenador: subimos de carpeta hasta encontrar config.yaml
    while not os.path.exists("config.yaml") and os.getcwd() != "/":
        os.chdir("..")
    if os.path.abspath("src") not in sys.path:
        sys.path.insert(0, os.path.abspath("src"))
print("Carpeta de trabajo:", os.getcwd())
'''

SETUP_CODE_DATA_ONLY = f'''
# ⚙️ Preparación: ejecuta esta celda primero (Mayús + Intro).
# Descarga los textos de ejemplo del curso (solo hace falta una vez).
import os, sys
REPO_URL = "{REPO_URL}"
REPO_DIR = "{REPO_NAME}"
if "google.colab" in sys.modules:
    if not os.path.exists(REPO_DIR):
        !git clone -q {{REPO_URL}}
    os.chdir(REPO_DIR)
else:
    while not os.path.exists("config.yaml") and os.getcwd() != "/":
        os.chdir("..")
print("Carpeta de trabajo:", os.getcwd())
'''


def write_notebook(rel_path: str, cells: list, title_for_badge: bool = True) -> Path:
    nb = nbf.v4.new_notebook()
    if title_for_badge:
        cells = [md(colab_badge(rel_path))] + cells
    nb.cells = cells
    # ids estables (derivados de la ruta y la posición) para que regenerar no cambie el diff
    import hashlib
    for i, c in enumerate(nb.cells):
        c["id"] = hashlib.md5(f"{rel_path}:{i}".encode()).hexdigest()[:8]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    }
    path = ROOT / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print("escrito:", rel_path)
    return path


def part(label: str, cells: list, drop_sources: tuple = ()) -> list:
    """Convierte las celdas de un antiguo cuaderno en una «parte» de un cuaderno combinado.

    - El título «# 03 · Algo» pasa a «# Parte B · Algo».
    - Se quitan las celdas repetidas en la costura (p. ej. la celda de preparación).
    """
    import copy
    import re
    out = []
    drop = {d.strip() for d in drop_sources}
    for i, c in enumerate(cells):
        c = copy.deepcopy(c)
        if i == 0 and c["cell_type"] == "markdown":
            c["source"] = re.sub(r"^# [^·\n]+· ", f"# {label} · ", c["source"], count=1)
        if c["source"].strip() in drop:
            continue
        out.append(c)
    return out


def opcional(text: str):
    """Celda de texto marcada como opcional (para quien vaya rápido o en casa)."""
    return md("> ⭐ **Opcional** — puedes saltarte esta sección si vas justo de tiempo.\n\n" + text.strip("\n"))
