"""Ejecuta todos los cuadernos en modo de prueba rápida (DHLLM_SMOKE_TEST=1).

Uso:  python tests/run_notebooks.py [patrón]      (p. ej. "0*" o "P*")
Los cuadernos con `ENTRENAR = ...` se ejecutan dos veces (entrenando y con la copia de seguridad).
Los cuadernos ejecutados se guardan en outputs/notebooks_ejecutados/ (los originales no cambian).
"""

import os
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DHLLM_SMOKE_TEST", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
out_dir = ROOT / "outputs" / "notebooks_ejecutados"
out_dir.mkdir(parents=True, exist_ok=True)

import re

def variants(nb):
    """Los cuadernos con ENTRENAR se prueban dos veces: entrenando y con la copia de seguridad."""
    src = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    if not re.search(r"^ENTRENAR = (True|False)", src, flags=re.M):
        return [("", nb)]
    out = []
    for valor in ("True", "False"):
        v = nbformat.reads(nbformat.writes(nb), as_version=4)
        for c in v.cells:
            if c.cell_type == "code":
                c.source = re.sub(r"^ENTRENAR = (True|False)", f"ENTRENAR = {valor}", c.source, flags=re.M)
        out.append((f" [ENTRENAR={valor}]", v))
    return out


failed = []
for nb_path in sorted((ROOT / "notebooks").glob(f"**/{pattern}.ipynb")):
    for etiqueta, nb in variants(nbformat.read(nb_path, as_version=4)):
        t0 = time.time()
        try:
            ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
                nb, {"metadata": {"path": str(nb_path.parent)}})
            status = "OK"
        except Exception as e:  # noqa: BLE001
            status = f"ERROR: {str(e)[-1500:]}"
            failed.append(nb_path.name + etiqueta)
        sufijo = etiqueta.strip(" []").replace("=", "_")
        nombre = nb_path.stem + (f"__{sufijo}" if sufijo else "") + ".ipynb"
        nbformat.write(nb, out_dir / nombre)
        print(f"[{time.time() - t0:6.1f} s] {nb_path.relative_to(ROOT)}{etiqueta}: {status}", flush=True)

print("\nFallos:", failed or "ninguno")
sys.exit(1 if failed else 0)
