"""Cargar y explorar textos propios (.txt)."""

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .utils import in_colab

WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letras (incluye á, ñ, ü...)


@dataclass
class Text:
    """Un texto con su nombre (normalmente el nombre del archivo)."""
    name: str
    content: str

    def __repr__(self):
        return f"Text('{self.name}', {len(self.content):,} caracteres)"


def clean_text(text: str, lowercase: bool = False) -> str:
    """Limpieza mínima: normaliza Unicode y saltos de línea, quita espacios repetidos."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if lowercase:
        text = text.lower()
    return text.strip() + "\n"


def strip_gutenberg_header(text: str) -> str:
    """Quita la cabecera y el pie de licencia de los libros de Project Gutenberg."""
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", text)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
    if start:
        text = text[start.end():]
        if end:
            end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", text)
            text = text[: end.start()] if end else text
    return text


def load_texts(folders=("data/sample",), root=None, encoding="utf-8", lowercase=False) -> list[Text]:
    """Lee todos los archivos .txt de una o varias carpetas.

    folders: una carpeta, un archivo o una lista de carpetas/archivos.
    """
    if isinstance(folders, (str, Path)):
        folders = [folders]
    root = Path(root) if root else Path.cwd()
    texts = []
    for folder in folders:
        p = Path(folder)
        if not p.is_absolute():
            p = root / p
        files = [p] if p.is_file() else sorted(p.glob("**/*.txt"))
        for f in files:
            try:
                raw = f.read_text(encoding=encoding)
            except UnicodeDecodeError:
                raw = f.read_text(encoding="latin-1")
                print(f"Aviso: '{f.name}' no estaba en UTF-8; se ha leído como latin-1.")
            raw = strip_gutenberg_header(raw)
            texts.append(Text(f.stem, clean_text(raw, lowercase=lowercase)))
    if not texts:
        print(f"No se ha encontrado ningún archivo .txt en {folders}.")
    else:
        total = sum(len(t.content) for t in texts)
        print(f"Cargados {len(texts)} textos ({total:,} caracteres en total).".replace(",", "."))
    return texts


def upload_texts(target_folder="data/mis_textos") -> list[Path]:
    """En Colab: abre un diálogo para subir archivos .txt propios a target_folder."""
    target = Path(target_folder)
    target.mkdir(parents=True, exist_ok=True)
    if not in_colab():
        print(f"Fuera de Colab: copia tus archivos .txt a la carpeta '{target}'.")
        return []
    from google.colab import files  # type: ignore
    uploaded = files.upload()
    saved = []
    for name, content in uploaded.items():
        path = target / Path(name).name
        path.write_bytes(content)
        saved.append(path)
        print(f"Guardado: {path}")
    return saved


def join_texts(texts: list[Text], separator: str = "\n\n") -> str:
    """Une todos los textos en una sola cadena (para entrenar)."""
    return separator.join(t.content for t in texts)


def words(text: str, lowercase: bool = True) -> list[str]:
    """Divide un texto en palabras (sin puntuación ni números)."""
    found = WORD_RE.findall(text)
    return [w.lower() for w in found] if lowercase else found


def corpus_stats(texts: list[Text]) -> list[dict]:
    """Tabla con estadísticas básicas por texto: caracteres, palabras, vocabulario..."""
    rows = []
    for t in texts:
        ws = words(t.content)
        sentences = [s for s in re.split(r"[.!?¡¿;:]+\s", t.content) if s.strip()]
        rows.append({
            "texto": t.name,
            "caracteres": len(t.content),
            "palabras": len(ws),
            "palabras_distintas": len(set(ws)),
            "riqueza_léxica": round(len(set(ws)) / max(1, len(ws)), 3),
            "long_media_palabra": round(sum(map(len, ws)) / max(1, len(ws)), 2),
            "frases_aprox": len(sentences),
        })
    _print_table(rows)
    return rows


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    try:  # en Colab/Jupyter, pandas da una tabla bonita
        import pandas as pd
        from IPython.display import display
        display(pd.DataFrame(rows).set_index(list(rows[0])[0]))
        return
    except Exception:
        pass
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


SPANISH_STOPWORDS = set("""
de la que el en y a los se del las un por con no una su para es al lo como más
mas pero sus le ya o este sí si porque esta entre cuando muy sin sobre también me hasta
hay donde quien desde todo nos durante todos uno les ni contra otros ese eso ante
ellos e esto mí antes algunos qué unos yo otro otras otra él tanto esa estos mucho
quienes nada muchos cual poco ella estar estas algunas algo nosotros mi mis tú te
ti tu tus ellas nosotras vosotros vosotras os mío mía míos mías tuyo tuya tuyos
tuyas suyo suya suyos suyas nuestro nuestra nuestros nuestras vuestro vuestra
vuestros vuestras esos esas estoy estás está estamos estáis están ha he había
era fue ser dijo así tan tenía hacer aunque cuanto vuestra merced sino pues
""".split())


def top_words(texts, n: int = 20, remove_stopwords: bool = True, plot: bool = True):
    """Las n palabras más frecuentes (opcionalmente sin palabras vacías)."""
    text = join_texts(texts) if isinstance(texts, list) else texts
    counts = Counter(w for w in words(text) if not (remove_stopwords and w in SPANISH_STOPWORDS))
    common = counts.most_common(n)
    if plot:
        import matplotlib.pyplot as plt
        labels, values = zip(*common) if common else ([], [])
        plt.figure(figsize=(10, 4))
        plt.bar(labels, values, color="#8c2d19")
        plt.xticks(rotation=60, ha="right")
        plt.title(f"Las {n} palabras más frecuentes")
        plt.ylabel("frecuencia")
        plt.tight_layout()
        plt.show()
    return common


def kwic(texts, keyword: str, width: int = 40, max_results: int = 10) -> list[str]:
    """«Palabra clave en contexto» (KWIC): muestra dónde aparece una palabra."""
    text = join_texts(texts) if isinstance(texts, list) else texts
    flat = text.replace("\n", " ")
    results = []
    for m in re.finditer(rf"\b{re.escape(keyword)}\b", flat, flags=re.IGNORECASE):
        left = flat[max(0, m.start() - width): m.start()].rjust(width)
        right = flat[m.end(): m.end() + width]
        results.append(f"{left} [{m.group(0)}] {right}")
        if len(results) >= max_results:
            break
    total = len(re.findall(rf"\b{re.escape(keyword)}\b", flat, flags=re.IGNORECASE))
    print(f"'{keyword}': {total} apariciones")
    for line in results:
        print(line)
    return results


def train_val_split(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Separa el texto en entrenamiento (90 %) y validación (10 %)."""
    cut = int(len(text) * (1 - val_fraction))
    return text[:cut], text[cut:]
