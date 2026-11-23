"""Teatro de Calderón en TEI-XML: leer, preguntar con RAG y crear datos de entrenamiento.

Basado en el prototipo `tei_parser.py` / `qa_generator.py` preparado para el curso.
Funciona con dos variantes de TEI:
  - estilo DraCor / txt2tei (div type="act" → sp → l, castList con roleDesc)
  - ARTELOPE / EMOTHE (div1 type="Act", lg, role xml:id)
Solo usa la biblioteca estándar (xml.etree): no hay que instalar nada.
"""

import collections
import gzip
import json
import math
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://www.tei-c.org/ns/1.0}"
SYSTEM_CALDERON = ("Eres un asistente experto en el teatro de Calderón de la Barca. "
                   "Responde en español, con brevedad y precisión.")


# =====================================================================
# 1. Leer TEI
# =====================================================================
def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def _text(el):
    return _norm("".join(el.itertext())) if el is not None else ""


def _header(root, path):
    h = root.find(NS + "teiHeader")
    ts = h.find(f".//{NS}titleStmt")
    titles = ts.findall(NS + "title") if ts is not None else []
    main = (next((_text(t) for t in titles if t.get("type") == "main"), None)
            or next((_text(t) for t in titles if not t.get("key") and not t.get("type")), None)
            or (_text(titles[0]) if titles else Path(path).stem))
    sub = next((_text(t) for t in titles if t.get("type") == "sub"), "")
    if sub.lower() == "subtitle":
        sub = ""
    genre = ""
    for term in h.iter(NS + "term"):
        if term.get("type") in ("genreTitle", "genre"):
            genre = _text(term)
    av = h.find(f".//{NS}availability")
    licence = _text(av)[:300] if av is not None else ""
    src = ""
    for b in h.iter(NS + "bibl"):
        if b.get("type") == "originalSource":
            src = _text(b)
    if not src:
        sd = h.find(f".//{NS}sourceDesc")
        src = _text(sd)[:300] if sd is not None else ""
    date = ""
    for d in root.iter(NS + "date"):
        if d.get("type") in ("written", "print", "premiere"):
            nb, na, w = d.get("notBefore"), d.get("notAfter"), d.get("when")
            date = w or (f"{nb}-{na}" if nb and na and nb != na else (nb or na or ""))
            if date:
                break
    return dict(title=main, subtitle=sub, genre=genre, licence=licence, source=src, date=date)


def _cast(root):
    cast = []
    for ci in root.iter(NS + "castItem"):
        role = ci.find(NS + "role")
        name = re.sub(r"^<el>", "", _text(role) if role is not None else _text(ci)).strip(" .,")
        if name:
            cast.append({"name": name, "desc": _text(ci.find(NS + "roleDesc")).strip(" .,")})
    if not cast:
        for p in root.iter(NS + "person"):
            pn = p.find(NS + "persName")
            if pn is not None:
                cast.append({"name": _text(pn).strip(" .,"), "desc": ""})
    return cast


def _walk(el, ctx, out):
    tag, typ = el.tag, (el.get("type") or "").lower()
    if tag in (NS + "div", NS + "div1") and typ in ("act", "jornada"):
        ctx["acts"][0] += 1
        ctx = dict(ctx, act=ctx["acts"][0])
    if tag == NS + "sp":
        spk = (_text(el.find(NS + "speaker")) or (el.get("who") or "").lstrip("#")).strip(" .:")
        ctx["sps"][0] += 1
        for sub in el.iter():
            if sub.tag in (NS + "l", NS + "p"):
                out.append(dict(kind="verse" if sub.tag == NS + "l" else "prose", act=ctx["act"],
                                sp=ctx["sps"][0], speaker=spk, n=sub.get("n"), part=sub.get("part", ""),
                                text=_text(sub)))
            elif sub.tag == NS + "stage":
                out.append(dict(kind="stage", act=ctx["act"], sp=None, speaker=None, n=None, part="",
                                text=_text(sub)))
        return
    if tag == NS + "stage":
        out.append(dict(kind="stage", act=ctx["act"], sp=None, speaker=None, n=None, part="", text=_text(el)))
        return
    for ch in el:
        _walk(ch, ctx, out)


def parse_play(path) -> dict:
    """Lee una obra en TEI. Devuelve un diccionario con metadatos, reparto y versos."""
    root = ET.parse(path).getroot()
    play = _header(root, path)
    play["file"] = Path(path).name
    play["cast"] = _cast(root)
    recs = []
    body = root.find(f".//{NS}body")
    if body is not None:
        _walk(body, dict(act=None, acts=[0], sps=[0]), recs)
    for r in recs:
        r["act"] = r["act"] or 1  # los autos no tienen jornadas: acto único
    play["records"] = [r for r in recs if r["text"]]
    verses = [r for r in play["records"] if r["kind"] == "verse"]
    play["n_acts"] = len({r["act"] for r in play["records"]})
    play["n_verses"] = sum(1 for r in verses if r["part"] not in ("M", "F"))
    play["top_speakers"] = collections.Counter(r["speaker"] for r in verses).most_common(5)
    return play


def load_plays(folder="data/calderon/tei", show=True) -> list[dict]:
    """Lee todas las obras TEI de una carpeta y muestra un resumen."""
    files = sorted(Path(folder).glob("*.xml"))
    plays = [parse_play(f) for f in files]
    if show:
        print(f"{len(plays)} obras cargadas de {folder}\n")
        for p in plays:
            print(f"  · {p['title'][:45]:<45} {p['genre'][:18]:<18} {p['n_acts']} jorn. {p['n_verses']:>5} versos")
    return plays


def clean_name(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip(" .,:")
    return s.title() if s.isupper() else s


def play_to_text(play: dict) -> str:
    """Texto plano de una obra («HABLANTE: verso»), útil para leer o para entrenar el estilo."""
    out = [f"# {play['title']}", ""]
    act, spk = None, None
    for r in play["records"]:
        if r["act"] != act:
            act, spk = r["act"], None
            out += ["", f"## Jornada {act}", ""]
        if r["kind"] == "stage":
            out.append(f"[{r['text']}]")
            spk = None
        else:
            if r["speaker"] != spk:
                out.append(f"{clean_name(r['speaker'])}:")
                spk = r["speaker"]
            out.append("    " + r["text"])
    return "\n".join(out) + "\n"


def speeches(play: dict) -> list[dict]:
    """Agrupa los versos por parlamento (lo que dice un personaje de una vez)."""
    out = collections.OrderedDict()
    for r in play["records"]:
        if r["kind"] in ("verse", "prose"):
            s = out.setdefault((r["act"], r["sp"]), {"speaker": r["speaker"], "act": r["act"],
                                                     "n": r["n"], "lines": []})
            s["lines"].append(r["text"])
    return list(out.values())


# =====================================================================
# 2. RAG: buscar fragmentos relevantes (BM25)
# =====================================================================
STOP = set("""de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o
este si porque esta entre cuando muy sin sobre me hasta hay donde quien desde todo nos ni ese eso yo él mi
tan así era fue qué quién quiénes cuál cuáles cómo dónde cuándo obra calderón dice dicen verso versos
personaje personajes""".split())


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def tokenize_es(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-zñ]+", _fold(text)) if w not in {_fold(x) for x in STOP} and len(w) > 1]


def play_card(play: dict) -> str:
    """«Ficha» de una obra con sus metadatos (reparto, jornadas, versos, personajes con más texto)."""
    cast = [c for c in play["cast"] if c["name"] and not NOISE.search(c["name"])]
    reparto = "; ".join(clean_name(c["name"]) + (f" ({c['desc'].lower()})" if c.get("desc") else "") for c in cast)
    cnt = collections.Counter(s["speaker"] for s in speeches(play) for _ in s["lines"])
    top = ", ".join(f"{clean_name(n)} ({v} versos)" for n, v in cnt.most_common(3))
    gen = play["genre"].lower() if play.get("genre") else "obra dramática"
    return (f"Ficha de «{play['title']}» ({gen} de Calderón de la Barca). Reparto: {reparto}. "
            f"Jornadas: {play['n_acts']}. Versos: {play['n_verses']}. Personajes con más versos: {top}.")


def make_fragments(plays: list[dict], lines_per_fragment: int = 12, with_cards: bool = True) -> list[dict]:
    """Divide cada obra en fragmentos de ~12 versos (sin mezclar jornadas).
    with_cards=True añade una «ficha» por obra (reparto, jornadas...) que también se puede encontrar."""
    frags = []
    for p in plays:
        if with_cards:
            card = play_card(p)
            frags.append({"obra": p["title"], "jornada": "-", "verso": "ficha", "texto": card, "buscar": card})
        buf, act, first_n = [], None, None
        spk = None

        raw = []  # texto sin los nombres de los hablantes (para la búsqueda)

        def flush():
            if buf:
                frags.append({"obra": p["title"], "jornada": act, "verso": first_n, "texto": "\n".join(buf),
                              "buscar": " ".join(raw)})

        for r in p["records"]:
            if r["act"] != act or len([b for b in buf if not b.startswith("[")]) >= lines_per_fragment:
                flush()
                buf, raw, act, first_n, spk = [], [], r["act"], r.get("n"), None
            raw.append(r["text"])
            if r["kind"] == "stage":
                buf.append(f"[{r['text']}]")
                spk = None
            else:
                prefix = f"{clean_name(r['speaker'])}: " if r["speaker"] != spk else "    "
                spk = r["speaker"]
                buf.append(prefix + r["text"])
                first_n = first_n or r.get("n")
        flush()
    return frags


class BM25Index:
    """Buscador clásico BM25 (como un mini-Google para nuestros versos). Funciona en CPU, sin descargas."""

    def __init__(self, fragments: list[dict], k1: float = 1.5, b: float = 0.75):
        self.fragments = fragments
        self.k1, self.b = k1, b
        self.docs = [collections.Counter(tokenize_es(f.get("buscar", f["texto"]))) for f in fragments]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        df = collections.Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        # Nombres para reconocer la obra en la pregunta: título completo, antes de la coma, 4 primeras palabras
        self.aliases = []
        for t in {f["obra"] for f in fragments}:
            for a in {t, t.split(",")[0], " ".join(t.split()[:4])}:
                if len(a) > 8:
                    self.aliases.append((_fold(a), t))
        self.aliases.sort(key=lambda x: len(x[0]), reverse=True)

    def detect_play(self, query: str):
        """Si la pregunta menciona el título de una obra, devuelve ese título."""
        q = _fold(query)
        for alias, title in self.aliases:
            if alias in q:
                return title
        return None

    def search(self, query: str, k: int = 4, play: str | None = "auto", with_card: bool = True) -> list[dict]:
        """Devuelve los k fragmentos más parecidos a la pregunta.
        play="auto": si la pregunta menciona una obra, solo se busca en esa obra
        y (with_card=True) su ficha (reparto, jornadas...) va siempre en primer lugar."""
        if play == "auto":
            play = self.detect_play(query)
        card = []
        if play and with_card:
            card = [dict(f, puntuacion="ficha") for f in self.fragments if f["obra"] == play and f["verso"] == "ficha"]
            k = max(1, k - len(card))
        q = tokenize_es(query)
        scores = []
        for i, d in enumerate(self.docs):
            if play and self.fragments[i]["obra"] != play:
                continue
            s = 0.0
            for w in q:
                if w in d:
                    tf = d[w]
                    s += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        hits = [dict(self.fragments[i], puntuacion=round(s, 2)) for s, i in scores
                if self.fragments[i]["verso"] != "ficha" or not card][:k]
        return card + hits

    # guardar / cargar (copia de seguridad precalculada)
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(self.fragments, f, ensure_ascii=False)
        print(f"Índice guardado en {path} ({len(self.fragments)} fragmentos)")

    @classmethod
    def load(cls, path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls(json.load(f))


def show_fragments(frags: list[dict], max_chars: int = 400):
    for i, f in enumerate(frags, 1):
        print(f"[{i}] «{f['obra']}», jornada {f['jornada']}, v. {f['verso']}  (puntuación {f.get('puntuacion', '-')})")
        print("    " + f["texto"][:max_chars].replace("\n", "\n    ") + "\n")


def context_from_fragments(frags: list[dict]) -> str:
    return "\n\n".join(f"[{i}] «{f['obra']}», jornada {f['jornada']}, v. {f['verso']}:\n{f['texto']}"
                       for i, f in enumerate(frags, 1))


def ask_with_rag(model, tokenizer, question: str, index: BM25Index, k: int = 4, show: bool = False, **kw):
    """Busca fragmentos y se los da al modelo junto con la pregunta (RAG)."""
    from .sft import ask
    frags = index.search(question, k=k)
    if show:
        show_fragments(frags)
    kw.setdefault("temperature", 0)
    answer = ask(model, tokenizer, question, system=SYSTEM_CALDERON, context=context_from_fragments(frags), **kw)
    return answer, frags


# =====================================================================
# 3. Pares pregunta-respuesta automáticos (para SFT)
# =====================================================================
NOISE = re.compile(r"^(personas|personajes|todos|dentro|acompañamiento|música|músicos|criados?|soldados?|"
                   r"guardas?|gente|otros|voces|una voz)\b|\d", re.I)

PARAFRASIS = {
    "reparto": ["¿Qué personajes aparecen en «{t}»?", "¿Quiénes son los personajes de «{t}»?",
                "Dime el reparto de «{t}»."],
    "rol": ["¿Quién es {n} en «{t}»?", "¿Qué papel tiene {n} en «{t}»?"],
    "protagonismo": ["¿Qué personajes hablan más en «{t}»?", "¿Quién tiene más versos en «{t}»?",
                     "¿Cuáles son los personajes principales de «{t}» por número de versos?"],
    "inicio": ["¿Quién pronuncia los primeros versos de «{t}»?", "¿Cómo empieza «{t}»?"],
    "jornadas": ["¿Cuántas jornadas tiene «{t}»?", "¿En cuántas jornadas se divide «{t}»?"],
    "extension": ["¿Cuántos versos tiene «{t}»?", "¿Qué extensión tiene «{t}»?"],
    "cita_hablante": ["¿Quién dice estos versos en «{t}»?: «{q}»", "En «{t}», ¿qué personaje pronuncia «{q}»?"],
    "cita_obra": ["¿De qué obra de Calderón son estos versos?: «{q}»", "¿En qué comedia o auto aparece «{q}»?"],
    "continuar": ["Continúa estos versos de «{t}»: «{q}»", "¿Cómo sigue este pasaje de «{t}»?: «{q}»"],
}


def generate_qa(plays: list[dict], quotes_per_play: int = 120, seed: int = 0) -> list[dict]:
    """Crea pares pregunta-respuesta verificables a partir de la estructura TEI (sin inventar nada)."""
    rng = random.Random(seed)
    out = []

    def add(kind, play, answer, **fmt):
        q = rng.choice(PARAFRASIS[kind]).format(t=play["title"], **fmt)
        out.append({"pregunta": q, "respuesta": answer, "tipo": kind, "obra": play["title"]})

    for p in plays:
        t = p["title"]
        cast = [c for c in p["cast"] if c["name"] and not NOISE.search(c["name"])]
        if cast:
            add("reparto", p, f"En «{t}» aparecen: {', '.join(clean_name(c['name']) for c in cast)}.")
            for c in cast:
                if c.get("desc") and len(c["desc"]) < 60:
                    add("rol", p, f"En el reparto de «{t}», {clean_name(c['name'])} figura como {c['desc'].lower()}.",
                        n=clean_name(c["name"]))
        sp = speeches(p)
        cnt = collections.Counter(s["speaker"] for s in sp for _ in s["lines"])
        if cnt:
            top = ", ".join(f"{clean_name(n)} ({v} versos)" for n, v in cnt.most_common(3))
            add("protagonismo", p, f"Por número de versos, los personajes con más texto en «{t}» son: {top}.")
        if sp:
            add("inicio", p, f"La obra comienza con {clean_name(sp[0]['speaker'])}: «{' / '.join(sp[0]['lines'][:2])}».")
        if p["n_acts"] > 1:
            add("jornadas", p, f"«{t}» está dividida en {p['n_acts']} jornadas.")
        add("extension", p, f"En la edición digital del corpus, «{t}» tiene {p['n_verses']} versos.")
        cands = [s for s in sp if len(s["lines"]) >= 2 and all(len(x) > 12 for x in s["lines"][:2])]
        for s in rng.sample(cands, min(quotes_per_play, len(cands))):
            q2, who = " / ".join(s["lines"][:2]), clean_name(s["speaker"])
            k = rng.random()
            if k < 0.45:
                add("cita_hablante", p, f"Los dice {who}, en la jornada {s['act']}.", q=q2)
            elif k < 0.75:
                add("cita_obra", p, f"Son de «{t}»; los pronuncia {who} (jornada {s['act']}).", q=q2)
            elif len(s["lines"]) >= 4:
                add("continuar", p, " / ".join(s["lines"][2:6]), q=q2)
    rng.shuffle(out)
    for x in out:
        x["system"] = SYSTEM_CALDERON
    tipos = collections.Counter(x["tipo"] for x in out)
    print(f"{len(out)} pares generados: " + ", ".join(f"{k} {v}" for k, v in tipos.most_common()))
    return out


def add_context(pairs: list[dict], index: "BM25Index", fraction: float = 0.3, k: int = 2, seed: int = 0) -> list[dict]:
    """Añade fragmentos recuperados (RAG) a una parte de los pares de entrenamiento.
    Así el modelo aprende también a USAR los fragmentos cuando se los damos (SFT + RAG)."""
    rng = random.Random(seed)
    out = []
    for p in pairs:
        p = dict(p)
        if rng.random() < fraction:
            p["contexto"] = context_from_fragments(index.search(p["pregunta"], k=k))
        out.append(p)
    print(f"{sum('contexto' in p for p in out)} de {len(out)} pares llevan fragmentos (contexto RAG).")
    return out


def split_qa(pairs: list[dict], test_size: int = 20, seed: int = 0):
    """Separa unos pares para evaluar (el modelo no los verá al entrenar)."""
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    return pairs[test_size:], pairs[:test_size]


def comparison_table(rows: list[dict]):
    """Tabla HTML: una fila por pregunta, una columna por método."""
    try:
        import html
        from IPython.display import HTML, display
        cols = list(rows[0])
        head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
        body = "".join("<tr>" + "".join(f"<td style='vertical-align:top;text-align:left;white-space:pre-wrap'>"
                                        f"{html.escape(str(r[c]))}</td>" for c in cols) + "</tr>" for r in rows)
        display(HTML(f"<table><tr>{head}</tr>{body}</table>"))
    except ImportError:
        for r in rows:
            for c, v in r.items():
                print(f"{c}: {v}")
            print("-" * 60)


def prepare_training_pairs(plays: list[dict], index: "BM25Index", settings: dict, seed: int = 0):
    """Genera los pares, separa los de evaluación y añade contexto RAG a una parte."""
    pairs = generate_qa(plays, quotes_per_play=settings["quotes_per_play"], seed=seed)
    train, test = split_qa(pairs, test_size=settings["test_size"], seed=seed)
    train = train[: settings["max_train_pairs"]]
    train = add_context(train, index, fraction=settings["context_fraction"], k=2, seed=seed)
    print(f"Entrenamiento: {len(train)} pares · evaluación: {len(test)} pares")
    return train, test
