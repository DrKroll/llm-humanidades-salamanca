"""Tokenizador BPE (Byte Pair Encoding) entrenado con tus propios textos."""

import html
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]
_COLORS = ["#fde2e4", "#e2ece9", "#dfe7fd", "#fff1c1", "#e8dff5", "#d7f0f7", "#fce1c8"]


class SimpleTokenizer:
    """Envoltorio pequeño alrededor de `tokenizers.Tokenizer` con nombres fáciles."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok

    # --- uso básico -----------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Texto -> lista de números (ids de tokens)."""
        return self.tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Lista de números -> texto."""
        return self.tok.decode(list(ids))

    def tokens(self, text: str) -> list[str]:
        """Texto -> lista de trozos (tokens) legibles."""
        return [self.decode([i]) for i in self.encode(text)]

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def eos_id(self) -> int:
        return self.tok.token_to_id("<|endoftext|>")

    # --- guardar / cargar ---------------------------------------------
    def save(self, folder) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "tokenizer.json"
        self.tok.save(str(path))
        print(f"Tokenizador guardado en {path}")
        return path

    @classmethod
    def load(cls, folder) -> "SimpleTokenizer":
        path = Path(folder)
        path = path / "tokenizer.json" if path.is_dir() else path
        return cls(Tokenizer.from_file(str(path)))

    def __repr__(self):
        return f"SimpleTokenizer(vocabulario={self.vocab_size} tokens)"


def train_bpe(text_or_texts, vocab_size: int = 4000, min_frequency: int = 2) -> SimpleTokenizer:
    """Entrena un tokenizador BPE a nivel de bytes (como GPT-2) con tus textos.

    El algoritmo empieza con letras sueltas y va uniendo los pares más
    frecuentes ("q"+"u" -> "qu", "qu"+"e" -> "que"...) hasta tener
    `vocab_size` tokens.
    """
    if isinstance(text_or_texts, str):
        corpus = [text_or_texts]
    else:
        corpus = [getattr(t, "content", t) for t in text_or_texts]
    # Partimos en párrafos para que el entrenamiento sea más eficiente
    lines = [p for doc in corpus for p in doc.split("\n") if p.strip()]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(lines, trainer=trainer)
    st = SimpleTokenizer(tok)
    print(f"Tokenizador entrenado: {st.vocab_size} tokens en el vocabulario.")
    return st


def show_tokens(tokenizer, text: str, as_html: bool = True):
    """Muestra el texto coloreado según sus tokens (cada color = un token)."""
    if hasattr(tokenizer, "tokens"):
        pieces = tokenizer.tokens(text)
    else:  # tokenizador de Hugging Face
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([i]) for i in ids]
    try:
        from IPython.display import HTML, display
        if not as_html:
            raise ImportError
        spans = "".join(
            f'<span style="background:{_COLORS[i % len(_COLORS)]};padding:1px 0;'
            f'border-right:1px solid #999;white-space:pre-wrap">{html.escape(p)}</span>'
            for i, p in enumerate(pieces)
        )
        display(HTML(f'<div style="font-family:monospace;font-size:15px;line-height:2;color:#000">{spans}</div>'
                     f'<p>{len(pieces)} tokens · {len(text)} caracteres</p>'))
    except ImportError:
        print(" | ".join(pieces))
        print(f"{len(pieces)} tokens · {len(text)} caracteres")
    return pieces


def tokens_per_word(tokenizer, text: str) -> float:
    """Promedio de tokens por palabra (cuanto más bajo, más «eficiente»)."""
    n_words = max(1, len(text.split()))
    if hasattr(tokenizer, "tokens"):
        n_tokens = len(tokenizer.encode(text))
    else:
        n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return round(n_tokens / n_words, 2)


def compare_tokenizers(text: str, tokenizers: dict) -> list[dict]:
    """Compara varios tokenizadores sobre el mismo texto.

    tokenizers: {"nombre": tokenizador, ...}. Puede incluir tokenizadores
    de Hugging Face (p. ej. AutoTokenizer.from_pretrained("gpt2")).
    """
    rows = []
    for name, tok in tokenizers.items():
        if hasattr(tok, "tokens"):
            n = len(tok.encode(text))
        else:
            n = len(tok.encode(text, add_special_tokens=False))
        rows.append({"tokenizador": name, "tokens": n,
                     "tokens_por_palabra": round(n / max(1, len(text.split())), 2)})
    for r in rows:
        print(f"{r['tokenizador']:>25}: {r['tokens']:>6} tokens  ({r['tokens_por_palabra']} por palabra)")
    return rows


def show_vocabulary_sample(tokenizer: SimpleTokenizer, n: int = 40, longest: bool = True) -> list[str]:
    """Muestra algunos tokens del vocabulario (por defecto, los más largos)."""
    vocab = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]
    vocab = [v for v in vocab if v.strip() and v not in SPECIAL_TOKENS]
    sample = sorted(vocab, key=len, reverse=True)[:n] if longest else vocab[-n:]
    print(" · ".join(repr(s) for s in sample))
    return sample
