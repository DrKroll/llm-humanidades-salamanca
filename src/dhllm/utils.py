"""Utilidades generales: dispositivo (CPU/GPU), semillas, entorno."""

import os
import random
import platform

import numpy as np
import torch


def in_colab() -> bool:
    """Devuelve True si estamos dentro de Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def smoke_test_mode() -> bool:
    """Modo de prueba rápida (solo para comprobar que todo funciona).

    Se activa con la variable de entorno DHLLM_SMOKE_TEST=1. Reduce mucho
    los pasos de entrenamiento. Las personas participantes no lo necesitan.
    """
    return os.environ.get("DHLLM_SMOKE_TEST", "0") == "1"


def get_device(prefer: str = "auto") -> str:
    """Elige el mejor dispositivo disponible: 'cuda' (GPU), 'mps' (Mac) o 'cpu'."""
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Fija la «semilla» aleatoria para que los resultados se puedan repetir."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def describe_environment() -> dict:
    """Muestra un resumen del entorno (Python, PyTorch, GPU...)."""
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dispositivo": get_device(),
        "colab": in_colab(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["memoria_gpu_GB"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    for name, label in [("transformers", "transformers"), ("tokenizers", "tokenizers"), ("peft", "peft")]:
        try:
            mod = __import__(name)
            info[label] = mod.__version__
        except ImportError:
            info[label] = "no instalado"
    print("Entorno de trabajo")
    print("-" * 30)
    for k, v in info.items():
        print(f"{k:>16}: {v}")
    if info["dispositivo"] == "cpu":
        print("\nAviso: no hay GPU. Todo funciona, pero el entrenamiento será lento.")
        print("En Colab: Entorno de ejecución > Cambiar tipo de entorno > GPU T4.")
    return info


def count_parameters(model) -> int:
    """Número de parámetros (los «números» que el modelo aprende)."""
    return sum(p.numel() for p in model.parameters())


def human_number(n: int) -> str:
    """1234567 -> '1,2 M' (formato fácil de leer)."""
    for unit, div in [("mil M", 1e9), ("M", 1e6), ("mil", 1e3)]:
        if n >= div:
            return f"{n / div:.1f} {unit}".replace(".", ",")
    return str(n)
