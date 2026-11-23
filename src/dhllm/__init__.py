"""dhllm: herramientas sencillas para el taller «LLM para las Humanidades Digitales».

La idea es que en los cuadernos (notebooks) solo haya que llamar a funciones
de alto nivel. Quien tenga curiosidad puede leer el código de este paquete:
está comentado en español y evita trucos innecesarios.
"""

__version__ = "0.1.0"

from .utils import get_device, set_seed, describe_environment  # noqa: F401
from .config import load_config  # noqa: F401
