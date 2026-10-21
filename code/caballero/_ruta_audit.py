"""Deja ``code/audit`` importable desde ``code/caballero``.

Los módulos de este directorio dependen de dos archivos que viven en
``code/audit`` y que no se mueven de ahí porque preservan la evidencia de la
auditoría, a saber ``venegas_model.py``, el motor original transcrito del
notebook de la tesis que se continúa, y ``metricas_tesis.py``, la capa de
métricas recuperada por ingeniería inversa.

Importar este módulo tiene como único efecto agregar ``code/audit`` a
``sys.path``, resolviendo la ruta desde la ubicación del propio archivo y no
desde el directorio de trabajo. Por eso los archivos de esta carpeta corren
sin necesidad de ``PYTHONPATH``, tanto si se ejecutan desde la raíz del
repositorio como desde su propia carpeta.

Uso, siempre antes de importar nada de ``code/audit``

    import _ruta_audit  # noqa: F401
    from venegas_model import run_price_dynamics

Es el único lugar donde se define esa ruta. Si la estructura de carpetas
cambia, se corrige aquí.
"""

import sys
from pathlib import Path

_AUDIT = Path(__file__).resolve().parent.parent / "audit"

if not _AUDIT.is_dir():                                     # pragma: no cover
    raise ImportError(
        f"No se encontro el directorio de la auditoria en {_AUDIT}. "
        "Se espera la estructura code/audit y code/caballero como carpetas "
        "hermanas dentro del repositorio."
    )

if str(_AUDIT) not in sys.path:
    sys.path.insert(0, str(_AUDIT))
