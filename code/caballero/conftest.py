"""Deja ``code/audit`` importable durante la recolección de pytest.

pytest carga este archivo antes de recolectar las pruebas de la carpeta, de
modo que las baterías corren desde cualquier directorio de trabajo. La ruta
se define en un solo lugar, ver ``_ruta_audit``.
"""

import _ruta_audit  # noqa: F401
