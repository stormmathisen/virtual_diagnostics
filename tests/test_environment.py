"""Guard against an environment silently losing an optional dependency.

Every optional integration in this package is skipped when its dependency is
missing, which is right for a developer machine and dangerous in CI: a runner
without ngspice or Cheetah quietly reports success while a fifth of the suite
never runs.

These tests turn that skip into a failure when ``VD_REQUIRE_ALL_EXTRAS`` is set,
which CI does. Run them locally the same way to check your own environment::

    VD_REQUIRE_ALL_EXTRAS=1 pytest tests/test_environment.py -v
"""

import os

import pytest

from virtual_diagnostics import ngspice_available, ngspice_version

pytestmark = pytest.mark.skipif(
    not os.environ.get("VD_REQUIRE_ALL_EXTRAS"),
    reason="set VD_REQUIRE_ALL_EXTRAS=1 to require every optional dependency",
)


def test_ngspice_is_installed():
    assert ngspice_available(), (
        "ngspice is missing, so every SPICE test would silently skip. "
        "Install it with 'apt install ngspice', or unset VD_REQUIRE_ALL_EXTRAS."
    )
    assert "ngspice" in (ngspice_version() or "").lower()


def test_cheetah_is_installed():
    import cheetah  # noqa: F401

    from virtual_diagnostics import from_cheetah  # noqa: F401


def test_h5py_is_installed():
    import h5py  # noqa: F401
