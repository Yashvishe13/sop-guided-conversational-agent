"""Every module imports cleanly (catches stale import paths after a file is moved)."""

import importlib
import pkgutil

import pytest

import insurance_claims

MODULES = sorted(m.name for m in pkgutil.walk_packages(insurance_claims.__path__, prefix="insurance_claims."))


@pytest.mark.parametrize("name", [*MODULES, "evals.live_eval"])
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
