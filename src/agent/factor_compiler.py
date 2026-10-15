"""AST inspection, restricted loading and output validation for generated alpha factors."""

from __future__ import annotations

import ast
import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.factors.base import AbstractFactor

ALLOWED_MODULES: frozenset[str] = frozenset({"pandas", "numpy", "math", "src.factors.base"})
_EXACT_ONLY: frozenset[str] = frozenset({"src.factors.base"})

FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars",
        "getattr", "setattr", "delattr", "input", "breakpoint", "memoryview", "type",
    }
)
# I/O entry points reachable through the allowed pandas / numpy modules.
FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "to_csv", "to_pickle", "to_parquet", "to_excel", "to_json", "to_hdf", "to_sql",
        "to_feather", "to_clipboard", "to_html", "to_latex", "to_markdown", "to_stata",
        "to_orc", "to_xml", "load", "save", "savez", "savetxt", "loadtxt", "genfromtxt",
        "fromfile", "tofile", "memmap", "ctypeslib", "testing", "eval", "query",
    }
)
_SAFE_BUILTINS: tuple[str, ...] = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "object", "pow", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "super", "tuple", "zip",
    "property", "staticmethod", "classmethod", "ValueError", "TypeError", "KeyError",
    "NotImplementedError", "Exception", "True", "False", "None",
)
_ALLOWED_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.Assign, ast.AnnAssign)


class FactorCompileError(Exception):
    """Raised when generated code fails inspection or cannot be loaded."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


def _module_allowed(name: str) -> bool:
    if name in _EXACT_ONLY:
        return True
    return any(name == root or name.startswith(root + ".") for root in ALLOWED_MODULES - _EXACT_ONLY)


@dataclass(frozen=True)
class InspectionReport:
    errors: list[str]
    class_name: str | None

    @property
    def ok(self) -> bool:
        return not self.errors


class FactorCodeInspector(ast.NodeVisitor):
    """Collects security and interface violations found in generated factor source."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def _fail(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if not _module_allowed(alias.name):
                self._fail(node, f"import of '{alias.name}' is not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        if node.level or not _module_allowed(module):
            self._fail(node, f"import from '{'.' * node.level}{module}' is not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            self._fail(node, f"call to '{node.func.id}' is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr in FORBIDDEN_ATTRS or attr.startswith("read_"):
            self._fail(node, f"access to attribute '{attr}' is forbidden")
        elif attr.startswith("__") and attr != "__init__":
            self._fail(node, f"access to dunder attribute '{attr}' is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("__") and node.id not in {"__init__"}:
            self._fail(node, f"use of dunder name '{node.id}' is forbidden")

    def visit_Global(self, node: ast.Global) -> None:
        self._fail(node, "global statements are forbidden")


def _is_abstract_factor_base(base: ast.expr) -> bool:
    return (isinstance(base, ast.Name) and base.id == "AbstractFactor") or (
        isinstance(base, ast.Attribute) and base.attr == "AbstractFactor"
    )


def _check_compute_signature(cls: ast.ClassDef, errors: list[str]) -> None:
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute"]
    if not methods:
        errors.append(f"class '{cls.name}' must override compute(self, prices: pd.DataFrame)")
        return
    fn = methods[0]
    names = [a.arg for a in fn.args.args]
    if names != ["self", "prices"] or fn.args.vararg or fn.args.kwonlyargs or fn.args.kwarg:
        errors.append("compute must have exactly the signature compute(self, prices)")
    prices_arg = next((a for a in fn.args.args if a.arg == "prices"), None)
    if prices_arg is None or prices_arg.annotation is None or "DataFrame" not in ast.unparse(prices_arg.annotation):
        errors.append("compute parameter 'prices' must be annotated as pd.DataFrame")
    if fn.returns is None or "DataFrame" not in ast.unparse(fn.returns):
        errors.append("compute must be annotated to return pd.DataFrame")


def inspect_factor_source(source: str) -> InspectionReport:
    """Statically check ``source`` without executing it."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return InspectionReport([f"syntax error at line {exc.lineno}: {exc.msg}"], None)

    inspector = FactorCodeInspector()
    inspector.visit(tree)
    errors = inspector.errors

    for node in tree.body:
        is_docstring = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (isinstance(node, _ALLOWED_TOP_LEVEL) or is_docstring):
            errors.append(f"line {node.lineno}: top-level {type(node).__name__} statements are not allowed")

    factor_classes = [
        n for n in tree.body if isinstance(n, ast.ClassDef) and any(_is_abstract_factor_base(b) for b in n.bases)
    ]
    class_name: str | None = None
    if len(factor_classes) != 1:
        errors.append(f"expected exactly one AbstractFactor subclass, found {len(factor_classes)}")
    else:
        class_name = factor_classes[0].name
        _check_compute_signature(factor_classes[0], errors)
    return InspectionReport(errors, class_name)


def _restricted_import(
    name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0
) -> Any:
    if level or not _module_allowed(name):
        raise ImportError(f"import of '{name}' is not allowed")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _restricted_namespace() -> dict[str, Any]:
    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTINS}
    safe["__build_class__"] = builtins.__build_class__
    safe["__import__"] = _restricted_import
    return {"__builtins__": safe, "__name__": "generated_factor"}


def compile_factor(source: str) -> type[AbstractFactor]:
    """Inspect ``source``, execute it in a restricted namespace and return the factor class."""
    report = inspect_factor_source(source)
    if not report.ok or report.class_name is None:
        raise FactorCompileError(report.errors)

    namespace = _restricted_namespace()
    try:
        exec(compile(source, "<generated_factor>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # generated code may raise anything at import time
        raise FactorCompileError([f"loading failed: {type(exc).__name__}: {exc}"]) from exc

    cls = namespace.get(report.class_name)
    if not (isinstance(cls, type) and issubclass(cls, AbstractFactor)):
        raise FactorCompileError([f"'{report.class_name}' does not subclass AbstractFactor"])
    return cls


def instantiate_factor(cls: type[AbstractFactor]) -> AbstractFactor:
    """Create the factor with no arguments (generated factors must be self-contained)."""
    try:
        factor = cls()  # type: ignore[call-arg]  # generated classes take no arguments
    except Exception as exc:
        raise FactorCompileError([f"factor must be constructible with no arguments: {exc}"]) from exc
    if not (isinstance(factor.name, str) and factor.name.isidentifier()):
        raise FactorCompileError(["factor.name must be a valid Python identifier string"])
    if not isinstance(factor.higher_is_better, bool):
        raise FactorCompileError(["factor.higher_is_better must be a bool"])
    return factor


def make_synthetic_prices(n_days: int = 400, n_assets: int = 6, seed: int = 7) -> pd.DataFrame:
    """Deterministic geometric-random-walk price matrix used for validation."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n_days, name="Date")
    steps = rng.normal(0.0004, 0.02, size=(n_days, n_assets))
    columns = [chr(ord("A") + i) for i in range(n_assets)]
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=index, columns=columns)


def validate_factor_output(factor: AbstractFactor, prices: pd.DataFrame) -> pd.DataFrame:
    """Run ``factor`` and verify the T x N contract; returns the computed matrix."""
    try:
        values = factor.compute(prices.copy())
    except Exception as exc:
        raise FactorCompileError([f"compute raised {type(exc).__name__}: {exc}"]) from exc

    errors: list[str] = []
    if not isinstance(values, pd.DataFrame):
        raise FactorCompileError([f"compute returned {type(values).__name__}, expected DataFrame"])
    if values.shape != prices.shape:
        errors.append(f"output shape {values.shape} != prices shape {prices.shape} (T x N)")
    else:
        if not (values.index.equals(prices.index) and values.columns.equals(prices.columns)):
            errors.append("output index/columns must match prices")
        all_nan = [str(c) for c in values.columns[values.isna().all()]]
        if all_nan:
            errors.append(f"all-NaN columns: {all_nan}")
        if np.isinf(values.to_numpy(dtype=float)).any():
            errors.append("output contains +/-inf")
    if errors:
        raise FactorCompileError(errors)
    return values


def check_no_lookahead(
    factory: Callable[[], AbstractFactor], prices: pd.DataFrame, cutoff: int | None = None
) -> None:
    """Factor values up to ``cutoff`` must not change when later prices are altered."""
    t = cutoff if cutoff is not None else int(len(prices) * 0.8)
    altered = prices.copy()
    future = altered.iloc[t + 1 :].to_numpy()
    altered.iloc[t + 1 :] = future * np.random.default_rng(1).uniform(0.2, 5.0, size=future.shape)

    base = factory().compute(prices.copy()).iloc[: t + 1]
    shifted = factory().compute(altered).iloc[: t + 1]
    try:
        pd.testing.assert_frame_equal(base, shifted, check_exact=True)
    except AssertionError as exc:
        raise FactorCompileError([f"look-ahead bias: values at or before row {t} depend on later prices"]) from exc
