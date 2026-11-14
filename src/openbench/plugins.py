"""Plugin discovery for metrics, model providers and dataset sources (v0.5.0).

Third-party packages extend OpenBench by declaring Python entry points; installing the package
is enough, with no code changes in OpenBench:

.. code-block:: toml

    [project.entry-points."openbench.metrics"]
    my_metric = "my_package.metrics:MyMetric"          # a BaseMetric subclass

    [project.entry-points."openbench.providers"]
    my_server = "my_package.adapters:MyAdapter"        # a BaseModelAdapter subclass

    [project.entry-points."openbench.datasets"]
    my_source = "my_package.datasets:load"             # callable(argument: str) -> Dataset

Plugins are loaded lazily: only when a name is not built in, or when a listing command needs
them. A plugin that fails to import raises :class:`~openbench.exceptions.PluginError` naming the
plugin and its package. Set ``OPENBENCH_DISABLE_PLUGINS=1`` to turn discovery off.

Plugins are trusted code: installing one runs its code with your permissions.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from openbench.exceptions import PluginError

PLUGIN_GROUPS: dict[str, str] = {
    "metrics": "openbench.metrics",
    "providers": "openbench.providers",
    "datasets": "openbench.datasets",
}

#: Attribute set on plugin-provided classes; read by provenance and listing code.
PLUGIN_SOURCE_ATTR = "_openbench_plugin_source"


@dataclass(frozen=True)
class PluginInfo:
    """Where a plugin-provided component came from."""

    kind: str
    name: str
    entry_point: str
    distribution: str | None
    version: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "entry_point": self.entry_point,
            "distribution": self.distribution,
            "version": self.version,
        }


#: Dataset sources registered by plugins: name -> callable(argument) -> Dataset.
DATASET_SOURCES: dict[str, Callable[[str], Any]] = {}

_LOADED: dict[str, list[PluginInfo]] = {}
_LOCK = threading.RLock()


def plugins_disabled() -> bool:
    return os.environ.get("OPENBENCH_DISABLE_PLUGINS", "").strip().lower() in {"1", "true", "yes"}


def _entry_points(group: str) -> list[metadata.EntryPoint]:
    try:
        return list(metadata.entry_points(group=group))
    except TypeError:  # pragma: no cover - Python < 3.10 API, kept defensively
        return list(metadata.entry_points().get(group, []))  # type: ignore[attr-defined]


def _info_for(kind: str, ep: metadata.EntryPoint) -> PluginInfo:
    dist = getattr(ep, "dist", None)
    dist_name = getattr(dist, "name", None) if dist is not None else None
    if dist_name is None and dist is not None:
        try:
            dist_name = dist.metadata["Name"]
        except Exception:  # pragma: no cover
            dist_name = None
    version = getattr(dist, "version", None) if dist is not None else None
    return PluginInfo(kind=kind, name=ep.name, entry_point=ep.value, distribution=dist_name, version=version)


def _register(kind: str, info: PluginInfo, obj: Any) -> None:
    label = f"{kind} plugin '{info.name}' ({info.entry_point}"
    label += f", package {info.distribution} {info.version})" if info.distribution else ")"
    if kind == "metrics":
        from openbench.metrics import register_metric
        from openbench.metrics.base import BaseMetric

        if not (isinstance(obj, type) and issubclass(obj, BaseMetric)):
            raise PluginError(f"The {label} does not point to a BaseMetric subclass.")
        setattr(obj, PLUGIN_SOURCE_ATTR, info.to_dict())
        register_metric(obj)
    elif kind == "providers":
        from openbench.adapters.base import BaseModelAdapter, register_provider

        if not (isinstance(obj, type) and issubclass(obj, BaseModelAdapter)):
            raise PluginError(f"The {label} does not point to a BaseModelAdapter subclass.")
        setattr(obj, PLUGIN_SOURCE_ATTR, info.to_dict())
        register_provider(obj, aliases=[info.name] if info.name.lower() != obj.provider_name.lower() else None)
    elif kind == "datasets":
        if not callable(obj):
            raise PluginError(f"The {label} is not callable; dataset plugins must be callable(argument) -> Dataset.")
        DATASET_SOURCES[info.name.strip().lower()] = obj
    else:  # pragma: no cover - guarded by PLUGIN_GROUPS
        raise PluginError(f"Unknown plugin kind '{kind}'")


def load_plugins(kind: str | None = None) -> list[PluginInfo]:
    """Discover and register installed plugins; return what was loaded.

    Each group is loaded at most once per process. ``kind`` is ``"metrics"``, ``"providers"``,
    ``"datasets"`` or ``None`` for all.
    """
    if plugins_disabled():
        return []
    kinds = [kind] if kind else list(PLUGIN_GROUPS)
    loaded: list[PluginInfo] = []
    with _LOCK:
        for k in kinds:
            if k not in PLUGIN_GROUPS:
                raise PluginError(f"Unknown plugin kind '{k}'. Use one of: {', '.join(PLUGIN_GROUPS)}")
            if k in _LOADED:
                loaded.extend(_LOADED[k])
                continue
            infos: list[PluginInfo] = []
            for ep in sorted(_entry_points(PLUGIN_GROUPS[k]), key=lambda e: e.name):
                info = _info_for(k, ep)
                try:
                    obj = ep.load()
                except Exception as exc:
                    raise PluginError(
                        f"Could not load {k} plugin '{ep.name}' ({ep.value}"
                        + (f", package {info.distribution}" if info.distribution else "")
                        + f"): {type(exc).__name__}: {exc}"
                    ) from exc
                _register(k, info, obj)
                infos.append(info)
            _LOADED[k] = infos
            loaded.extend(infos)
    return loaded


def reset_plugins_for_tests() -> None:
    """Forget which groups were loaded (tests only; registered classes stay registered)."""
    with _LOCK:
        _LOADED.clear()
        DATASET_SOURCES.clear()


def plugin_source(obj: Any) -> dict[str, Any] | None:
    """Return the plugin provenance recorded on a class or instance, or ``None`` if built in."""
    src = getattr(obj, PLUGIN_SOURCE_ATTR, None)
    if src is None and not isinstance(obj, type):
        src = getattr(type(obj), PLUGIN_SOURCE_ATTR, None)
    return dict(src) if isinstance(src, dict) else None


def load_dataset_from_plugin(spec: str) -> Any:
    """Load a dataset from a ``plugin:<name>[:<argument>]`` spec."""
    if not spec.startswith("plugin:"):
        raise PluginError(f"Not a plugin dataset spec: {spec!r}")
    rest = spec[len("plugin:"):]
    name, _, argument = rest.partition(":")
    key = name.strip().lower()
    if not key:
        raise PluginError("Plugin dataset spec must name a source, e.g. 'plugin:my_source:argument'")
    load_plugins("datasets")
    if key not in DATASET_SOURCES:
        available = ", ".join(sorted(DATASET_SOURCES)) or "none installed"
        raise PluginError(f"Unknown dataset plugin '{name}'. Available: {available}")
    from openbench.dataset import Dataset

    result = DATASET_SOURCES[key](argument)
    if not isinstance(result, Dataset):
        raise PluginError(f"Dataset plugin '{name}' returned {type(result).__name__}, not a Dataset.")
    return result
