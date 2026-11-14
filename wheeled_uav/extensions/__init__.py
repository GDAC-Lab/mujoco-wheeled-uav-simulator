"""Scene extensions: Python objects that add to the model and take part in every physics step.

An extension adds what the simulator itself does not model: bodies, joints, sites and the forces
on them, the state it reports and the commands it takes. Extensions are declared by name in the
``extensions`` section of the params file::

    "extensions": {
      "goal": {
        "class": "wheeled_uav.extensions.marker:Marker",
        "position": [1.0, 0.0, 1.2]
      }
    }

``class`` names the class as ``module:ClassName`` (in-process callers may give the class itself),
``path`` (optional) is a directory or a list of directories put on ``sys.path`` before the import
(relative paths are taken from the params file's directory), and ``"enabled": false`` leaves the entry
out. The other keys are the extension's own configuration. Names and keys starting with ``_`` are
comments, as elsewhere in the params. A params file that names a class runs that class's code: load
params files only from sources you trust, as you would a Python script.

The simulator calls, in this order (docs/EXTENSIONS.md):

    Class(name, config)            when the scene is built; config: the entry without class, path, enabled
    edit_spec(spec, context)       on the MjSpec of the rendered model, before it is compiled
    bind(model)                    on the compiled model: look up the ids of what it added
    reset(model, data)             on the fresh data (spawn state, initial velocities set), before mj_forward
    apply_command(command)         with the ``extensions[name]`` value of each command that is applied
    before_step(model, data)       every physics step, after the rotor and wrench inputs, before mj_step
    state(model, data)             for each state packet: a JSON-ready dict (or None) sent as extensions[name]

Every method is optional; :class:`Extension` provides no-op defaults (its apply_command rejects any
command). The scene is compiled from the edited spec, so the vehicles' part of the model is exactly the
one without extensions; the edited model is also written as ``<model>.extended.xml`` next to the
generated XML, for viewing. Without an ``extensions`` section nothing changes.
"""

from __future__ import annotations

import copy
import importlib
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

__all__ = [
    "Extension",
    "ExtensionContext",
    "ExtensionError",
    "SceneExtensions",
    "VehicleInfo",
    "apply_spec_edits",
    "load_extensions",
    "to_jsonable",
]

# Keys of an entry that the loader reads; the rest is the extension's configuration.
LOADER_KEYS = ("class", "path", "enabled")
# Names become JSON keys, MATLAB struct fields and prefixes of model element names.
_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,62}$")
# The calls made for every extension that defines them (apply_command is looked up per command).
_HOOKS = ("edit_spec", "bind", "reset", "before_step", "state")


class ExtensionError(ValueError):
    """An extension entry, configuration or command that cannot be used."""


@dataclass(frozen=True)
class VehicleInfo:
    """A vehicle of the scene: its position in the vehicle list, its name in state packets, its body."""

    index: int
    name: str
    body: str


@dataclass(frozen=True)
class ExtensionContext:
    """What edit_spec may need to know about the scene."""

    vehicles: tuple[VehicleInfo, ...]
    # The physics timestep [s] (also spec.option.timestep).
    timestep: float
    # The run's params, after randomization (read them; do not change them).
    params: Mapping[str, Any] = field(repr=False)
    # The directory of the params file, for files an extension names relative to it.
    params_dir: Path
    # The run's seed (simulation.seed or --seed), for extensions that draw values of their own.
    seed: int = 0


class Extension:
    """Base class with no-op defaults: an extension overrides only what it needs.

    ``keys`` lists the configuration keys the extension accepts; any other key is an error, so that a
    misspelt key cannot pass unnoticed. Leave it as None to accept any key.
    """

    keys: frozenset[str] | None = None

    def __init__(self, name: str, config: Mapping[str, Any]):
        self.name = name
        self.config = dict(config)
        if self.keys is not None:
            unknown = sorted(set(self.config) - set(self.keys))
            if unknown:
                raise ExtensionError(f"extensions.{name}: unknown keys {unknown}; known: {sorted(self.keys)}")

    def edit_spec(self, spec: mujoco.MjSpec, context: ExtensionContext) -> None:
        pass

    def bind(self, model: mujoco.MjModel) -> None:
        pass

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pass

    def apply_command(self, command: Any) -> None:
        raise ExtensionError(f"extension {self.name!r} takes no commands")

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pass

    def state(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any] | None:
        return None


def to_jsonable(value: Any) -> Any:
    """``value`` with numpy arrays and numbers turned into plain lists and numbers (for json.dumps)."""
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _overrides(instance: Any, hook: str) -> bool:
    """Whether ``instance`` has ``hook``, other than Extension's no-op default (which need not be called)."""
    method = getattr(type(instance), hook, None)
    if method is None:
        return callable(getattr(instance, hook, None))
    return method is not getattr(Extension, hook)


class SceneExtensions:
    """The extensions of one scene, in the order of the params file, and the calls the runtime makes."""

    def __init__(self, entries: Iterable[tuple[str, Any]]):
        self._entries: dict[str, Any] = {}
        for name, instance in entries:
            if name in self._entries:
                raise ExtensionError(f"extensions.{name}: given twice")
            self._entries[name] = instance
        self._hooks = {
            hook: [(name, getattr(instance, hook)) for name, instance in self._entries.items() if _overrides(instance, hook)]
            for hook in _HOOKS
        }
        self._warned: set[tuple[str, ...]] = set()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __getitem__(self, name: str) -> Any:
        return self._entries[name]

    def items(self):
        return self._entries.items()

    @property
    def steps(self) -> bool:
        """Whether any extension has a before_step."""
        return bool(self._hooks["before_step"])

    def edit_spec(self, spec: mujoco.MjSpec, context: ExtensionContext) -> None:
        for name, method in self._hooks["edit_spec"]:
            try:
                method(spec, context)
            except ExtensionError:
                raise
            except ValueError as error:  # MjSpec's own errors (a repeated name, a bad value)
                raise ExtensionError(f"extensions.{name}: editing the model: {error}") from error

    def bind(self, model: mujoco.MjModel) -> None:
        for _, method in self._hooks["bind"]:
            method(model)

    def reset(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for _, method in self._hooks["reset"]:
            method(model, data)

    def before_step(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        for _, method in self._hooks["before_step"]:
            method(model, data)

    def states(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
        """{name: state} of the extensions that report one (JSON-ready)."""
        states: dict[str, Any] = {}
        for name, method in self._hooks["state"]:
            value = method(model, data)
            if value is not None:
                states[name] = to_jsonable(value)
        return states

    def apply_commands(self, commands: Mapping[str, Any], *, strict: bool = True) -> None:
        """Hand each extension its command. ``strict``: raise on a command that cannot be used
        (in-process callers); otherwise warn once per problem and ignore it (commands from the network)."""
        for name, command in commands.items():
            if name not in self._entries:
                self._reject(strict, ("unknown", str(name)), f"no extension {name!r} in this scene (extensions: {list(self._entries)})")
                continue
            method = getattr(self._entries[name], "apply_command", None)
            if method is None:
                self._reject(strict, ("no-commands", name), f"extension {name!r} takes no commands")
                continue
            try:
                method(command)
            except (ValueError, TypeError, KeyError) as error:
                if strict:
                    raise
                self._reject(False, ("error", name, str(error)), f"the command for extension {name!r}: {error}")

    def _reject(self, strict: bool, key: tuple[str, ...], message: str) -> None:
        if strict:
            raise ExtensionError(message)
        if key not in self._warned:
            self._warned.add(key)
            print(f"warning: ignoring {message}")


def _search_paths(entry: Mapping[str, Any], name: str, params_dir: Path) -> list[Path]:
    paths = entry.get("path", [])
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not isinstance(paths, list) or not all(isinstance(path, (str, Path)) for path in paths):
        raise ExtensionError(f"extensions.{name}.path must be a directory or a list of directories")
    resolved = []
    for path in paths:
        directory = Path(path)
        directory = directory if directory.is_absolute() else (params_dir / directory)
        if not directory.is_dir():
            raise ExtensionError(f"extensions.{name}.path: {directory} is not a directory")
        resolved.append(directory.resolve())
    return resolved


def _resolve_class(spec: Any, name: str, search_paths: list[Path]) -> Any:
    if not isinstance(spec, str):
        if callable(spec):
            return spec
        raise ExtensionError(f"extensions.{name}.class must be 'module:ClassName', got {spec!r}")
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ExtensionError(f"extensions.{name}.class must be 'module:ClassName', got {spec!r}")
    for directory in reversed(search_paths):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as error:
        raise ExtensionError(f"extensions.{name}: cannot import {module_name!r} ({error})") from error
    for part in attribute.split("."):
        try:
            target = getattr(target, part)
        except AttributeError:
            raise ExtensionError(f"extensions.{name}: {module_name!r} has no {attribute!r}") from None
    if not callable(target):
        raise ExtensionError(f"extensions.{name}: {spec!r} is not a class")
    return target


def load_extensions(params: Mapping[str, Any], params_dir: Path) -> SceneExtensions | None:
    """The extensions the params declare, constructed (None: no ``extensions`` section, or none enabled)."""
    section = params.get("extensions")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ExtensionError("extensions must be an object: {name: {class: 'module:ClassName', ...}}")
    entries = []
    for name, entry in section.items():
        if isinstance(name, str) and name.startswith("_"):
            continue  # a comment
        if not isinstance(name, str) or not _NAME_PATTERN.match(name):
            raise ExtensionError(
                f"extension name {name!r}: use a letter, then letters, digits or '_' (at most 63 characters)"
            )
        if not isinstance(entry, Mapping):
            raise ExtensionError(f"extensions.{name} must be an object")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ExtensionError(f"extensions.{name}.enabled must be true or false")
        if not enabled:
            continue
        if "class" not in entry:
            raise ExtensionError(f"extensions.{name}: no class ('module:ClassName')")
        factory = _resolve_class(entry["class"], name, _search_paths(entry, name, params_dir))
        config = {
            key: copy.deepcopy(value)
            for key, value in entry.items()
            if key not in LOADER_KEYS and not (isinstance(key, str) and key.startswith("_"))
        }
        entries.append((name, factory(name, config)))
    return SceneExtensions(entries) if entries else None


def apply_spec_edits(
    extensions: SceneExtensions,
    xml_path: Path,
    *,
    vehicles: Iterable[VehicleInfo],
    params: Mapping[str, Any],
    params_dir: Path,
    seed: int = 0,
    output_path: Path | None = None,
) -> mujoco.MjSpec:
    """Let the extensions edit the model of ``xml_path``; return the edited spec.

    The scene is compiled from the returned spec, so the vehicles' part of the model is exactly the one
    of ``xml_path``. ``output_path``, when given, receives MuJoCo's XML of the edited model, for viewing
    and checking (XML writes some numbers rounded, so a model loaded from it can differ in the last digits).
    """
    spec = mujoco.MjSpec.from_file(str(xml_path))
    context = ExtensionContext(
        vehicles=tuple(vehicles),
        timestep=float(spec.option.timestep),
        params=params,
        params_dir=Path(params_dir),
        seed=int(seed),
    )
    actuators = len(spec.actuators)
    extensions.edit_spec(spec, context)
    # The rotor commands own data.ctrl (every actuator, in the vehicles' order) and the timing
    # follows the timestep: an extension acts through its own bodies, joints and forces instead.
    if len(spec.actuators) != actuators:
        raise ExtensionError(
            f"extensions ({', '.join(extensions.names)}) must not add actuators: the rotor commands own data.ctrl; "
            "apply forces in before_step (xfrc_applied of the extension's bodies, qfrc_applied of its joints)"
        )
    if float(spec.option.timestep) != context.timestep:
        raise ExtensionError(f"extensions ({', '.join(extensions.names)}) must not change the timestep (simulation.timestep)")
    try:
        spec.compile()
    except ValueError as error:
        raise ExtensionError(f"the model does not compile after the extensions' edits ({', '.join(extensions.names)}): {error}") from error
    if output_path is not None:
        Path(output_path).write_text(spec.to_xml(), encoding="utf-8")
    return spec
