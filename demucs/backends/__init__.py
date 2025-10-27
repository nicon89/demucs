"""Separator backend registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping

BackendFactory = Callable[..., "SeparatorLike"]


class UnknownBackendError(RuntimeError):
    """Raised when attempting to use an unknown separator backend."""


class SeparatorLike:
    """Protocol-like base class for type checking.

    Every backend must expose the methods provided by :class:`demucs.api.Separator`.
    The class intentionally keeps the surface minimal to avoid a hard dependency on the
    concrete implementation from the Demucs wrapper while still enabling static tooling
    to reason about the behaviour.  Backends may freely inherit from the Demucs
    ``Separator`` class or implement the methods manually.
    """

    def separate_tensor(self, wav, sr=None):  # pragma: no cover - protocol like behaviour
        raise NotImplementedError

    def separate_audio_file(self, file):  # pragma: no cover - protocol like behaviour
        raise NotImplementedError

    @property
    def samplerate(self):  # pragma: no cover - protocol like behaviour
        raise NotImplementedError

    @property
    def audio_channels(self):  # pragma: no cover - protocol like behaviour
        raise NotImplementedError

    @property
    def model(self):  # pragma: no cover - protocol like behaviour
        raise NotImplementedError


@dataclass
class _BackendDefinition:
    name: str
    factory: BackendFactory
    description: str


class _BackendRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, _BackendDefinition] = {}

    def register(self, name: str, description: str) -> Callable[[BackendFactory], BackendFactory]:
        normalized = name.lower()

        def _decorator(factory: BackendFactory) -> BackendFactory:
            if normalized in self._registry:
                raise ValueError(f"Backend '{name}' already registered")
            self._registry[normalized] = _BackendDefinition(
                name=normalized, factory=factory, description=description
            )
            return factory

        return _decorator

    def create(self, name: str, **kwargs) -> SeparatorLike:
        normalized = name.lower()
        try:
            definition = self._registry[normalized]
        except KeyError as exc:  # pragma: no cover - defensive branch
            available = ", ".join(sorted(self._registry)) or "<none>"
            raise UnknownBackendError(
                f"Unknown separator backend '{name}'. Available backends: {available}."
            ) from exc
        return definition.factory(**kwargs)

    def names(self) -> Iterable[str]:
        return sorted(self._registry.keys())

    def describe(self) -> Mapping[str, str]:
        return {name: definition.description for name, definition in self._registry.items()}


_REGISTRY = _BackendRegistry()


def register_backend(name: str, description: str) -> Callable[[BackendFactory], BackendFactory]:
    """Decorator used by backend implementations to register themselves."""

    return _REGISTRY.register(name=name, description=description)


def create_separator(name: str, **kwargs) -> SeparatorLike:
    """Instantiate the separator backend referenced by ``name``."""

    return _REGISTRY.create(name=name, **kwargs)


def available_backends() -> Iterable[str]:
    """Return the names of registered backends."""

    return _REGISTRY.names()


def describe_backends() -> Mapping[str, str]:
    """Return a mapping of backend names to human readable descriptions."""

    return _REGISTRY.describe()


# Import built-in backends so that they register themselves when the package is
# imported.  The imports are intentionally placed at the end of the module to
# avoid circular dependencies during interpreter start-up.
from . import bs_roformer_backend as _bs_roformer_backend  # noqa: F401
from . import demucs_backend as _demucs_backend  # noqa: F401
