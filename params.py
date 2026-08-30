"""Safe Python wrapper around the native GlottDNN Param object."""

import numpy as np

import glottdnn_cpp


_PARAM_MEMBER_NAMES = tuple(
    name
    for name in dir(glottdnn_cpp.Param)
    if not name.startswith("_") and not callable(getattr(glottdnn_cpp.Param, name, None))
)


class ParamWrapper:
    """Python-side wrapper around a native glottdnn_cpp.Param object."""

    def __init__(self, native):
        if not isinstance(native, glottdnn_cpp.Param):
            raise TypeError("native must be a glottdnn_cpp.Param")
        object.__setattr__(self, "_native", native)

    @property
    def native(self):
        return self._native

    @staticmethod
    def _coerce_value(name, value):
        current = getattr(glottdnn_cpp.Param(), name)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(current, bool):
            return bool(value)
        if isinstance(current, int):
            return int(value)
        if isinstance(current, float):
            return float(value)
        if isinstance(current, str):
            return str(value)
        return value

    def as_native(self):
        return self._native

    def member_names(self):
        return list(_PARAM_MEMBER_NAMES)

    def members(self):
        return {name: getattr(self, name) for name in self.member_names()}

    def keys(self):
        return self.member_names()

    def items(self):
        return list(self.members().items())

    def __contains__(self, name):
        return name in self.member_names()

    def __iter__(self):
        return iter(self.member_names())

    def __getattr__(self, name):
        if name not in self.member_names():
            raise AttributeError("Param has no member '{}'".format(name))
        return getattr(self._native, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self.member_names():
            object.__setattr__(self, name, value)
            return
        setattr(self._native, name, self._coerce_value(name, value))

    def __setitem__(self, name, value):
        self.__setattr__(name, value)

    def __getitem__(self, name):
        return getattr(self, name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self.member_names()))

    def __repr__(self):
        fields = []
        for name in self.member_names():
            value = getattr(self._native, name)
            if isinstance(value, bool):
                rendered = "True" if value else "False"
            elif isinstance(value, str):
                rendered = repr(value)
            else:
                rendered = str(value)
            fields.append("{}={}".format(name, rendered))
        return "Param({})".format(", ".join(fields))

    def __str__(self):
        return repr(self)


def load_config(default_config, user_config=""):
    """Load a mutable native configuration object wrapped in a safe Python proxy."""
    if isinstance(default_config, ParamWrapper):
        params = default_config
    elif isinstance(default_config, glottdnn_cpp.Param):
        params = ParamWrapper(default_config)
    else:
        params = ParamWrapper(glottdnn_cpp.analysis.load_params(default_config, user_config))
    return params


def _unwrap_params(value):
    if isinstance(value, ParamWrapper):
        return value.as_native()
    if isinstance(value, glottdnn_cpp.Param):
        return value
    raise TypeError("expected a Param or ParamWrapper")


def _resolve_params(default_config, user_config="", verbose=None):
    """Resolve a Param object and optionally toggle native progress output."""
    if isinstance(default_config, ParamWrapper):
        params = default_config
    elif isinstance(default_config, glottdnn_cpp.Param):
        params = ParamWrapper(default_config)
    else:
        params = ParamWrapper(glottdnn_cpp.analysis.load_params(default_config, user_config))
    if verbose is not None:
        params.verbose = bool(verbose)
    return params
