"""Minimal torch-compatible API backed by numpy + ml_dtypes.

Drop-in replacement for torch in convert_hf_to_gguf.py, eliminating the
~574MB PyTorch dependency.  Only the subset of torch actually used by the
converter is implemented.

Requires: numpy, ml_dtypes (tiny)
"""
from __future__ import annotations

import contextlib
import math
import struct
import sys
from typing import Any, Sequence

import numpy as np

try:
    import ml_dtypes  # ~100KB, provides bfloat16 / float8
except ImportError:
    ml_dtypes = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# dtype shims – behave like torch.float32, torch.bfloat16, etc.
# ---------------------------------------------------------------------------

class _DType:
    """Thin wrapper around a numpy dtype so we can attach metadata."""
    __slots__ = ("np_dtype", "name", "itemsize")

    def __init__(self, np_dtype: np.dtype | type, name: str):
        self.np_dtype = np.dtype(np_dtype)
        self.name = name
        self.itemsize = self.np_dtype.itemsize

    def __repr__(self) -> str:
        return f"torch_compat.{self.name}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DType):
            return self.name == other.name
        if isinstance(other, np.dtype):
            return self.np_dtype == other
        return NotImplemented

    def __hash__(self) -> int:
        # Must match np.dtype hash so _DType keys work with np.dtype lookups
        return hash(self.np_dtype)


# Standard numeric types
float16  = _DType(np.float16,  "float16")
float32  = _DType(np.float32,  "float32")
float64  = _DType(np.float64,  "float64")
int8     = _DType(np.int8,     "int8")
int16    = _DType(np.int16,    "int16")
int32    = _DType(np.int32,    "int32")
int64    = _DType(np.int64,    "int64")
uint8    = _DType(np.uint8,    "uint8")
uint16   = _DType(np.uint16,   "uint16")
uint32   = _DType(np.uint32,   "uint32")
uint64   = _DType(np.uint64,   "uint64")
bool_    = _DType(np.bool_,    "bool")

# bfloat16 / float8 – require ml_dtypes
if ml_dtypes is not None:
    bfloat16     = _DType(ml_dtypes.bfloat16,     "bfloat16")
    float8_e4m3fn = _DType(ml_dtypes.float8_e4m3fn, "float8_e4m3fn")
    float8_e5m2   = _DType(ml_dtypes.float8_e5m2,   "float8_e5m2")
else:
    # Fallback: treat bfloat16 as uint16 (same byte width) for raw storage
    bfloat16      = _DType(np.uint16, "bfloat16")
    float8_e4m3fn = _DType(np.uint8,  "float8_e4m3fn")
    float8_e5m2   = _DType(np.uint8,  "float8_e5m2")

# Alias used by the script
bool = bool_  # type: ignore[assignment]  # shadows builtin, matching torch

# Reverse lookup: _DType -> np.dtype (used in Tensor internals)
_all_dtypes: dict[str, _DType] = {
    d.name: d for d in [
        float16, float32, float64,
        int8, int16, int32, int64,
        uint8, uint16, uint32, uint64,
        bool_, bfloat16, float8_e4m3fn, float8_e5m2,
    ]
}


def _resolve_dtype(dtype: _DType | np.dtype | type | None) -> np.dtype | None:
    """Convert our _DType or any dtype-like to a numpy dtype."""
    if dtype is None:
        return None
    if isinstance(dtype, _DType):
        return dtype.np_dtype
    return np.dtype(dtype)


def _wrap_dtype(np_dtype: np.dtype) -> _DType:
    """Convert a numpy dtype back to our _DType."""
    for d in _all_dtypes.values():
        if d.np_dtype == np_dtype:
            return d
    return _DType(np_dtype, str(np_dtype))


# ---------------------------------------------------------------------------
# Tensor – np.ndarray subclass with torch-like methods
# ---------------------------------------------------------------------------

class Tensor(np.ndarray):
    """numpy ndarray subclass that provides the torch.Tensor API surface used
    by convert_hf_to_gguf.py."""

    # ---- construction helpers ----

    def __new__(cls, data: Any = None, *, dtype: Any = None) -> "Tensor":
        np_dt = _resolve_dtype(dtype)
        if data is None:
            arr = np.empty(0, dtype=np_dt)
        else:
            arr = np.asarray(data, dtype=np_dt)
        return arr.view(cls)

    def __array_finalize__(self, obj: Any) -> None:
        pass

    # Note: we do NOT override the dtype property. numpy needs it to be
    # the real np.dtype for internal operations (view, etc.).
    # Comparison with our _DType objects works via _DType.__eq__.

    # ---- torch-style methods ----

    def float(self) -> "Tensor":
        return self.astype(np.float32).view(Tensor)

    def half(self) -> "Tensor":
        return self.astype(np.float16).view(Tensor)

    def to(self, dtype: Any = None, *, device: str | None = None) -> "Tensor":
        """Supports .to(dtype) and .to(device=...).  device is ignored."""
        if dtype is not None and not isinstance(dtype, str):
            np_dt = _resolve_dtype(dtype)
            return self.astype(np_dt).view(Tensor)
        return self.view(Tensor)

    def contiguous(self) -> "Tensor":
        return np.ascontiguousarray(self).view(Tensor)

    def numpy(self) -> np.ndarray:
        return np.asarray(self)

    def dim(self) -> int:
        return self.ndim

    def numel(self) -> int:
        return np.ndarray.size.__get__(self)  # access numpy's .size property

    def item(self) -> Any:
        return np.ndarray.item(self)

    def tolist(self) -> list:
        return np.ndarray.tolist(self)

    @property
    def device(self) -> str:
        return "cpu"

    # ---- view: dtype-reinterpret OR reshape ----

    def view(self, *args: Any) -> "Tensor":  # type: ignore[override]
        if len(args) == 1 and isinstance(args[0], type) and issubclass(args[0], np.ndarray):
            # Class-based view: e.g. arr.view(Tensor)
            return np.ndarray.view(self, args[0])
        if len(args) == 1 and isinstance(args[0], (_DType, np.dtype, type)):
            # Dtype reinterpret cast: e.g. tensor.view(torch.uint8)
            np_dt = _resolve_dtype(args[0])
            result = np.ndarray.view(self, np_dt)
            return np.ndarray.view(result, Tensor)
        # shape-based view → reshape (numpy's view is stricter, reshape is fine)
        shape = args[0] if len(args) == 1 and isinstance(args[0], tuple) else args
        return self.reshape(*shape)

    def unsqueeze(self, dim: int) -> "Tensor":
        return np.expand_dims(self, dim).view(Tensor)

    def squeeze(self, dim: int | None = None) -> "Tensor":
        if dim is None:
            return np.squeeze(self).view(Tensor)
        return np.squeeze(self, axis=dim).view(Tensor)

    def permute(self, *dims: int) -> "Tensor":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        return np.transpose(self, dims).view(Tensor)

    def transpose(self, dim0: int | None = None, dim1: int | None = None) -> "Tensor":
        if dim0 is None and dim1 is None:
            return super().transpose().view(Tensor)
        return np.swapaxes(self, dim0, dim1).view(Tensor)

    @property
    def T(self) -> "Tensor":
        return super().transpose().view(Tensor)

    def expand(self, *sizes: int | tuple) -> "Tensor":
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        return np.broadcast_to(self, sizes).view(Tensor)

    def repeat_interleave(self, repeats: int, dim: int | None = None) -> "Tensor":
        return np.repeat(self, repeats, axis=dim).view(Tensor)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if start_dim == 0 and end_dim == -1:
            return np.ndarray.flatten(self).view(Tensor)
        # Partial flatten
        shape = self.shape
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        new_shape = shape[:start_dim] + (-1,) + shape[end_dim + 1:]
        return self.reshape(new_shape).view(Tensor)

    def div_(self, value: Any) -> "Tensor":
        np.divide(self, value, out=self)
        return self

    def clamp_(self, min: float | None = None, max: float | None = None) -> "Tensor":
        np.clip(self, min, max, out=self)
        return self

    # Keep result as Tensor for chained ops
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Convert Tensor inputs to plain ndarray for ufunc, then re-wrap
        args = [np.asarray(x) if isinstance(x, Tensor) else x for x in inputs]
        out = kwargs.get("out", None)
        if out is not None:
            kwargs["out"] = tuple(np.asarray(o) if isinstance(o, Tensor) else o for o in out)
        result = getattr(ufunc, method)(*args, **kwargs)
        if isinstance(result, np.ndarray):
            return result.view(Tensor)
        return result

    def __array_function__(self, func, types, args, kwargs):
        # Unwrap Tensor → ndarray, call, re-wrap
        new_args = []
        for a in args:
            if isinstance(a, Tensor):
                new_args.append(np.asarray(a))
            elif isinstance(a, (list, tuple)):
                new_args.append(type(a)(np.asarray(x) if isinstance(x, Tensor) else x for x in a))
            else:
                new_args.append(a)
        result = func(*new_args, **kwargs)
        if isinstance(result, np.ndarray):
            return result.view(Tensor)
        return result


# ---------------------------------------------------------------------------
# Size – tuple subclass like torch.Size
# ---------------------------------------------------------------------------

class Size(tuple):
    pass


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def tensor(data: Any, dtype: Any = None) -> Tensor:
    np_dt = _resolve_dtype(dtype)
    return np.asarray(data, dtype=np_dt).view(Tensor)


def from_numpy(arr: np.ndarray) -> Tensor:
    return arr.view(Tensor)


def arange(*args, dtype: Any = None, **kwargs) -> Tensor:
    np_dt = _resolve_dtype(dtype)
    return np.arange(*args, dtype=np_dt, **kwargs).view(Tensor)


def zeros(*size, dtype: Any = None, device: str | None = None) -> Tensor:
    if len(size) == 1 and isinstance(size[0], (tuple, list)):
        size = tuple(size[0])
    np_dt = _resolve_dtype(dtype) or np.float32
    return np.zeros(size, dtype=np_dt).view(Tensor)


def empty(*, size: tuple | None = None, dtype: Any = None, device: str | None = None) -> Tensor:
    """Matches torch.empty(size=..., dtype=..., device=...).
    device='meta' returns a zero-stride array (minimal memory, correct shape)."""
    np_dt = _resolve_dtype(dtype) or np.float32
    if size is None:
        size = (0,)
    if device == "meta":
        # Use as_strided trick: 1 element of storage, but appears as full shape.
        # Same approach as LazyNumpyTensor.meta_with_dtype_and_shape.
        cheat = np.zeros(1, np_dt)
        return np.lib.stride_tricks.as_strided(cheat, size, tuple(0 for _ in size)).view(Tensor)
    return np.empty(size, dtype=np_dt).view(Tensor)


# ---------------------------------------------------------------------------
# Torch-style functions
# ---------------------------------------------------------------------------

def cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    arrays = [np.asarray(t) for t in tensors]
    return np.concatenate(arrays, axis=dim).view(Tensor)


def concat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    return cat(tensors, dim)


def stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    arrays = [np.asarray(t) for t in tensors]
    return np.stack(arrays, axis=dim).view(Tensor)


def split(tensor_in: Tensor, split_size_or_sections: int | list[int], dim: int = 0) -> list[Tensor]:
    arr = np.asarray(tensor_in)
    if isinstance(split_size_or_sections, int):
        # split into chunks of this size
        n = arr.shape[dim]
        indices = list(range(split_size_or_sections, n, split_size_or_sections))
    else:
        indices = []
        acc = 0
        for s in split_size_or_sections[:-1]:
            acc += s
            indices.append(acc)
    parts = np.split(arr, indices, axis=dim)
    return [p.view(Tensor) for p in parts]


def chunk(tensor_in: Tensor, chunks: int, dim: int = 0) -> list[Tensor]:
    parts = np.array_split(np.asarray(tensor_in), chunks, axis=dim)
    return [p.view(Tensor) for p in parts]


def sqrt(t: Tensor) -> Tensor:
    return np.sqrt(np.asarray(t)).view(Tensor)


def exp(t: Tensor) -> Tensor:
    return np.exp(np.asarray(t)).view(Tensor)


def sin(t: Tensor) -> Tensor:
    return np.sin(np.asarray(t)).view(Tensor)


def cos(t: Tensor) -> Tensor:
    return np.cos(np.asarray(t)).view(Tensor)


def norm(t: Tensor, p: float = 2, dim: int | None = None, keepdim: bool = False) -> Tensor:
    arr = np.asarray(t, dtype=np.float64)
    result = np.linalg.norm(arr, ord=p, axis=dim, keepdims=keepdim)
    return np.asarray(result).view(Tensor)


def equal(a: Tensor, b: Tensor) -> builtins_bool:
    return np.array_equal(np.asarray(a), np.asarray(b))


def bitwise_and(a: Any, b: Any) -> Tensor:
    return np.bitwise_and(np.asarray(a), np.asarray(b)).view(Tensor)


def bitwise_right_shift(a: Any, b: Any) -> Tensor:
    return np.right_shift(np.asarray(a), np.asarray(b)).view(Tensor)


# ---------------------------------------------------------------------------
# torch.load replacement (safetensors-only; .bin not supported)
# ---------------------------------------------------------------------------

def load(path: str, *, map_location: str = "cpu", mmap: builtins_bool = False, weights_only: builtins_bool = True) -> dict[str, Tensor]:
    """Load model weights.  Only safetensors format is supported without PyTorch.
    For .bin files, raises an error with a helpful message."""
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        np_dict = load_file(path)
        return {k: from_numpy(v) for k, v in np_dict.items()}
    else:
        raise RuntimeError(
            f"Cannot load '{path}' without PyTorch.\n"
            f"Only safetensors format is supported in torch-free mode.\n"
            f"Convert your model to safetensors first:\n"
            f"  python -c \"from transformers import AutoModelForCausalLM; "
            f"m = AutoModelForCausalLM.from_pretrained('MODEL_ID'); m.save_pretrained('MODEL_ID', safe_serialization=True)\""
        )


# ---------------------------------------------------------------------------
# Context managers (no-ops without autograd)
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def inference_mode():
    yield


@contextlib.contextmanager
def no_grad():
    yield


# ---------------------------------------------------------------------------
# torch.distributions.normal.Normal replacement
# ---------------------------------------------------------------------------

import builtins as _builtins
builtins_bool = _builtins.bool

class distributions:
    class normal:
        class Normal:
            def __init__(self, loc: float = 0, scale: float = 1):
                self.loc = loc
                self.scale = scale

            def icdf(self, value: Any) -> Tensor:
                """Inverse CDF (quantile / ppf) of the normal distribution.
                Uses rational approximation (Abramowitz & Stegun 26.2.23)."""
                arr = np.asarray(value, dtype=np.float64)
                result = np.vectorize(self._ndtri)(arr) * self.scale + self.loc
                return tensor(result, dtype=float32)

            @staticmethod
            def _ndtri(p: float) -> float:
                """Inverse normal CDF (probit). Accurate to ~1.5e-9."""
                import math as _math
                if p <= 0:
                    return float('-inf')
                if p >= 1:
                    return float('inf')
                if p == 0.5:
                    return 0.0
                # Rational approximation
                if p < 0.5:
                    t = _math.sqrt(-2.0 * _math.log(p))
                    # Numerator coefficients
                    c0, c1, c2 = 2.515517, 0.802853, 0.010328
                    d1, d2, d3 = 1.432788, 0.189269, 0.001308
                    return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))
                else:
                    t = _math.sqrt(-2.0 * _math.log(1 - p))
                    c0, c1, c2 = 2.515517, 0.802853, 0.010328
                    d1, d2, d3 = 1.432788, 0.189269, 0.001308
                    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
