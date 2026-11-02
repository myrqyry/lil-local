#!/usr/bin/env python3
"""
Production Weight Exporter & Verifier for Pix2Pix U-Net 256 (`export_weights.py`)
=================================================================================
Zero-dependency compatible (works with standard library `array`/`struct`/`zipfile`/`pickle`
OR accelerated `numpy`/`torch` when installed).

Fixes all 5 critical weight-export bugs in the original plan:
1. Correct PyTorch `UnetSkipConnectionBlock` recursive prefix indexing:
   - Outermost block (`e1` / `d8`) wraps `.model.1` (because `e1` has no ReLU/Norm).
   - Intermediate blocks (`e2..e7` / `d7..d2`) wrap `.model.3` (after ReLU[0], Conv[1], Norm[2]).
2. Standardizes missing conv biases (when trained with `--norm batch` where `use_bias=False`)
   and missing norm affine parameters (when trained with `--norm instance` where `affine=False`)
   by synthesizing explicit zero biases and identity `(gamma=1, beta=0)` vectors so the binary
   stream strictly matches the 54,419,459-float (217,677,836-byte) Rust `WeightReader` contract.
3. Explicitly extracts only `.weight` and `.bias` from normalization modules, never exporting
   `BatchNorm2d.running_mean`, `running_var`, or the int64 `num_batches_tracked` tensor.
4. Explicitly transposes all 8 decoder `ConvTranspose2d` weights (`d1..d8`) from PyTorch's
   `[C_in, C_out, 4, 4]` layout to `[C_out, C_in, 4, 4]`, fixing the bug where `d1`
   (`...model.3.weight`) could be missed by fragile substring checks.
5. Swaps the skip-connection channel halves along `C_in` for `d2..d8` when exporting from
   `junyanz/pytorch-CycleGAN-and-pix2pix` (`torch.cat([x, self.model(x)], 1)` = `[skip, dec]`)
   so it aligns with the WGSL shader's `[deconv_in_a (dec), deconv_in_b (skip)]` convention.
"""

import argparse
from array import array
import collections
import io
import math
import os
import pickle
import struct
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zipfile

try:
    import numpy as _np  # type: ignore
except ImportError:
    _np = None

EXPECTED_TOTAL_FLOATS = 54_419_459
EXPECTED_TOTAL_BYTES = EXPECTED_TOTAL_FLOATS * 4  # 217,677,836 bytes

# Canonical layer specification for U-Net 256:
# (name, c_in_single, c_out, has_skip, has_norm)
ENCODER_SPECS: List[Tuple[str, int, int, bool, bool]] = [
    ("e1", 3,   64,  False, False),
    ("e2", 64,  128, False, True),
    ("e3", 128, 256, False, True),
    ("e4", 256, 512, False, True),
    ("e5", 512, 512, False, True),
    ("e6", 512, 512, False, True),
    ("e7", 512, 512, False, True),
    ("e8", 512, 512, False, False),
]

DECODER_SPECS: List[Tuple[str, int, int, bool, bool]] = [
    ("d1", 512, 512, False, True),
    ("d2", 512, 512, True,  True),
    ("d3", 512, 512, True,  True),
    ("d4", 512, 512, True,  True),
    ("d5", 512, 256, True,  True),
    ("d6", 256, 128, True,  True),
    ("d7", 128, 64,  True,  True),
    ("d8", 64,  3,   True,  False),
]


class Float32Tensor:
    """
    Portable, zero-dependency N-D float32 tensor backed by Python's `array('f')`
    (little-endian IEEE-754 binary32) with fast C-buffer export, axis-0 half-swapping,
    and 4D `(C_in, C_out, KH, KW) -> (C_out, C_in, KH, KW)` transposition.
    """

    __slots__ = ("shape", "buffer")

    def __init__(self, shape: Tuple[int, ...], buffer: Optional[array] = None) -> None:
        expected_size = math.prod(shape)
        if buffer is None:
            buf = array("f", [0.0]) * expected_size
        else:
            if len(buffer) != expected_size:
                raise ValueError(
                    f"Buffer length {len(buffer)} does not match shape {shape} ({expected_size})"
                )
            buf = buffer
        self.shape = tuple(shape)
        self.buffer = buf

    @property
    def size(self) -> int:
        return len(self.buffer)

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> "Float32Tensor":
        return cls(shape, array("f", [0.0]) * math.prod(shape))

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> "Float32Tensor":
        return cls(shape, array("f", [1.0]) * math.prod(shape))

    @classmethod
    def from_any(cls, obj: Any) -> "Float32Tensor":
        if isinstance(obj, cls):
            return cls(obj.shape, array("f", obj.buffer))
        if hasattr(obj, "detach"):
            obj = obj.detach().cpu()
        if _np is not None and isinstance(obj, _np.ndarray):
            arr32 = _np.ascontiguousarray(obj, dtype="<f4")
            buf = array("f")
            buf.frombytes(arr32.tobytes())
            return cls(tuple(int(d) for d in arr32.shape), buf)
        if hasattr(obj, "numpy"):
            arr32 = _np.ascontiguousarray(obj.numpy(), dtype="<f4")
            buf = array("f")
            buf.frombytes(arr32.tobytes())
            return cls(tuple(int(d) for d in arr32.shape), buf)
        if isinstance(obj, (list, tuple)):
            shape: List[int] = []
            curr = obj
            while isinstance(curr, (list, tuple)):
                shape.append(len(curr))
                curr = curr[0] if len(curr) > 0 else None
            flat: List[float] = []

            def _flatten(item: Any) -> None:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        _flatten(sub)
                else:
                    flat.append(float(item))

            _flatten(obj)
            return cls(tuple(shape), array("f", flat))
        raise TypeError(f"Unsupported tensor object type: {type(obj)!r}")

    def swap_axis0_halves(self) -> "Float32Tensor":
        """Swaps `[0..half, ...]` and `[half..2*half, ...]` along axis 0."""
        if not self.shape or self.shape[0] % 2 != 0:
            raise ValueError(f"Axis 0 must be even to swap halves, got shape {self.shape}")
        half_elems = self.size // 2
        swapped = array("f")
        swapped.extend(self.buffer[half_elems:])
        swapped.extend(self.buffer[:half_elems])
        return Float32Tensor(self.shape, swapped)

    def transpose_01_4d(self) -> "Float32Tensor":
        """Transposes a 4D tensor from `(C_in, C_out, KH, KW)` to `(C_out, C_in, KH, KW)`."""
        if len(self.shape) != 4:
            raise ValueError(f"Expected 4D tensor for transpose_01_4d, got {self.shape}")
        d0, d1, d2, d3 = self.shape
        kernel_elems = d2 * d3
        src = self.buffer
        dst = array("f", [0.0]) * self.size

        for c_in in range(d0):
            for c_out in range(d1):
                src_off = (c_in * d1 + c_out) * kernel_elems
                dst_off = (c_out * d0 + c_in) * kernel_elems
                dst[dst_off : dst_off + kernel_elems] = src[src_off : src_off + kernel_elems]

        return Float32Tensor((d1, d0, d2, d3), dst)

    def permute_rgb_bgr_axis0(self) -> "Float32Tensor":
        """
        Folds Torch7 `util.deprocess` BGR -> RGB channel permutation (`perm = {3, 2, 1}`)
        into axis 0 of a 3-channel output weight `[3, C_in, 4, 4]` or bias `[3]` tensor
        at export time with zero runtime GPU cost.
        """
        if not self.shape or self.shape[0] != 3:
            raise ValueError(f"Axis 0 must have length 3 for BGR<->RGB fold, got {self.shape}")
        plane = self.size // 3
        src = self.buffer
        dst = array("f")
        dst.extend(src[2 * plane : 3 * plane])
        dst.extend(src[1 * plane : 2 * plane])
        dst.extend(src[0 * plane : 1 * plane])
        return Float32Tensor(self.shape, dst)

    def permute_rgb_bgr_axis1(self) -> "Float32Tensor":
        """
        Folds Torch7 `util.preprocess` RGB -> BGR channel permutation (`perm = {3, 2, 1}`)
        into axis 1 of `e1` weights `[C_out, 3, 4, 4]` at export time with zero runtime GPU cost.
        """
        if len(self.shape) != 4 or self.shape[1] != 3:
            raise ValueError(f"Expected (C_out, 3, KH, KW) for axis-1 BGR<->RGB fold, got {self.shape}")
        c_out, _, kh, kw = self.shape
        k_elems = kh * kw
        src = self.buffer
        dst = array("f", [0.0]) * self.size
        for oc in range(c_out):
            base = oc * 3 * k_elems
            dst[base + 0 * k_elems : base + 1 * k_elems] = src[base + 2 * k_elems : base + 3 * k_elems]
            dst[base + 1 * k_elems : base + 2 * k_elems] = src[base + 1 * k_elems : base + 2 * k_elems]
            dst[base + 2 * k_elems : base + 3 * k_elems] = src[base + 0 * k_elems : base + 1 * k_elems]
        return Float32Tensor(self.shape, dst)

    def to_le_bytes(self) -> bytes:
        buf = array("f", self.buffer)
        if sys.byteorder != "little":
            buf.byteswap()
        return buf.tobytes()


# ==============================================================================
# Zero-Dependency PyTorch `.pth` ZIP Archive & Legacy Pickle Checkpoint Reader
# ==============================================================================
class _RawStorageRef:
    """Represents a typed PyTorch storage blob inside a `.pth` ZIP archive."""

    __slots__ = ("dtype_tag", "key", "num_elements", "float_buffer")

    def __init__(self, dtype_tag: str, key: str, num_elements: int, float_buffer: array) -> None:
        self.dtype_tag = dtype_tag
        self.key = key
        self.num_elements = num_elements
        self.float_buffer = float_buffer


def _contiguous_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    running = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = running
        running *= shape[i]
    return tuple(strides)


def _extract_strided_to_contiguous(
    storage_buf: array,
    storage_offset: int,
    shape: Tuple[int, ...],
    stride: Tuple[int, ...],
) -> array:
    total = math.prod(shape) if shape else 1
    if not shape:
        return array("f", [storage_buf[storage_offset]])
    if stride == _contiguous_strides(shape):
        return array("f", storage_buf[storage_offset : storage_offset + total])

    out = array("f", [0.0]) * total
    ndim = len(shape)
    coords = [0] * ndim
    for flat_idx in range(total):
        src_idx = storage_offset
        for d in range(ndim):
            src_idx += coords[d] * stride[d]
        out[flat_idx] = storage_buf[src_idx]
        for d in range(ndim - 1, -1, -1):
            coords[d] += 1
            if coords[d] < shape[d]:
                break
            coords[d] = 0
    return out


def _rebuild_tensor_v2(
    storage: Any,
    storage_offset: int,
    size: Sequence[int],
    stride: Sequence[int],
    requires_grad: bool = False,
    backward_hooks: Any = None,
    metadata: Any = None,
) -> Union[Float32Tensor, int, float]:
    shape = tuple(int(s) for s in size)
    stride_tuple = tuple(int(s) for s in stride)
    if isinstance(storage, _RawStorageRef):
        if len(shape) == 0:
            val = storage.float_buffer[storage_offset] if len(storage.float_buffer) > storage_offset else 0.0
            return int(val) if storage.dtype_tag in ("LongStorage", "IntStorage", "ShortStorage", "ByteStorage") else float(val)
        buf = _extract_strided_to_contiguous(storage.float_buffer, int(storage_offset), shape, stride_tuple)
        return Float32Tensor(shape, buf)
    return Float32Tensor.from_any(storage)


def _rebuild_tensor(
    storage: Any,
    storage_offset: int,
    size: Sequence[int],
    stride: Sequence[int],
) -> Union[Float32Tensor, int, float]:
    return _rebuild_tensor_v2(storage, storage_offset, size, stride)


def _rebuild_parameter(data: Any, requires_grad: bool = False, backward_hooks: Any = None) -> Any:
    return data


class _PyTorchZipUnpickler(pickle.Unpickler):
    """
    Pure-Python `pickle.Unpickler` that resolves PyTorch `.pth` ZIP archive
    `persistent_id` storage references (`data/0`, `data/1`, ...) directly into
    `Float32Tensor` objects without requiring `torch` or `numpy`.
    """

    _STORAGE_DTYPE_SPECS: Dict[str, Tuple[str, int]] = {
        "FloatStorage": ("f", 4),
        "DoubleStorage": ("d", 8),
        "HalfStorage": ("e", 2),
        "LongStorage": ("q", 8),
        "IntStorage": ("i", 4),
        "ShortStorage": ("h", 2),
        "CharStorage": ("b", 1),
        "ByteStorage": ("B", 1),
        "BoolStorage": ("?", 1),
    }

    def __init__(self, file_obj: io.BufferedReader, zf: zipfile.ZipFile, data_dir_prefix: str) -> None:
        super().__init__(file_obj)
        self._zf = zf
        self._prefix = data_dir_prefix
        self._storage_cache: Dict[str, _RawStorageRef] = {}

    def find_class(self, module: str, name: str) -> Any:
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor_v3"):
            return _rebuild_tensor_v2
        if name == "_rebuild_tensor":
            return _rebuild_tensor
        if name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module.startswith("torch") and name.endswith("Storage"):
            return name
        return super().find_class(module, name)

    def _read_storage_blob(self, dtype_tag: str, key: str, num_elements: int) -> _RawStorageRef:
        cache_key = f"{dtype_tag}:{key}"
        if cache_key in self._storage_cache:
            return self._storage_cache[cache_key]

        entry_name = f"{self._prefix}data/{key}"
        raw_bytes = self._zf.read(entry_name)

        if dtype_tag == "FloatStorage":
            buf = array("f")
            buf.frombytes(raw_bytes[: num_elements * 4])
            if sys.byteorder != "little":
                buf.byteswap()
        elif dtype_tag == "DoubleStorage":
            arr64 = array("d")
            arr64.frombytes(raw_bytes[: num_elements * 8])
            if sys.byteorder != "little":
                arr64.byteswap()
            buf = array("f", (float(x) for x in arr64))
        elif dtype_tag == "HalfStorage":
            vals = struct.unpack(f"<{num_elements}e", raw_bytes[: num_elements * 2])
            buf = array("f", vals)
        elif dtype_tag in self._STORAGE_DTYPE_SPECS:
            fmt_char, elem_bytes = self._STORAGE_DTYPE_SPECS[dtype_tag]
            vals = struct.unpack(f"<{num_elements}{fmt_char}", raw_bytes[: num_elements * elem_bytes])
            buf = array("f", (float(v) for v in vals))
        else:
            raise ValueError(f"Unsupported PyTorch storage type in .pth archive: {dtype_tag}")

        ref = _RawStorageRef(dtype_tag, key, num_elements, buf)
        self._storage_cache[cache_key] = ref
        return ref

    def persistent_load(self, pid: Any) -> Any:
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            storage_type, key, _location, num_elements = pid[1], str(pid[2]), pid[3], int(pid[4])
            dtype_tag = storage_type if isinstance(storage_type, str) else getattr(storage_type, "__name__", "FloatStorage")
            return self._read_storage_blob(dtype_tag, key, num_elements)
        raise pickle.UnpicklingError(f"Unsupported persistent_load ID in PyTorch archive: {pid!r}")


class _PyTorchZipPickler(pickle.Pickler):
    """
    Pure-Python `pickle.Pickler` that serializes `Float32Tensor` and int64 scalar entries
    into standard PyTorch `.pth` ZIP archive format (`data.pkl` + `data/<idx>` blobs)
    using `_rebuild_tensor_v2` and `FloatStorage` / `LongStorage` persistent storage IDs.
    """

    def __init__(self, file_obj: io.BytesIO, storage_blobs: Dict[str, bytes]) -> None:
        super().__init__(file_obj, protocol=2)
        self._storage_blobs = storage_blobs

    def persistent_id(self, obj: Any) -> Any:
        if isinstance(obj, _RawStorageRef):
            return ("storage", obj.dtype_tag, obj.key, "cpu", obj.num_elements)
        return None


class _TorchTensorPickleProxy:
    """Pickles a `Float32Tensor` as `_rebuild_tensor_v2(storage, 0, shape, stride, False, OrderedDict())`."""

    def __init__(self, storage_ref: _RawStorageRef, shape: Tuple[int, ...]) -> None:
        self.storage_ref = storage_ref
        self.shape = shape

    def __reduce__(self) -> Tuple[Any, Tuple[Any, ...]]:
        return (
            _rebuild_tensor_v2,
            (
                self.storage_ref,
                0,
                self.shape,
                _contiguous_strides(self.shape),
                False,
                collections.OrderedDict(),
            ),
        )


def build_pytorch_zip_checkpoint_bytes(
    tensors: Dict[str, Float32Tensor],
    extra_int64_scalars: Optional[Dict[str, int]] = None,
    archive_prefix: str = "archive",
) -> bytes:
    """
    Constructs a genuine PyTorch `.pth` ZIP archive in memory (`{archive_prefix}/data.pkl` +
    `{archive_prefix}/data/0`, `{archive_prefix}/data/1`, ...) without requiring `torch`.
    """
    storage_blobs: Dict[str, bytes] = {}
    state_dict_proxies: Dict[str, Any] = collections.OrderedDict()
    idx = 0

    for k, tensor in tensors.items():
        key = str(idx)
        idx += 1
        raw_bytes = tensor.to_le_bytes()
        storage_blobs[key] = raw_bytes
        ref = _RawStorageRef("FloatStorage", key, tensor.size, tensor.buffer)
        state_dict_proxies[k] = _TorchTensorPickleProxy(ref, tensor.shape)

    if extra_int64_scalars:
        for k, val in extra_int64_scalars.items():
            key = str(idx)
            idx += 1
            storage_blobs[key] = struct.pack("<q", int(val))
            ref = _RawStorageRef("LongStorage", key, 1, array("f", [float(val)]))
            state_dict_proxies[k] = _TorchTensorPickleProxy(ref, ())

    pkl_io = io.BytesIO()
    pickler = _PyTorchZipPickler(pkl_io, storage_blobs)
    pickler.dump(state_dict_proxies)

    zip_io = io.BytesIO()
    with zipfile.ZipFile(zip_io, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(f"{archive_prefix}/version", b"3\n")
        zf.writestr(f"{archive_prefix}/byteorder", b"little")
        zf.writestr(f"{archive_prefix}/data.pkl", pkl_io.getvalue())
        for key, raw_bytes in storage_blobs.items():
            zf.writestr(f"{archive_prefix}/data/{key}", raw_bytes)

    return zip_io.getvalue()


def _load_pytorch_zip_stream(stream: Union[str, io.BytesIO]) -> Dict[str, Float32Tensor]:
    with zipfile.ZipFile(stream, "r") as zf:
        names = zf.namelist()
        data_pkl_name = next((n for n in names if n.endswith("data.pkl")), None)
        if data_pkl_name is None:
            raise ValueError("ZIP checkpoint does not contain a PyTorch 'data.pkl' entry.")
        prefix = data_pkl_name[: -len("data.pkl")]
        with zf.open(data_pkl_name, "r") as pkl_file:
            unpickler = _PyTorchZipUnpickler(io.BufferedReader(pkl_file), zf, prefix)
            ckpt = unpickler.load()

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict state_dict in PyTorch ZIP archive, got {type(ckpt)!r}")

    out: Dict[str, Float32Tensor] = {}
    for k, v in ckpt.items():
        if isinstance(v, Float32Tensor):
            out[k] = v
        elif isinstance(v, (int, float)):
            # Skip scalar tracking buffers such as BatchNorm2d.num_batches_tracked
            continue
        else:
            try:
                out[k] = Float32Tensor.from_any(v)
            except TypeError:
                continue
    return out


def get_junyanz_block_prefixes(root_prefix: str = "model") -> List[str]:
    """
    Returns the exact module prefixes for levels 0..7 in junyanz/pytorch-CycleGAN-and-pix2pix:
      Level 0 (outermost, e1 + d8): `{root}`
      Level 1 (e2 + d7):            `{root}.model.1`
      Level 2 (e3 + d6):            `{root}.model.1.model.3`
      ...
      Level 7 (innermost, e8 + d1): `{root}.model.1` + `.model.3` * 6
    """
    prefixes = [root_prefix]
    curr = f"{root_prefix}.model.1" if root_prefix else "model.1"
    prefixes.append(curr)
    for _ in range(2, 8):
        curr = f"{curr}.model.3"
        prefixes.append(curr)
    return prefixes


def normalize_flat_checkpoint(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map explicit down1..8/up1..7/final flat U-Net weights into recursive keys.

    Reject partial layouts rather than silently manufacturing convolution weights.
    Normalization affine parameters and convolution biases remain optional in
    export_junyanz_state_dict and are synthesized by that established path.
    """
    prefixes = get_junyanz_block_prefixes()
    mapping: Dict[str, str] = {}
    for i in range(8):
        mapping[f"down{i + 1}.model.0.weight"] = (
            f"{prefixes[i]}.model.{0 if i == 0 else 1}.weight"
        )
    for i in range(7):
        level = 7 - i
        mapping[f"up{i + 1}.model.0.weight"] = (
            f"{prefixes[level]}.model.{3 if level == 7 else 5}.weight"
        )
    mapping["final.2.weight"] = "model.model.3.weight"
    mapping["final.2.bias"] = "model.model.3.bias"
    missing = sorted(set(mapping) - set(state_dict))
    if missing:
        raise ValueError(f"Flat Pix2Pix U-Net checkpoint missing required keys: {missing}")
    unknown = sorted(set(state_dict) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized flat Pix2Pix U-Net keys: {unknown}")
    return {dest: state_dict[src] for src, dest in mapping.items()}


def detect_root_prefix(state_dict: Dict[str, Any]) -> str:
    """Detect the prefix before the outer UnetSkipConnectionBlock model container.

    Official junyanz/pytorch-CycleGAN-and-pix2pix checkpoints use
    model.model.0.weight (or module.model.model.0.weight under DataParallel).
    Older flattened probes remain supported as compatibility fallbacks.
    """
    for candidate in ("model", "module.model", "netG.model", ""):
        canonical = f"{candidate}.model.0.weight" if candidate else "model.0.weight"
        if canonical in state_dict:
            return candidate
        legacy = f"{candidate}.0.weight" if candidate else "0.weight"
        if legacy in state_dict:
            return candidate
    raise KeyError(
        "Could not locate outermost encoder weight "
        "('model.model.0.weight', 'model.0.weight', or compatible prefixed form)."
    )


def _join(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix


def detect_normalization_mode(state_dict: Dict[str, Any]) -> str:
    """Return the runtime normalization semantics required by this checkpoint."""
    has_batch_stats = any(
        key.endswith(".running_mean") or key.endswith(".running_var")
        for key in state_dict
    )
    return "batch_eval_folded" if has_batch_stats else "instance"


def fold_batch_norm_eval(
    gamma: Float32Tensor,
    beta: Float32Tensor,
    running_mean: Float32Tensor,
    running_var: Float32Tensor,
    *,
    eps: float = 1e-5,
) -> Tuple[Float32Tensor, Float32Tensor]:
    """Fold BatchNorm eval into per-channel affine scale/shift.

    y = ((x - mean) / sqrt(var + eps)) * gamma + beta
      = x * scale + shift
    """
    if not (gamma.shape == beta.shape == running_mean.shape == running_var.shape):
        raise ValueError(
            "BatchNorm tensors must share one channel shape: "
            f"gamma={gamma.shape}, beta={beta.shape}, mean={running_mean.shape}, var={running_var.shape}"
        )
    scale = array("f")
    shift = array("f")
    for g, b, mean, variance in zip(
        gamma.buffer, beta.buffer, running_mean.buffer, running_var.buffer
    ):
        if variance < 0.0:
            raise ValueError(f"BatchNorm running variance must be non-negative, got {variance}")
        s = float(g) / math.sqrt(float(variance) + eps)
        scale.append(s)
        shift.append(float(b) - float(mean) * s)
    return Float32Tensor(gamma.shape, scale), Float32Tensor(beta.shape, shift)


def validate_runtime_normalization(state_dict: Dict[str, Any]) -> str:
    """Validate global normalization metadata and return the required runtime mode."""
    mode = detect_normalization_mode(state_dict)
    if mode == "batch_eval_folded":
        means = {k[:-13] for k in state_dict if k.endswith(".running_mean")}
        variances = {k[:-12] for k in state_dict if k.endswith(".running_var")}
        if means != variances:
            raise ValueError("BatchNorm checkpoint has incomplete running_mean/running_var pairs")
    return mode


def _export_norm_affine(
    state_dict: Dict[str, Any],
    gamma_key: str,
    beta_key: str,
    c_out: int,
    *,
    mode: str,
    eps: float = 1e-5,
) -> Tuple[Float32Tensor, Float32Tensor]:
    if gamma_key in state_dict and beta_key in state_dict:
        gamma = Float32Tensor.from_any(state_dict[gamma_key])
        beta = Float32Tensor.from_any(state_dict[beta_key])
    else:
        gamma = Float32Tensor.ones((c_out,))
        beta = Float32Tensor.zeros((c_out,))
    if gamma.shape != (c_out,) or beta.shape != (c_out,):
        raise ValueError(
            f"Normalization affine shape mismatch for {gamma_key}: "
            f"expected {(c_out,)}, got gamma={gamma.shape}, beta={beta.shape}"
        )
    if mode != "batch_eval_folded":
        return gamma, beta

    prefix = gamma_key.rsplit(".", 1)[0]
    mean_key = prefix + ".running_mean"
    var_key = prefix + ".running_var"
    if mean_key not in state_dict or var_key not in state_dict:
        raise ValueError(f"BatchNorm layer {prefix!r} is missing running statistics")
    running_mean = Float32Tensor.from_any(state_dict[mean_key])
    running_var = Float32Tensor.from_any(state_dict[var_key])
    return fold_batch_norm_eval(gamma, beta, running_mean, running_var, eps=eps)


def export_junyanz_state_dict(
    state_dict: Dict[str, Any],
    output_path: str,
    swap_skip_channels: bool = True,
    bgr_to_rgb: bool = False,
) -> int:
    """
    Exports a junyanz/pytorch-CycleGAN-and-pix2pix or Torch7 UnetGenerator state_dict
    into the exact 217,677,836-byte binary layout expected by `src/lib.rs`.
    When `bgr_to_rgb=True`, folds Torch7's `util.preprocess`/`util.deprocess`
    `perm = {3, 2, 1}` BGR<->RGB swizzle into `e1` and `d8` weights at zero runtime cost.
    Returns total bytes written.
    """
    normalization_mode = validate_runtime_normalization(state_dict)
    root = detect_root_prefix(state_dict)
    prefixes = get_junyanz_block_prefixes(root)
    floats_written = 0

    with open(output_path, "wb") as f:
        # 1. Export Encoder Layers (e1 .. e8)
        for i, (name, c_in, c_out, _, has_norm) in enumerate(ENCODER_SPECS):
            p = prefixes[i]
            conv_idx = "0" if i == 0 else "1"
            w_key = _join(p, f"model.{conv_idx}.weight")
            b_key = _join(p, f"model.{conv_idx}.bias")

            if w_key not in state_dict:
                raise KeyError(f"Missing encoder weight key '{w_key}' for {name}")

            w = Float32Tensor.from_any(state_dict[w_key])
            expected_w_shape = (c_out, c_in, 4, 4)
            if w.shape != expected_w_shape:
                raise ValueError(
                    f"{name} weight shape mismatch: expected {expected_w_shape}, got {w.shape}"
                )

            if name == "e1" and bgr_to_rgb:
                w = w.permute_rgb_bgr_axis1()

            if b_key in state_dict:
                b = Float32Tensor.from_any(state_dict[b_key])
                if b.shape != (c_out,):
                    raise ValueError(f"{name} bias shape mismatch: expected {(c_out,)}, got {b.shape}")
            else:
                b = Float32Tensor.zeros((c_out,))

            f.write(w.to_le_bytes())
            f.write(b.to_le_bytes())
            floats_written += w.size + b.size

            if has_norm:
                g_key = _join(p, "model.2.weight")
                beta_key = _join(p, "model.2.bias")
                gamma, beta = _export_norm_affine(
                    state_dict, g_key, beta_key, c_out, mode=normalization_mode
                )

                f.write(gamma.to_le_bytes())
                f.write(beta.to_le_bytes())
                floats_written += gamma.size + beta.size

        # 2. Export Decoder Layers (d1 .. d8)
        for i, (name, c_in_single, c_out, has_skip, has_norm) in enumerate(DECODER_SPECS):
            block_level = 7 - i
            p = prefixes[block_level]

            if block_level == 7:
                upconv_idx, upnorm_idx = "3", "4"
            elif block_level == 0:
                upconv_idx, upnorm_idx = "3", None
            else:
                upconv_idx, upnorm_idx = "5", "6"

            w_key = _join(p, f"model.{upconv_idx}.weight")
            b_key = _join(p, f"model.{upconv_idx}.bias")

            if w_key not in state_dict:
                raise KeyError(f"Missing decoder weight key '{w_key}' for {name}")

            w_pt = Float32Tensor.from_any(state_dict[w_key])
            total_c_in = c_in_single * 2 if has_skip else c_in_single
            expected_pt_shape = (total_c_in, c_out, 4, 4)
            if w_pt.shape != expected_pt_shape:
                raise ValueError(
                    f"{name} PyTorch weight shape mismatch: expected {expected_pt_shape}, got {w_pt.shape}"
                )

            if has_skip and swap_skip_channels:
                w_pt = w_pt.swap_axis0_halves()

            w_wgsl = w_pt.transpose_01_4d()

            if b_key in state_dict:
                b = Float32Tensor.from_any(state_dict[b_key])
                if b.shape != (c_out,):
                    raise ValueError(f"{name} bias shape mismatch: expected {(c_out,)}, got {b.shape}")
            else:
                b = Float32Tensor.zeros((c_out,))

            if name == "d8" and bgr_to_rgb:
                w_wgsl = w_wgsl.permute_rgb_bgr_axis0()
                b = b.permute_rgb_bgr_axis0()

            f.write(w_wgsl.to_le_bytes())
            f.write(b.to_le_bytes())
            floats_written += w_wgsl.size + b.size

            if has_norm and upnorm_idx is not None:
                g_key = _join(p, f"model.{upnorm_idx}.weight")
                beta_key = _join(p, f"model.{upnorm_idx}.bias")
                gamma, beta = _export_norm_affine(
                    state_dict, g_key, beta_key, c_out, mode=normalization_mode
                )

                f.write(gamma.to_le_bytes())
                f.write(beta.to_le_bytes())
                floats_written += gamma.size + beta.size

    bytes_written = floats_written * 4
    if bytes_written != EXPECTED_TOTAL_BYTES:
        raise RuntimeError(
            f"Export size mismatch: wrote {bytes_written} bytes, expected {EXPECTED_TOTAL_BYTES} bytes"
        )
    return bytes_written


def verify_binary_file(bin_path: str) -> Dict[str, Any]:
    """Verifies that a `pix2pix_unet256.bin` file has the exact byte length and layer offsets."""
    actual_size = os.path.getsize(bin_path)
    if actual_size != EXPECTED_TOTAL_BYTES:
        raise ValueError(
            f"Invalid binary size: expected {EXPECTED_TOTAL_BYTES} bytes, got {actual_size} bytes"
        )

    offset = 0
    layer_offsets: Dict[str, Tuple[int, int]] = {}
    for name, c_in, c_out, _, has_norm in ENCODER_SPECS:
        floats = c_out * c_in * 16 + c_out + (2 * c_out if has_norm else 0)
        byte_len = floats * 4
        layer_offsets[name] = (offset, offset + byte_len)
        offset += byte_len

    for name, c_in_single, c_out, has_skip, has_norm in DECODER_SPECS:
        tot_c_in = c_in_single * 2 if has_skip else c_in_single
        floats = c_out * tot_c_in * 16 + c_out + (2 * c_out if has_norm else 0)
        byte_len = floats * 4
        layer_offsets[name] = (offset, offset + byte_len)
        offset += byte_len

    assert offset == EXPECTED_TOTAL_BYTES
    return {
        "valid": True,
        "total_bytes": actual_size,
        "total_floats": actual_size // 4,
        "d1_offset": layer_offsets["d1"],
        "d8_offset": layer_offsets["d8"],
    }


def load_pytorch_checkpoint_any(checkpoint_source: Union[str, bytes, io.BytesIO]) -> Dict[str, Float32Tensor]:
    """
    Loads a PyTorch `.pth` checkpoint using `torch` if installed, or via the
    zero-dependency pure-Python ZIP archive unpickler (`_PyTorchZipUnpickler`) /
    legacy pickle loader. Accepts a file path, raw bytes, or `io.BytesIO`.
    """
    if isinstance(checkpoint_source, bytes):
        stream: Union[str, io.BytesIO] = io.BytesIO(checkpoint_source)
    else:
        stream = checkpoint_source

    if isinstance(stream, str):
        try:
            import torch  # type: ignore

            ckpt = torch.load(stream, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            return {
                k: Float32Tensor.from_any(v)
                for k, v in ckpt.items()
                if hasattr(v, "detach") and getattr(v, "ndim", 0) > 0
            }
        except ImportError:
            pass

    if zipfile.is_zipfile(stream):
        if isinstance(stream, io.BytesIO):
            stream.seek(0)
        return _load_pytorch_zip_stream(stream)

    if isinstance(stream, io.BytesIO):
        stream.seek(0)
        data = pickle.load(stream)
    else:
        with open(stream, "rb") as f:
            data = pickle.load(f)

    if isinstance(data, dict) and "state_dict" in data:
        data = data["state_dict"]
    return {
        k: Float32Tensor.from_any(v)
        for k, v in data.items()
        if not isinstance(v, (int, float))
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export and verify Pix2Pix U-Net 256 checkpoints for Rust + WebGPU."
    )
    parser.add_argument("checkpoint", nargs="?", help="Path to PyTorch .pth checkpoint")
    parser.add_argument("-o", "--output", default="pix2pix_unet256.bin", help="Output .bin path")
    parser.add_argument(
        "--no-swap-skip",
        action="store_true",
        help="Disable swapping [skip, decoder] -> [decoder, skip] (use only if checkpoint already uses Torch7 order)",
    )
    parser.add_argument(
        "--bgr-to-rgb",
        action="store_true",
        help="Fold phillipi/pix2pix Torch7 util.preprocess/deprocess BGR<->RGB permutation ({3,2,1}) into e1 and d8 weights",
    )
    parser.add_argument(
        "--verify",
        metavar="BIN_PATH",
        help="Verify an existing pix2pix_unet256.bin file's byte layout and exit",
    )
    args = parser.parse_args()

    if args.verify:
        report = verify_binary_file(args.verify)
        print(f"Verified {args.verify}: {report}")
        return

    if not args.checkpoint:
        parser.error("Provide a checkpoint path or use --verify BIN_PATH.")

    state_dict = load_pytorch_checkpoint_any(args.checkpoint)
    if any(key.startswith('down1.') for key in state_dict):
        state_dict = normalize_flat_checkpoint(state_dict)
    total_bytes = export_junyanz_state_dict(
        state_dict,
        args.output,
        swap_skip_channels=not args.no_swap_skip,
        bgr_to_rgb=args.bgr_to_rgb,
    )
    print(f"Exported {args.output}: {total_bytes:,} bytes ({total_bytes / (1024 * 1024):.2f} MiB)")


if __name__ == "__main__":
    main()
