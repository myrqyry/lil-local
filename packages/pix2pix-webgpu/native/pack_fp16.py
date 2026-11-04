#!/usr/bin/env python3
"""Convert the one pinned folded edges2shoes FP32 artifact to packed FP16 conv weights.

Only convolution/deconvolution coefficients change encoding. Bias and folded
BatchNorm affine terms remain FP32. The format is a concatenation of
little-endian FP16 convolution spans and FP32 non-convolution spans in the
existing Rust WeightReader's fixed layer order. Python/NumPy is an OFFLINE
export dependency, not a browser runtime dependency.
"""
import argparse
from array import array
import hashlib
from pathlib import Path
import tempfile

import numpy as np

SOURCE_SHA256 = "eaea6a9c6ec2ed281589c9986ca1b16e755b06d8062673dd0b26d5e4ea235a34"
TOTAL_FLOATS = 54_419_459
PACKED_BYTES = 108_869_644
ENC = [(3,64,False),(64,128,True),(128,256,True),(256,512,True),
       (512,512,True),(512,512,True),(512,512,True),(512,512,False)]
DEC = [(512,512,False,True),(512,512,True,True),(512,512,True,True),
       (512,512,True,True),(512,256,True,True),(256,128,True,True),
       (128,64,True,True),(64,3,True,False)]


def layout():
    """(buffer label, number of scalar values, is convolution kernel)."""
    result = []
    for prefix, specs in (("e", ENC), ("d", DEC)):
        for index, values in enumerate(specs, 1):
            if prefix == "e":
                input_channels, output_channels, norm = values
                skip = False
            else:
                input_channels, output_channels, skip, norm = values
            name = f"{prefix}{index}"
            result.append((f"{name}_weights", output_channels * input_channels *
                           (2 if skip else 1) * 16, True))
            result.append((f"{name}_bias", output_channels, False))
            if norm:
                result.append((f"{name}_norm_gamma", output_channels, False))
                result.append((f"{name}_norm_beta", output_channels, False))
    return result


def pack_values(values, is_conv):
    """Pack a single span, maintaining the fixed endianness."""
    data = np.asarray(values, dtype="<f4")
    if is_conv:
        if not np.isfinite(data).all():
            raise ValueError("Non-finite convolution weights")
        return data.astype("<f2").tobytes()
    return data.tobytes()


def file_sha256(path):
    result = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def convert(source, destination):
    source = Path(source).resolve()
    destination = Path(destination).absolute()
    if source == destination:
        raise ValueError("Source and output must differ")
    if destination.exists():
        raise FileExistsError("Output already exists; refusing to replace it")
    if source.stat().st_size != TOTAL_FLOATS * 4 or file_sha256(source) != SOURCE_SHA256:
        raise ValueError("Input is not the pinned, verified FP32 folded model")

    spans = layout()
    assert sum(count for _, count, _ in spans) == TOTAL_FLOATS
    assert sum(count * (2 if is_conv else 4) for _,count,is_conv in spans) == PACKED_BYTES
    values = np.memmap(source, dtype="<f4", mode="r")
    destination.parent.mkdir(parents=True,exist_ok=True)
    offset = 0
    with tempfile.NamedTemporaryFile(dir=destination.parent,prefix=".pix2pix-fp16-",delete=False) as temp:
        pending = Path(temp.name)
        try:
            for label,count,is_conv in spans:
                temp.write(pack_values(values[offset:offset+count],is_conv))
                offset += count
            temp.flush()
            if temp.tell()!=PACKED_BYTES or offset!=TOTAL_FLOATS:
                raise ValueError("Output size/layout mismatch")
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    if destination.exists():
        pending.unlink()
        raise FileExistsError("Output appeared during conversion")
    pending.rename(destination)
    print(f"{destination}: {destination.stat().st_size} bytes; SHA256 {file_sha256(destination)}")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    convert(args.source,args.output)
