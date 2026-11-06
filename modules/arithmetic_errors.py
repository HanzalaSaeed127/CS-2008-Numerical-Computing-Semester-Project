from __future__ import annotations
import struct
import numpy as np


def error_metrics(exact, approximate):
    absolute = abs(float(exact) - float(approximate))
    relative = absolute / abs(float(exact)) if exact != 0 else None
    return {"absolute": absolute, "relative": relative, "percentage": None if relative is None else relative * 100}


def ieee754(value, precision="double"):
    if precision == "single":
        bits = format(struct.unpack(">I", struct.pack(">f", float(value)))[0], "032b")
        exponent_bits, bias = 8, 127
    else:
        bits = format(struct.unpack(">Q", struct.pack(">d", float(value)))[0], "064b")
        exponent_bits, bias = 11, 1023
    return {"sign": bits[0], "exponent": bits[1:1 + exponent_bits], "fraction": bits[1 + exponent_bits:], "bits": bits, "bias": bias}


def machine_info():
    return {"epsilon": np.finfo(float).eps, "max": np.finfo(float).max, "tiny": np.finfo(float).tiny, "digits": np.finfo(float).nmant}


def rounding_demo(value, digits):
    rounded = round(float(value), int(digits))
    truncated = np.trunc(float(value) * 10**int(digits)) / 10**int(digits)
    return {"original": float(value), "rounded": rounded, "truncated": truncated}
