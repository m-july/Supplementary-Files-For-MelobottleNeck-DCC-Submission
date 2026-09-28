# simplemono_preproc/npy_stream.py
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


class NpyAppendWriter:
    """
    Append rows to a .npy file where only the first dimension grows.

    We write a placeholder header first (with big N), stream raw data,
    then rewrite header with actual N on close.

    The produced file is a standard .npy (C-order, non-Fortran).
    """

    MAGIC = b"\x93NUMPY"
    VERSION = (1, 0)

    def __init__(
        self,
        out_path: Path,
        dtype: np.dtype,
        row_shape: Sequence[int],
        max_rows_digits: int = 10,
    ):
        self.out_path = Path(out_path)
        self.dtype = np.dtype(dtype)
        self.row_shape = tuple(int(x) for x in row_shape)
        self.max_rows_digits = int(max_rows_digits)

        self._fp = None
        self._count = 0
        self._header_len = None
        self._header_start = None

        if self.out_path.exists():
            raise FileExistsError(f"File exists: {self.out_path}")
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        self._open_and_write_placeholder_header()

    @property
    def count(self) -> int:
        return self._count

    def _make_header_bytes(self, shape: Tuple[int, ...], fixed_header_len: Optional[int] = None) -> Tuple[bytes, int]:
        descr = self.dtype.str
        header_str = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"

        # header must end with newline, padded with spaces so that
        # (magic+version+hlen+header) is aligned to 16 bytes.
        magic_len = len(self.MAGIC) + 2  # + version bytes
        header_len_field = 2            # v1.0 uses uint16
        base = magic_len + header_len_field

        if fixed_header_len is None:
            # compute padding for alignment
            header_bytes = header_str.encode("latin1")
            pad = (- (base + len(header_bytes) + 1)) % 16
            header = header_bytes + (b" " * pad) + b"\n"
            return header, len(header)
        else:
            # fit into fixed_header_len
            hb = header_str.encode("latin1")
            if len(hb) + 1 > fixed_header_len:
                raise ValueError("New header would be longer than reserved header_len.")
            pad_spaces = fixed_header_len - (len(hb) + 1)
            header = hb + (b" " * pad_spaces) + b"\n"
            return header, fixed_header_len

    def _open_and_write_placeholder_header(self):
        placeholder_rows = int("9" * self.max_rows_digits)
        shape = (placeholder_rows,) + self.row_shape

        header, header_len = self._make_header_bytes(shape, fixed_header_len=None)

        with self.out_path.open("wb") as f:
            # magic + version
            f.write(self.MAGIC)
            f.write(struct.pack("BB", *self.VERSION))
            # header length
            if header_len >= 65536:
                raise ValueError("Header too large for npy v1.0")
            f.write(struct.pack("<H", header_len))
            # header
            header_start = f.tell()
            f.write(header)

        # reopen in append/update mode
        self._fp = self.out_path.open("r+b", buffering=0)
        # data starts after header
        self._header_len = header_len
        # header starts after magic+version+hlen field
        self._header_start = len(self.MAGIC) + 2 + 2  # magic + version(2) + hlen(2)

        # seek to end for appending data
        self._fp.seek(0, os.SEEK_END)

    def append(self, arr: np.ndarray):
        """
        Append one row (row_shape) or a batch (n, *row_shape).
        """
        if self._fp is None:
            raise RuntimeError("Writer is closed.")

        a = np.asarray(arr)
        if a.dtype != self.dtype:
            a = a.astype(self.dtype, copy=False)

        if a.ndim == len(self.row_shape):
            # single row
            if tuple(a.shape) != self.row_shape:
                raise ValueError(f"Row shape mismatch: got {a.shape}, want {self.row_shape}")
            a = a.reshape((1,) + self.row_shape)
        else:
            if a.ndim != len(self.row_shape) + 1:
                raise ValueError(f"Bad ndim: {a.ndim}")
            if tuple(a.shape[1:]) != self.row_shape:
                raise ValueError(f"Batch row shape mismatch: got {a.shape[1:]}, want {self.row_shape}")

        a = np.ascontiguousarray(a)
        self._fp.write(a.data)  # memoryview, no extra copy
        self._count += int(a.shape[0])

    def close(self):
        if self._fp is None:
            return

        # rewrite header with real shape
        shape = (self._count,) + self.row_shape
        header, _ = self._make_header_bytes(shape, fixed_header_len=self._header_len)

        self._fp.seek(self._header_start, os.SEEK_SET)
        self._fp.write(header)

        self._fp.close()
        self._fp = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
