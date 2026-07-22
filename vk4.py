"""
Reader for Keyence VK4 laser-confocal-microscope files (VK-X series).

Only the parts needed for pin-fin profilometry are decoded: the height map,
the (optional) laser-intensity image, and the X/Y/Z calibration.

Byte layout (verified empirically against the 071526 test-wafer dataset;
all 42 files are identical 2048x1536, 32-bit height, 0.6934 um/px, 0.1 nm/digit):

  0x00  magic            b"VK4_"
  0x04  dll version      uint32
  0x08  file type        uint32
  0x0C  offset table     uint32[...]   byte offsets of each data section:
            +12 setting (measurement conditions)
            +16 color_peak      +20 color_light
            +24 light (laser intensity)   ... +36 height
  setting -> measurement-conditions block, uint32 fields (index from block start):
            0 size   1 year 2 month 3 day 4 hour 5 min 6 sec 7 utc_offset_min
            ...
            42 x_length_per_pixel (pm)
            43 y_length_per_pixel (pm)
            44 z_length_per_digit (pm)
  each data block:
            uint32 width, height, bit_depth, compression, byte_size,
                   palette_range_min, palette_range_max      (28 bytes)
            768-byte RGB palette (256 entries)
            raw pixel data (width*height*bit_depth/8 bytes, row-major, LE)

Height pixels are unsigned integers ("digits"); physical height in micrometres
is  raw_digit * z_length_per_digit_pm * 1e-6.

Reference for the format: the community-reverse-engineered VK4 spec used by the
`vk4extract` project; the field offsets above were re-confirmed byte-by-byte on
this dataset.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MAGIC = b"VK4_"

# --- offset-table byte positions (little-endian uint32) ---
_OFF_SETTING = 12
_OFF_COLOR_PEAK = 16
_OFF_COLOR_LIGHT = 20
_OFF_LIGHT = 24
_OFF_HEIGHT = 36

# --- measurement-conditions uint32 indices, relative to the setting offset ---
_MC_YEAR = 1
_MC_X_PER_PIXEL = 42   # picometres per pixel, X
_MC_Y_PER_PIXEL = 43   # picometres per pixel, Y
_MC_Z_PER_DIGIT = 44   # picometres per height digit

# --- data-block geometry ---
_BLOCK_HEADER_BYTES = 28    # 7 x uint32
_PALETTE_BYTES = 768        # 256-entry RGB LUT preceding the raw data
_DTYPE_FOR_DEPTH = {8: np.uint8, 16: np.uint16, 32: np.uint32}


@dataclass
class VK4:
    """Decoded VK4 measurement."""

    path: str
    width: int
    height: int
    bit_depth: int
    x_um_per_px: float
    y_um_per_px: float
    z_um_per_digit: float
    height_raw: np.ndarray          # uint, "digits", shape (rows, cols)
    intensity: np.ndarray | None    # laser-intensity image, or None
    range_min: int
    range_max: int
    datetime: tuple[int, int, int, int, int, int]  # (Y, M, D, h, m, s)

    @property
    def height_um(self) -> np.ndarray:
        """Height map in micrometres (float64)."""
        return self.height_raw.astype(np.float64) * self.z_um_per_digit

    @property
    def extent_um(self) -> tuple[float, float, float, float]:
        """(x0, x1, y0, y1) physical extent in um, for matplotlib imshow."""
        return (0.0, self.width * self.x_um_per_px,
                0.0, self.height * self.y_um_per_px)

    def __repr__(self) -> str:
        return (f"VK4({Path(self.path).name}, {self.width}x{self.height}, "
                f"{self.x_um_per_px:.4f} um/px, "
                f"{self.z_um_per_digit*1000:.3f} nm/digit)")


def read_vk4(path: str | Path, *, load_intensity: bool = True) -> VK4:
    """Read a VK4 file and return a :class:`VK4`.

    Raises ValueError if the file is not a VK4 or uses a layout this reader
    was not validated against (e.g. compressed data blocks).
    """
    raw = Path(path).read_bytes()
    if raw[:4] != _MAGIC:
        raise ValueError(f"{path}: not a VK4 file (magic={raw[:4]!r})")

    def u32(byte_off: int) -> int:
        if byte_off < 0 or byte_off + 4 > len(raw):        # bounds-check so a truncated/corrupt
            raise ValueError(f"{path}: truncated or corrupt VK4 "  # file gives a clear error, not
                             f"(offset {byte_off} past end, len={len(raw)})")  # a raw struct.error
        return struct.unpack_from("<I", raw, byte_off)[0]

    setting = u32(_OFF_SETTING)
    height_off = u32(_OFF_HEIGHT)
    light_off = u32(_OFF_LIGHT)

    x_pm = u32(setting + _MC_X_PER_PIXEL * 4)
    y_pm = u32(setting + _MC_Y_PER_PIXEL * 4)
    z_pm = u32(setting + _MC_Z_PER_DIGIT * 4)
    # Sanity-check the calibration factors (picometres). A shifted / foreign setting-block layout can
    # land these on a reserved 0 or the wrong field, which would silently emit 0 or mis-scaled heights
    # (every reported depth then wrong, with no error). Plausible bands: x/y 0.001-100 um/px, z 1 pm
    # - 1 um/digit. Fail loudly on an unrecognised layout rather than trust a bad scale.
    for _nm, _v, _lo, _hi in (("x_per_pixel", x_pm, 1e3, 1e8), ("y_per_pixel", y_pm, 1e3, 1e8),
                              ("z_per_digit", z_pm, 1, 1e6)):
        if not (_lo <= _v <= _hi):
            raise ValueError(f"{path}: implausible {_nm} = {_v} pm (expected {_lo:g}-{_hi:g} pm); "
                             f"the VK4 setting-block layout may be unrecognised -- refusing to emit "
                             f"a mis-scaled height field.")
    dt = tuple(u32(setting + (_MC_YEAR + i) * 4) for i in range(6))  # Y,M,D,h,m,s

    def read_block(block_off: int) -> tuple[np.ndarray, int, int, int, int, int]:
        if block_off < 0 or block_off + 28 > len(raw):     # 7 * uint32 header; guard before unpack
            raise ValueError(f"{path}: truncated or corrupt VK4 "
                             f"(block@{block_off} header past end, len={len(raw)})")
        w, h, bit_depth, _compression, byte_size, rmin, rmax = struct.unpack_from(
            "<7I", raw, block_off
        )
        if bit_depth not in _DTYPE_FOR_DEPTH:
            raise ValueError(f"{path}: unsupported bit depth {bit_depth}")
        bytes_pp = bit_depth // 8
        expected = w * h * bytes_pp
        if byte_size != expected:
            raise ValueError(
                f"{path}: block@{block_off} byte_size {byte_size} != w*h*bpp "
                f"{expected}. Compressed VK4 blocks are not supported."
            )
        data_start = block_off + _BLOCK_HEADER_BYTES + _PALETTE_BYTES
        if data_start + byte_size > len(raw):
            raise ValueError(
                f"{path}: block@{block_off} data runs past end of file."
            )
        arr = np.frombuffer(
            raw, dtype=_DTYPE_FOR_DEPTH[bit_depth], count=w * h, offset=data_start
        ).reshape(h, w)
        return arr, w, h, bit_depth, rmin, rmax

    hgt, w, h, bit_depth, rmin, rmax = read_block(height_off)

    intensity = None
    if load_intensity and light_off:
        try:
            intensity = read_block(light_off)[0].copy()
        except ValueError as e:                          # present but unparseable (e.g. compressed):
            print(f"WARNING: {path}: intensity block present (light_off={light_off}) but unreadable "
                  f"({e}); continuing without an intensity channel.")   # surface it, don't fail the read
            intensity = None  # intensity is optional; never fail the whole read

    return VK4(
        path=str(path),
        width=w,
        height=h,
        bit_depth=bit_depth,
        x_um_per_px=x_pm / 1e6,
        y_um_per_px=y_pm / 1e6,
        z_um_per_digit=z_pm / 1e6,
        height_raw=hgt.copy(),           # copy: frombuffer view is read-only
        intensity=intensity,
        range_min=rmin,
        range_max=rmax,
        datetime=dt,
    )


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        vk = read_vk4(p)
        z = vk.height_um
        print(vk)
        print(f"  datetime  : {vk.datetime}")
        print(f"  FOV        : {vk.width*vk.x_um_per_px:.1f} x "
              f"{vk.height*vk.y_um_per_px:.1f} um")
        print(f"  height um  : min {z.min():.2f}  max {z.max():.2f}  "
              f"median {np.median(z):.2f}")
