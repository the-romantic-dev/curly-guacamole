"""Constants used by DCT forensic features."""

import numpy as np

STRIDE = 8
EPS = 1e-3


CHANNELS = (
    "y31",
    "y31_rel",
    "y42",
    "y42_rel",
    "c31",
    "c31_rel",
    "cy3",
    "cy3_rel",
    "latt",
    "nz_rel",
    "mid_rel",
    "qt",
)

CHANNEL_COUNT = len(CHANNELS)


CHANNEL_RANGE = np.array(
    [
        8.0,   # y31
        4.0,   # y31_rel
        8.0,   # y42
        4.0,   # y42_rel
        8.0,   # c31
        4.0,   # c31_rel
        8.0,   # cy3
        4.0,   # cy3_rel
        0.25,  # latt
        16.0,  # nz_rel
        6.0,   # mid_rel
        4.0,   # qt
    ],
    dtype=np.float32,
)


ZIGZAG = np.array(
    [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ],
    dtype=np.int64,
)


_u, _v = np.mgrid[0:8, 0:8]
_frequency_radius = _u + _v

BAND_LOW = (_frequency_radius >= 1) & (_frequency_radius <= 2)
BAND_MID = (_frequency_radius >= 3) & (_frequency_radius <= 5)
BAND_HIGH = (_frequency_radius >= 6) & (_frequency_radius <= 9)
BAND_TOP = _frequency_radius >= 10
