"""DCT forensic feature extraction."""

import numpy as np

from .constants import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MID,
    BAND_TOP,
    EPS,
)


def relative_to_frame_median(feature: np.ndarray) -> np.ndarray:
    """Express a block-wise feature relative to the frame median."""
    return feature - np.median(feature)


def band_energy(
    power: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """Sum DCT energy inside a frequency band."""
    return np.einsum(
        "hwij,ij->hw",
        power,
        band.astype(np.float32),
        optimize=True,
    )


def log_band_energy(
    power: np.ndarray,
    band: np.ndarray,
) -> np.ndarray:
    """Compute log2 DCT energy inside a frequency band."""
    return np.log2(band_energy(power, band) + EPS)


def spectral_features(
    dct_y: np.ndarray,
    dct_cr: np.ndarray,
    dct_cb: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Compute frequency-energy forensic features."""
    power_y = dct_y ** 2
    power_c = dct_cr ** 2 + dct_cb ** 2

    y_low = log_band_energy(power_y, BAND_LOW)
    y_mid = log_band_energy(power_y, BAND_MID)
    y_high = log_band_energy(power_y, BAND_HIGH)
    y_top = log_band_energy(power_y, BAND_TOP)

    c_low = log_band_energy(power_c, BAND_LOW)
    c_high = log_band_energy(power_c, BAND_HIGH)

    y31 = y_high - y_low
    y42 = y_top - y_mid
    c31 = c_high - c_low
    cy3 = c_high - y_high

    return y31, y42, c31, cy3, y_mid


def quantization_features(
    dct_y: np.ndarray,
    qtable: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute features related to the JPEG quantization grid."""
    quantized = dct_y / qtable[None, None, :, :]

    latt = np.abs(
        quantized - np.round(quantized)
    ).mean(axis=(2, 3))

    nz = (
        np.abs(quantized) > 0.5
    ).sum(axis=(2, 3)).astype(np.float32)

    qt = np.log2(float(qtable[0, 0]))

    return latt, nz, qt


def build_feature_maps(
    dct_y: np.ndarray,
    dct_cr: np.ndarray,
    dct_cb: np.ndarray,
    qtable: np.ndarray,
) -> np.ndarray:
    """Build all 12 DCT forensic channels."""
    y31, y42, c31, cy3, y_mid = spectral_features(
        dct_y,
        dct_cr,
        dct_cb,
    )

    latt, nz, qt = quantization_features(
        dct_y,
        qtable,
    )

    channels = [
        y31,
        relative_to_frame_median(y31),

        y42,
        relative_to_frame_median(y42),

        c31,
        relative_to_frame_median(c31),

        cy3,
        relative_to_frame_median(cy3),

        latt,
        relative_to_frame_median(nz),
        relative_to_frame_median(y_mid),

        np.full_like(y31, qt),
    ]

    return np.stack(channels).astype(np.float32)
