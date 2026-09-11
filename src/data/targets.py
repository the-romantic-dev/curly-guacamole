import numpy as np
import torch


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Бинаризует mask и возвращает тензор [1, H, W]."""
    binary_mask = (mask >= 0.5).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


class SignedDistanceTarget:
    """Kervadec-style foreground signed distances, divided by valid diagonal.

    Build after augmentation/resizing. Interior boundary pixels have distance
    zero; deeper foreground is negative and background positive. Empty/full
    masks have no observed interface and produce zeros. Padding is excluded.
    """

    def __call__(self, mask: torch.Tensor, valid_mask=None) -> torch.Tensor:
        from scipy.ndimage import distance_transform_edt

        binary = mask.squeeze(0).numpy() >= .5
        result = np.zeros(binary.shape, dtype=np.float32)
        valid = np.ones_like(binary) if valid_mask is None else valid_mask.squeeze(0).numpy() > .5
        ys, xs = np.nonzero(valid)
        if not len(ys):
            return torch.from_numpy(result).unsqueeze(0)
        region = np.s_[ys.min():ys.max()+1, xs.min():xs.max()+1]
        if not valid[region].all():
            raise ValueError('Boundary targets require rectangular valid content')
        foreground = binary[region]
        if foreground.any() and not foreground.all():
            outside = distance_transform_edt(~foreground)
            inside = distance_transform_edt(foreground)
            signed = outside * ~foreground - (inside - 1) * foreground
            result[region] = signed / np.hypot(*foreground.shape)
        return torch.from_numpy(result).unsqueeze(0)


# def build_component_map(mask: np.ndarray, max_components: int) -> torch.Tensor:
#     """Строит карту connected components фиксированного диапазона меток."""
#     binary_mask = (mask >= 0.5).astype(np.uint8)
#     count, labels = cv2.connectedComponents(binary_mask, connectivity=8)
#
#     if count > max_components:
#         labels[labels >= max_components] = 0
#
#     return torch.from_numpy(labels.astype(np.int16)).unsqueeze(0)
#
#
# def build_distance_map(mask: np.ndarray) -> torch.Tensor:
#     """Строит нормированное расстояние до положительной области маски."""
#     binary_mask = (mask >= 0.5).astype(np.uint8)
#
#     if binary_mask.any():
#         distance = cv2.distanceTransform(1 - binary_mask, cv2.DIST_L2, 3)
#         distance /= float(np.hypot(*binary_mask.shape))
#     else:
#         distance = np.zeros(binary_mask.shape, dtype=np.float32)
#
#     return torch.from_numpy(distance.astype(np.float32)).unsqueeze(0)
