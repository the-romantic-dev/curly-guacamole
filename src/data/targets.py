import numpy as np
import torch


def mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
    """Бинаризует mask и возвращает тензор [1, H, W]."""
    binary_mask = (mask >= 0.5).astype(np.float32)
    return torch.from_numpy(binary_mask).unsqueeze(0)


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
