from torch.utils.data import default_collate


class ValidationCollator:
    """Stack model inputs, retaining original masks with different spatial sizes."""

    def __call__(self, samples):
        batch = default_collate([
            {key: value for key, value in sample.items() if key != "original_mask"}
            for sample in samples
        ])
        if "original_mask" in samples[0]:
            batch["original_mask"] = [sample["original_mask"] for sample in samples]
        return batch
