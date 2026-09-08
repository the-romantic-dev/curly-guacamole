"""Train-only original inventory, exact deduplication and conservative pair checks."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from src.data.sample_io import read_image


class OriginalIndex:
    """Accept an original when at least one supplied pair matches outside its GT.

    Pixel hashes find exact decoded duplicates, not perceptual near-duplicates.
    Pair matching is a conservative geometric check, not proof of image provenance.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _path(self, value: str) -> Path:
        path = (self.root / str(value).replace('\\', '/')).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('image path escapes train root')
        return path

    def _probe(self, value: str) -> dict:
        result = {'orgl_img_path': value, 'readable': False}
        try:
            path = self._path(value)
            image = read_image(path)
            if image is None or min(image.shape[:2]) < 8:
                return result
            digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            with Image.open(path) as jpeg:
                table = getattr(jpeg, 'quantization', {}).get(0)
            result.update(readable=True, pixel_hash=digest, img_h=image.shape[0], img_w=image.shape[1],
                          q_kind=('unit' if max(table) == min(table) == 1 else 'nonunit') if table else 'unknown',
                          q_min=min(table) if table else None, q_max=max(table) if table else None,
                          thumbnail=cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA))
        except (OSError, ValueError):
            pass
        return result

    def _match(self, task: tuple[dict, dict]) -> dict:
        row, original = task
        result = {key: row[key] for key in ('orgl_img_path', 'chng_img_path', 'gt_path', 'group_id', 'domain')}
        result.update(status='unreadable_original', pixel_hash=original.get('pixel_hash'),
                      correlation=None, outside_mae=None)
        if not original['readable']:
            return result
        try:
            changed = read_image(self._path(row['chng_img_path']))
            mask = read_image(self._path(row['gt_path']), cv2.IMREAD_GRAYSCALE)
            if changed is None or mask is None:
                result['status'] = 'unreadable_pair'
                return result
            thumb = cv2.resize(changed, (64, 64), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_AREA)
            outside = cv2.dilate((mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) == 0
            if outside.mean() < .1:
                result['status'] = 'insufficient_background'
                return result
            a, b = original['thumbnail'][outside].astype(float), thumb[outside].astype(float)
            mae = float(np.abs(a - b).mean() / 255)
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if min(a.std(), b.std()) > 1 else 0.0
            result.update(correlation=corr, outside_mae=mae,
                          status='accepted' if mae <= .15 and corr >= .8 else 'unmatched')
        except (OSError, ValueError):
            result['status'] = 'unreadable_pair'
        return result

    def build(self, metadata: pd.DataFrame, workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
        rows = metadata.loc[metadata.orgl_img_path.notna() & metadata.orgl_img_path.astype(str).str.strip().ne('')]
        paths = sorted(rows.orgl_img_path.unique())
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            probes = dict(zip(paths, pool.map(self._probe, paths), strict=True))
            pairs = pd.DataFrame(pool.map(self._match, ((row, probes[row['orgl_img_path']])
                                                       for row in rows.to_dict('records'))))
        records = []
        for _digest, linked in pairs.loc[pairs.pixel_hash.notna()].groupby('pixel_hash', sort=True):
            accepted = linked.loc[linked.status == 'accepted']
            if accepted.empty:
                continue
            path = sorted(accepted.orgl_img_path.unique())[0]
            info = {key: value for key, value in probes[path].items() if key not in {'thumbnail', 'readable'}}
            records.append(dict(info, chng_img_path=path, gt_path=None, target_kind='original_zero',
                                group_ids=sorted(linked.group_id.unique()),
                                source_paths=sorted(linked.orgl_img_path.unique()),
                                group_id=sorted(linked.group_id.unique())[0],
                                domain='|'.join(sorted(linked.domain.unique())), generator='original',
                                is_negative=True, mask_area=0., broken=False, size_mismatch=False,
                                height=info['img_h'], width=info['img_w'], stem=Path(path).stem))
        return pd.DataFrame(records), pairs
