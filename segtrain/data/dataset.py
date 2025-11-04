import numpy as np, tensorflow as tf
from pathlib import Path
from typing import List, Tuple
from PIL import Image
from skimage.segmentation import find_boundaries

from segtrain.data.labelmap import build_color_to_index, mask_rgb_to_index
from segtrain.data.augment import AdvancedAugmentation

"""
VOClike multi-root dataset and tf.data pipeline: loads RGB images + color masks, converts masks to class indices, and builds boundary targets.
Applies train/eval preprocessing & augmentations (random scale/crop/flip, optional AdvancedAugmentation) with ImageNet-style normalization.
Returns (img, {"sem_logits": mask, "boundary_logits": boundary}); also provides class-weight computation for class imbalance.
"""


def make_boundary_targets(mask_batch: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = []
    for m in mask_batch:
        mm = m.copy(); mm[mm == ignore_index] = 0
        b = find_boundaries(mm, mode="inner").astype(np.float32)
        out.append(b[..., None])
    return np.stack(out, axis=0)

class EnhancedMultiRootVOCDataset:
    def __init__(self, roots: List[str], image_set: str,
                 names: List[str], colors: List[Tuple[int,int,int]],
                 crop_size: int = 512, random_scale=(0.5, 2.0),
                 hflip_prob: float = 0.5, ignore_index: int = 255,
                 use_advanced_aug: bool = True):
        self.roots = [Path(r) for r in roots]
        self.image_set = image_set
        self.names, self.colors = names, colors
        self.ignore_index = ignore_index
        self.crop_size, self.random_scale, self.hflip_prob = crop_size, random_scale, hflip_prob
        self.color_to_index = build_color_to_index(colors)
        self.use_advanced_aug = use_advanced_aug and (image_set == "train")
        if self.use_advanced_aug: self.aug = AdvancedAugmentation()
        self.samples = []
        for root in self.roots:
            set_file = root / "ImageSets" / "Segmentation" / f"{image_set}.txt"
            ids = [s.strip() for s in set_file.read_text().splitlines() if s.strip()]
            for img_id in ids: self.samples.append((root, img_id))
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self): return len(self.samples)

    def _load_sample(self, root: Path, img_id: str):
        img_dir, mask_dir = root / "JPEGImages", root / "SegmentationClass"
        img_path = img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            alt = img_dir / f"{img_id}.png"; img_path = alt if alt.exists() else img_path
        mask_path = mask_dir / f"{img_id}.png"
        image = Image.open(img_path).convert("RGB")
        mask_rgb = Image.open(mask_path)
        mask = mask_rgb_to_index(mask_rgb, self.color_to_index, ignore_index=self.ignore_index)
        return image, mask

    def _random_resize(self, img, mask):
        if self.random_scale:
            s = np.random.uniform(*self.random_scale)
            new_w, new_h = int(img.width*s), int(img.height*s)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            mask = Image.fromarray(mask, "L").resize((new_w, new_h), Image.NEAREST)
            mask = np.array(mask, dtype=np.int64)
        return img, mask

    def _random_crop(self, img, mask):
        th, tw = self.crop_size, self.crop_size
        if img.height < th or img.width < tw:
            pad_h, pad_w = max(0, th - img.height), max(0, tw - img.width)
            img = Image.fromarray(np.pad(np.array(img), ((0,pad_h),(0,pad_w),(0,0)), mode="constant").astype(np.uint8))
            mask = np.pad(mask, ((0,pad_h),(0,pad_w)), mode="constant", constant_values=self.ignore_index)
        i = np.random.randint(0, img.height - th + 1); j = np.random.randint(0, img.width - tw + 1)
        return img.crop((j, i, j+tw, i+th)), mask[i:i+th, j:j+tw]

    def _hflip(self, img, mask):
        if np.random.rand() < self.hflip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT); mask = mask[:, ::-1]
        return img, mask

    def _center_crop_or_resize(self, img, mask):
        short = min(img.width, img.height)
        if short < self.crop_size:
            s = self.crop_size / short
            img = img.resize((int(img.width*s), int(img.height*s)), Image.BILINEAR)
            mask = Image.fromarray(mask, "L").resize((int(mask.shape[1]*s), int(mask.shape[0]*s)), Image.NEAREST)
            mask = np.array(mask, dtype=np.int64)
        th, tw = self.crop_size, self.crop_size
        i = max(0, (img.height - th)//2); j = max(0, (img.width - tw)//2)
        return img.crop((j, i, j+tw, i+th)), mask[i:i+th, j:j+tw]

    def get_item(self, idx):
        root, img_id = self.samples[idx]
        img, mask = self._load_sample(root, img_id)
        if self.image_set == "train":
            img, mask = self._random_resize(img, mask)
            img, mask = self._random_crop(img, mask)
            img, mask = self._hflip(img, mask)
            img_np = np.asarray(img, dtype=np.float32) / 255.0
            if self.use_advanced_aug: img_np, mask = self.aug.apply(img_np, mask)
        else:
            img, mask = self._center_crop_or_resize(img, mask)
            img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        return img_np, mask.astype(np.int64)

def make_tf_dataset(voc: EnhancedMultiRootVOCDataset, batch_size: int, shuffle: bool, ignore_index: int):
    indices = np.arange(len(voc), dtype=np.int32)

    def _py_load(idx):
        img, mask = voc.get_item(int(idx))
        bt = make_boundary_targets(np.expand_dims(mask, 0), ignore_index=ignore_index)[0]
        return img.astype(np.float32), mask.astype(np.int32), bt.astype(np.float32)

    def _tf_map(idx):
        img, mask, bt = tf.numpy_function(_py_load, [idx], [tf.float32, tf.int32, tf.float32])
        img.set_shape([None, None, 3]); mask.set_shape([None, None]); bt.set_shape([None, None, 1])
        return img, {"sem_logits": mask, "boundary_logits": bt}

    ds = tf.data.Dataset.from_tensor_slices(indices)
    if shuffle: ds = ds.shuffle(buffer_size=len(voc), reshuffle_each_iteration=True)
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def compute_class_weights(masks, num_classes: int, ignore_index: int = 255):
    total_pixels = 0; class_counts = np.zeros(num_classes, dtype=np.float64)
    for mask in masks:
        valid = mask != ignore_index
        total_pixels += valid.sum()
        for c in range(num_classes):
            class_counts[c] += (mask == c).sum()
    cw = total_pixels / (num_classes * class_counts + 1e-6)
    cw = cw / cw.sum() * num_classes
    return cw.astype(np.float32)
