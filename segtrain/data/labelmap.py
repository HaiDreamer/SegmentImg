from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image

'''read labelmap file'''

def read_labelmap(labelmap_path: Path):
    if not labelmap_path.exists():
        raise FileNotFoundError(f"File not found: {labelmap_path}")
    names, colors = [], []
    for raw in Path(labelmap_path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"): continue
        if ":" not in line: raise ValueError(f"Missing colon in line: {line}")
        name, rest = line.split(":", 1)
        comps = rest.split(":", 1)[0].split(",")
        if len(comps) != 3: raise ValueError(f"RGB must have 3 components: {line}")
        r,g,b = [int(c.strip()) for c in comps]
        names.append(name.strip()); colors.append((r,g,b))
    return names, colors

def build_color_to_index(colors: List[Tuple[int,int,int]]) -> Dict[Tuple[int,int,int], int]:
    return {tuple(map(int, c)): i for i, c in enumerate(colors)}

def mask_rgb_to_index(mask_img: Image.Image, color_to_index: Dict[Tuple[int,int,int], int], ignore_index=255):
    '''Converts a color (RGB) mask image into a single-channel index mask'''
    m = np.array(mask_img.convert("RGB"), dtype=np.uint8)
    h, w, _ = m.shape
    flat = m.reshape(-1, 3)
    out = np.full((h*w,), ignore_index, dtype=np.uint8)
    keys = (flat[:,0].astype(np.int32) << 16) | (flat[:,1].astype(np.int32) << 8) | flat[:,2].astype(np.int32)
    lut = { (r<<16)|(g<<8)|b: idx for (r,g,b), idx in color_to_index.items() }
    for k, idx in lut.items():
        out[keys == k] = idx
    return out.reshape(h, w)
