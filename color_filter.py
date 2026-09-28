"""Color summaries of original candidate pixels, independent of DINO features."""
import numpy as np
from PIL import Image


def summarize_rgb(rgb):
    rgb = np.asarray(rgb, dtype=np.float32)
    r, g, b = np.moveaxis(rgb, -1, 0)
    high = rgb.max(axis=-1)
    chroma = high - rgb.min(axis=-1)
    # Red and magenta with a red-dominant channel; exclude nearly neutral pixels.
    valid = (r >= g) & (r >= b) & (chroma >= 12) & (chroma / np.maximum(high, 1) >= .12)
    hue = 60 * (g - b) / np.maximum(chroma, 1)
    score = np.floor(100 * (1 - np.maximum(hue, 0) / 60)).clip(0, 100).astype(int)
    hist = np.bincount(score[valid], minlength=101)
    return {'version': 1, 'pixels': int(r.size), 'hist': hist.tolist()}


def region_color(path, x, y, width, height):
    with Image.open(path) as im:
        if not (width > 0 and height > 0 and x >= 0 and y >= 0 and x + width <= im.width and y + height <= im.height):
            raise ValueError('颜色检查区域超出图片范围')
        return summarize_rgb(im.convert('RGB').crop((x, y, x + width, y + height)))
