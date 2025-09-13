"""Various utilitary functions."""

import io

import PIL
import numpy as np


def jpeg_compress(img, quality):
    """Applies JPEG compression to an image."""
    img = img.copy()
    pimg = PIL.Image.fromarray(img.astype(np.uint8))
    out = io.BytesIO()
    pimg.save(out, format='JPEG', quality=quality)
    out.seek(0)
    result = np.array(PIL.Image.open(out))
    return result

