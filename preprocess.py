# preprocess.py
import numpy as np


def compute_cross_difference(img: np.ndarray) -> np.ndarray:
    """
    Compute the cross-difference (a 2x2 alternating-sum absolute response) of an image.

    For an input image I, the cross-difference at integer location (y, x) is:
        | I[y, x] + I[y+1, x+1] - I[y, x+1] - I[y+1, x] |

    This is computed wherever the 2x2 neighborhood is fully inside the image,
    so the output has spatial shape (Y-1, X-1) for a grayscale image of shape (Y, X),
    or (Y-1, X-1, C) for a color image of shape (Y, X, C).

    Notes
    -----
    - The operation is fully vectorized and runs in O(Y*X).
    - The input is converted to float32 to avoid integer overflow and preserve precision.
    - Works for 2D (grayscale) and 3D (multi-channel) images.

    Parameters
    ----------
    img : np.ndarray
        Input image array with shape (Y, X) or (Y, X, C). Any numeric dtype is accepted.

    Returns
    -------
    np.ndarray
        Cross-difference map with shape:
            (Y-1, X-1)       if input is (Y, X)
            (Y-1, X-1, C)    if input is (Y, X, C)
        The dtype is float32.

    Raises
    ------
    ValueError
        If the input has fewer than 2 rows or columns, or if the array is not 2D/3D.
        If Y or X is odd.
    """
    if img.ndim not in (2, 3):
        raise ValueError(
            f"`img` must be 2D (Y, X) or 3D (Y, X, C), got shape {img.shape} with ndim={img.ndim}."
        )

    y, x = img.shape[:2]
    if y < 2 or x < 2:
        raise ValueError(
            f"`img` must have at least 2 rows and 2 columns, got shape {img.shape}."
        )

    # Convert to float32 to avoid overflow/underflow with integer inputs.
    I = img.astype(np.float32, copy=False)

    # Define 2x2 neighborhood slices:
    A = I[:-1, :-1]  # top-left
    B = I[1:, 1:]    # bottom-right
    C = I[:-1, 1:]   # top-right
    D = I[1:, :-1]   # bottom-left

    # Cross-difference (absolute alternating sum over the 2x2 window).
    cross = np.abs(A + B - C - D)

    return cross


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    """
    Ensure uint8 data (0..255). If already uint8, returns a contiguous view.
    If float/integer, clips and scales if needed.
    """
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    # Heuristic: if float in [0,1], scale to [0,255]; otherwise clip to [0,255]
    img = image
    if np.issubdtype(img.dtype, np.floating):
        # Try detect [0,1]
        finite = np.isfinite(img)
        if finite.any():
            vmin = float(np.nanmin(img[finite]))
            vmax = float(np.nanmax(img[finite]))
        else:
            vmin, vmax = 0.0, 1.0
        if 0.0 <= vmin and vmax <= 1.0:
            img = (img * 255.0).round()
        img = np.clip(img, 0.0, 255.0)
    else:
        img = np.clip(img, 0, 255)
    return np.ascontiguousarray(img.astype(np.uint8, copy=False))




def _trim_borders_after_filter(
    image: np.ndarray, method: str, rank_sz: int
) -> np.ndarray:
    """
    Remove border rows/columns that may suffer from border effects, depending on the filter.

    Parameters
    ----------
    image : np.ndarray
        Filtered image, shape (Y, X) or (Y, X, C).
    method : str
        Either "cross" (cross-difference) or "rank".
    rank_sz : int
        Half-window size used by the rank transform. Ignored if method == "cross".

    Returns
    -------
    np.ndarray
        Border-trimmed image.

    Notes
    -----
    - Cross-difference (as implemented) already reduces size by 1 on each axis
      via slicing. We additionally drop the **last** row and **last** column
      to avoid any residual 2x2 neighborhood touching the far border.
      => remove final row/col: `image[:-1, :-1, ...]` → overall -2 on each axis from original.
    - Rank transform leaves a `rank_sz`-wide border uncomputed (zeros). Remove
      `rank_sz` rows/cols from both the beginning and the end.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}.")

    if method == "cross":
        # Remove last row & last column (on top of the inherent -1 from the operator)
        if image.shape[0] < 2 or image.shape[1] < 2:
            raise ValueError("Image too small to trim borders after cross-difference.")
        return image[:-1, :-1] if image.ndim == 2 else image[:-1, :-1, :]

    if method == "rank":
        if rank_sz < 1:
            raise ValueError(f"`rank_sz` must be >= 1 for rank transform, got {rank_sz}.")
        y0 = rank_sz
        y1 = image.shape[0] - rank_sz
        x0 = rank_sz
        x1 = image.shape[1] - rank_sz
        if y1 <= y0 or x1 <= x0:
            raise ValueError(
                f"Image too small ({image.shape}) to remove rank borders with sz={rank_sz}."
            )
        return image[y0:y1, x0:x1] if image.ndim == 2 else image[y0:y1, x0:x1, :]

    raise ValueError(f"Unknown method '{method}'. Use 'cross' or 'rank'.")


def _crop_bottom_right_to_multiple(image: np.ndarray, P: int) -> np.ndarray:
    """
    Crop the image (only bottom and right) so that Y and X are multiples of P.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (Y, X) or (Y, X, C).
    P : int
        Period grouping (8 or 16).

    Returns
    -------
    np.ndarray
        Cropped image with Y % P == 0 and X % P == 0.

    Raises
    ------
    ValueError
        If the resulting size would be zero along any axis.
    """
    Y, X = image.shape[:2]
    newY = (Y // P) * P
    newX = (X // P) * P
    if newY == 0 or newX == 0:
        raise ValueError(
            f"After cropping to multiples of {P}, size would be ({newY}, {newX}). "
            f"Current size is ({Y}, {X}). Increase input size or adjust preprocessing."
        )
    if image.ndim == 2:
        return image[:newY, :newX]
    return image[:newY, :newX, :]


def _fft_magnitude(image: np.ndarray) -> np.ndarray:
    """
    Compute centered FFT magnitude channel-wise.

    Parameters
    ----------
    image : np.ndarray
        Real image, shape (Y, X) or (Y, X, C), float32/float64.

    Returns
    -------
    np.ndarray
        Magnitude spectrum with `fftshift` applied, same shape as input.
        If input is 2D -> (Y, X); if 3D -> (Y, X, C).
    """
    if image.ndim == 2:
        F = np.fft.fft2(image)
        return np.abs(np.fft.fftshift(F))
    elif image.ndim == 3:
        Y, X, C = image.shape
        mag = np.empty_like(image, dtype=np.float64)
        for c in range(C):
            F = np.fft.fft2(image[..., c])
            mag[..., c] = np.abs(np.fft.fftshift(F))
        return mag
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D.")

def _sample_harmonic_grid(mag: np.ndarray, P: int) -> np.ndarray:
    """
    Sample FFT magnitudes on a harmonic grid defined by step (Y/P, X/P), using a centered,
    fftshifted magnitude spectrum.

    Columns (u): nonnegative harmonics n = 0..P/2  -> indices: (cx + n*step_x) % X
    Rows    (v): symmetric     harmonics m = -P/2..P/2 -> indices: (cy + m*step_y) % Y

    Counts per channel: (P+1) * (P/2 + 1)
      P=8  ->  9 * 5  = 45
      P=16 -> 17 * 9  = 153

    If `mag` is grayscale, the single-channel vector is repeated 3x (to match RGB length).
    If `mag` is RGB, channels are sampled independently and concatenated in order.
    """
    if mag.ndim not in (2, 3):
        raise ValueError(f"Expected magnitude of shape (Y, X) or (Y, X, C), got {mag.shape}.")

    Y, X = mag.shape[:2]
    if (Y % P) != 0 or (X % P) != 0:
        raise ValueError(f"FFT size must be multiples of P={P}. Got (Y, X)=({Y}, {X}).")

    cy, cx = Y // 2, X // 2
    step_y = Y // P
    step_x = X // P

    # m in [-P/2 .. +P/2] (inclusive): P+1 rows
    # n in [0 .. P/2]      (inclusive): P/2+1 columns (one-sided in x)
    m_vals = np.arange(-P // 2, P // 2 + 1, dtype=int)
    n_vals = np.arange(0, P // 2 + 1, dtype=int)

    rows = (cy + m_vals * step_y) % Y   # (P+1,)   includes top edge (Nyquist) at 0
    cols = (cx + n_vals * step_x) % X   # (P/2+1,) includes left edge (Nyquist) at 0

    if mag.ndim == 2:
        # Single-channel: sample then repeat 3x to emulate RGB length
        samples = []
        for r in rows:
            for c in cols:
                samples.append(mag[r, c])
        vec = np.asarray(samples, dtype=np.float64)  # shape (P+1)*(P//2+1,)
        return np.tile(vec, 3)

    # Multi-channel: sample each channel and concatenate
    C = mag.shape[2]
    samples_per_c = (P + 1) * (P // 2 + 1)
    out = np.empty(C * samples_per_c, dtype=np.float64)
    k = 0
    for ch in range(C):
        for r in rows:
            for c in cols:
                out[k] = mag[r, c, ch]
                k += 1
    return out




def preprocess_for_fft_features(
    image: np.ndarray,
    method: str = "cross",
    rank_sz: int = 3,
    max_period: int = 8,
) -> np.ndarray:
    """
    Main preprocessing for Synthbuster features.

    Steps:
      1) Filter: either cross-difference or rank-transform (channel-wise for RGB).
      2) Remove border regions prone to filter border effects:
         - cross: drop last row and last column (further to the inherent -1 shrink).
         - rank : drop `rank_sz` rows/cols from both the start and the end.
      3) Crop bottom/right so (Y, X) are multiples of `max_period` (8 or 16).
      4) FFT2 + fftshift, magnitude per channel.
      5) Sample magnitudes on a harmonic grid defined by step (Y/max_period, X/max_period),
         using rows m=-P/2..+P/2 and columns n=0..P/2.
      6) Concatenate across channels. If grayscale, repeat single-channel vector 3x.
         Final size is:
           - 3 * 45  = 135 for P=8
           - 3 * 153 = 459 for P=16

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (Y, X) or (Y, X, 3). Any numeric dtype; converted to float32.
    method : str
        "cross" for cross-difference, or "rank" for rank transform.
    rank_sz : int
        Half-window size for rank transform (ignored for "cross").
    max_period : int
        8 or 16. Determines the harmonic sampling grid.

    Returns
    -------
    np.ndarray
        1D feature vector of length 135 (P=8) or 459 (P=16), dtype float64.

    Raises
    ------
    ValueError
        For invalid args or insufficient sizes.
    """
    if method not in ("cross", "rank"):
        raise ValueError(f"`method` must be 'cross' or 'rank', got {method}.")
    if max_period not in (8, 16):
        raise ValueError(f"`max_period` must be 8 or 16, got {max_period}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"`image` must be 2D or 3D, got shape {image.shape}.")

    # Ensure float32 and contiguous for FFT
    img_f32 = np.ascontiguousarray(image.astype(np.float32, copy=False))

    # 1) Apply the selected filter (channel-wise if needed)
    if method == "cross":
        filtered = compute_cross_difference(img_f32)
    else:
        filtered = rank_transform(img_f32, sz=rank_sz)

    # 2) Remove border regions prone to border effects
    filtered = _trim_borders_after_filter(filtered, method=method, rank_sz=rank_sz)

    # 3) Crop to multiples of max_period (bottom/right)
    filtered = _crop_bottom_right_to_multiple(filtered, max_period)

    # Sanity check on final size
    Y, X = filtered.shape[:2]
    if Y == 0 or X == 0:
        raise ValueError("Empty image after preprocessing; check earlier steps.")

    # 4) FFT magnitude (centered)
    mag = _fft_magnitude(filtered)

    # 5–6) Sample harmonic grid and concatenate across channels
    features = _sample_harmonic_grid(mag, P=max_period)

    # Final length check
    expected = 3 * ((max_period + 1) * (max_period // 2 + 1))
    if features.size != expected:
        raise RuntimeError(
            f"Unexpected feature length {features.size}, expected {expected} "
            f"for max_period={max_period}."
        )

    return features




if __name__ == "__main__":
    """
    Quick self-test:
    - Create synthetic grayscale (Y, X) and RGB (Y, X, 3) images with even dimensions.
    - Run compute_cross_difference and rank_transform on both.
    - Print basic stats to verify outputs.
    """
    import numpy as np

    # Reproducibility
    rng = np.random.default_rng(seed=42)

    # Use even dimensions (required by compute_cross_difference)
    Y, X = 128, 192

    # Synthetic images in uint8 range to mimic typical 8-bit images
    gray = rng.integers(low=0, high=256, size=(Y, X), dtype=np.uint8)
    rgb = rng.integers(low=0, high=256, size=(Y, X, 3), dtype=np.uint8)

    # --- Grayscale ---
    print("== Grayscale ==")
    cd_gray = compute_cross_difference(gray)
    rt_gray = rank_transform(gray, sz=2)
    print(f"cross-diff: shape={cd_gray.shape}, dtype={cd_gray.dtype}, "
          f"min={cd_gray.min():.3f}, max={cd_gray.max():.3f}, mean={cd_gray.mean():.3f}")
    print(f"rank-trfm : shape={rt_gray.shape}, dtype={rt_gray.dtype}, "
          f"min={rt_gray.min():.3f}, max={rt_gray.max():.3f}, mean={rt_gray.mean():.3f}")

    # --- RGB ---
    print("\n== RGB ==")
    cd_rgb = compute_cross_difference(rgb)
    rt_rgb = rank_transform(rgb, sz=2)
    print(f"cross-diff: shape={cd_rgb.shape}, dtype={cd_rgb.dtype}, "
          f"min={cd_rgb.min():.3f}, max={cd_rgb.max():.3f}, mean={cd_rgb.mean():.3f}")
    print(f"rank-trfm : shape={rt_rgb.shape}, dtype={rt_rgb.dtype}, "
          f"min={rt_rgb.min():.3f}, max={rt_rgb.max():.3f}, mean={rt_rgb.mean():.3f}")

    feat_gray_8  = preprocess_for_fft_features(gray, method="cross", max_period=8)
    feat_rgb_16  = preprocess_for_fft_features(rgb,  method="rank", rank_sz=2, max_period=16)
    print(feat_gray_8.shape, feat_rgb_16.shape)  # (135,), (459,)

    # Basic assertions to catch obvious mistakes
    assert cd_gray.shape == (Y - 1, X - 1), "Unexpected shape for grayscale cross-difference."
    assert cd_rgb.shape == (Y - 1, X - 1, 3), "Unexpected shape for RGB cross-difference."
    assert rt_gray.shape == (Y, X), "Unexpected shape for grayscale rank transform."
    assert rt_rgb.shape == (Y, X, 3), "Unexpected shape for RGB rank transform."
    assert 0.0 <= rt_gray.min() and rt_gray.max() <= 1.0 + 1e-6, "Rank transform (gray) not in [0, 1]."
    assert 0.0 <= rt_rgb.min() and rt_rgb.max() <= 1.0 + 1e-6, "Rank transform (RGB) not in [0, 1]."

    print("\nSelf-test completed successfully ✅")

