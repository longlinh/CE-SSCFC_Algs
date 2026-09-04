"""Synthetic multi-datasite scenes for the demo and the tests (no data files needed).

Each datasite is an H x W image of d-band pixels.  Class regions come from a Voronoi
tessellation of random seed points (contiguous patches), every class has a spectral
mean shared by all datasites (same physics; the per-datasite standardisation and the
different class proportions produce a mild domain shift), and pixels are mean +
Gaussian noise.  Labels are class-pure circular regions,
as in the paper's protocol; the test set is a spatial block hold-out.
"""

import numpy as np

from cesscfc import NO_LABEL


def standardize(X):
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (X - X.mean(axis=0, keepdims=True)) / std


def voronoi_labels(height, width, n_classes, seeds_per_class, rng):
    pts = rng.integers(0, [height, width], size=(n_classes * seeds_per_class, 2))
    cls = np.arange(len(pts)) % n_classes
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    grid = np.stack([yy.ravel(), xx.ravel()], axis=1)
    d = (grid[:, None, 0] - pts[None, :, 0]) ** 2 + (grid[:, None, 1] - pts[None, :, 1]) ** 2
    return cls[d.argmin(axis=1)]


def circular_labels(y_true, shape, n_regions, radius, rng, no_label=NO_LABEL):
    """Class-pure discs: `n_regions` centres per class, pixels of that class within
    `radius` are revealed.  Returns the partial label vector (-1 = unlabelled)."""
    H, W = shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    y_lab = np.full(H * W, no_label, dtype=int)
    for k in np.unique(y_true[y_true >= 0]):
        idx = np.where(y_true == k)[0]
        for c in rng.choice(idx, size=min(n_regions, len(idx)), replace=False):
            cy, cx = divmod(int(c), W)
            disc = ((yy - cy) ** 2 + (xx - cx) ** 2).ravel() <= radius ** 2
            y_lab[disc & (y_true == k)] = k
    return y_lab


def block_holdout(shape, y_lab, rng, block=16, test_ratio=0.3, valid=None):
    """Spatially disjoint test mask: ~test_ratio of block x block tiles, labelled
    pixels (and invalid pixels) excluded."""
    H, W = shape
    by, bx = -(-H // block), -(-W // block)
    n_blocks = by * bx
    test_blocks = np.zeros(n_blocks, bool)
    test_blocks[rng.choice(n_blocks, size=int(round(test_ratio * n_blocks)), replace=False)] = True
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    bid = (yy // block) * bx + (xx // block)
    mask = test_blocks[bid.ravel()] & (y_lab == NO_LABEL)
    if valid is not None:
        mask &= valid
    return mask


def make_site(height, width, means, noise, n_regions, radius, seed):
    """One datasite; returns dict(X standardised, y_true, y_lab, test_mask, shape)."""
    rng = np.random.default_rng(seed)
    C, d = means.shape
    y_true = voronoi_labels(height, width, C, 4, rng)
    X = means[y_true] + rng.normal(0.0, noise, size=(height * width, d))
    y_lab = circular_labels(y_true, (height, width), n_regions, radius, rng)
    test = block_holdout((height, width), y_lab, rng)
    return dict(X=standardize(X), y_true=y_true, y_lab=y_lab, test_mask=test,
                shape=(height, width))


def make_collab_scene(seed=42, height=96, width=96, d=4, n_classes=5, noise=0.3,
                      poor=(2, 3), rich=(4, 6)):
    """Three datasites sharing the class spectra: datasite 0 is label-poor
    (`poor` = (regions per class, radius)), datasites 1-2 are label-rich."""
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0, 1.0, size=(n_classes, d))
    means[1] = means[0] + rng.normal(0.0, 0.35, size=d)     # two spectrally close classes
    sites = []
    for s, (n_reg, rad) in enumerate([poor, rich, rich]):
        sites.append(make_site(height, width, means, noise, n_reg, rad, seed + 100 * s))
    return sites, n_classes
