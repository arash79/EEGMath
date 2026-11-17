"""Entropy and Lempel–Ziv estimators.

Permutation entropy follows Bandt and Pompe (2002): Shannon entropy of
ordinal patterns of length ``order`` at lag ``delay``, in bits, optionally
divided by ``log2(order!)``. Ties are broken by a stable argsort, which
matches the position-based tie break used for continuous EEG in practice.

Sample entropy and approximate entropy follow Richman and Moorman (2000)
and Pincus (1991). The neighbourhood radius defaults to ``0.2`` times the
population standard deviation. Distance is Chebyshev. Sample entropy
excludes self-matches and uses the same number of templates at embedding
``m`` and ``m+1`` (the length-``m`` embedding drops its last row).
Approximate entropy keeps self-matches and uses a different template
count at ``m+1``. That difference can make a short periodic series
slightly negative. The negative value is the estimator, not a failed
logarithm. When no length-``m+1`` matches fall inside the radius and
some length-``m`` matches do, sample entropy is ``+inf``. That is the
logarithm of a zero ratio. A missing value is returned when the
length-``m`` count is zero.

Multiscale entropy coarse-grains by non-overlapping means (Costa, Goldberger,
and Peng, 2002). The default radius is fixed from the scale-1 standard
deviation, which is their original-series convention. Their illustration
used ``m = 2`` and ``r = 0.15`` times that standard deviation (footnote 12).
The default fraction here is ``0.2``, the same default as sample entropy.
Pass ``r_fraction=0.15`` to match the 2002 illustration. Set
``r_mode="scale"`` to recompute the radius on each coarse series; that
variant is common but is not the original fixed-radius definition.

Lempel–Ziv complexity is the Kaspar and Schuster (1987) parsing, normalised
by ``n / log_b(n)`` when requested (Zhang et al., 2009).
"""

from __future__ import annotations

import math

import numpy as np

from eegmath.errors import ParameterError


def permutation_entropy(
    x: np.ndarray,
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    x = np.asarray(x, dtype=float)
    if order < 2 or delay < 1:
        raise ParameterError("permutation entropy requires order >= 2 and delay >= 1.")
    if x.size == 0 or not np.isfinite(x).all():
        return np.nan
    span = (order - 1) * delay
    n_patterns = x.size - span
    if n_patterns < 2:
        return np.nan
    embedded = np.column_stack([x[i * delay : i * delay + n_patterns] for i in range(order)])
    patterns = np.argsort(embedded, axis=1, kind="mergesort")
    codes = np.ravel_multi_index(patterns.T, (order,) * order)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    entropy = float(-np.sum(p * np.log2(p)))
    if normalize:
        entropy /= math.log2(math.factorial(order))
    return entropy


def _embed(x: np.ndarray, order: int, delay: int = 1) -> np.ndarray:
    n_templates = x.size - (order - 1) * delay
    if n_templates <= 0:
        raise ParameterError("Series is shorter than the embedding.")
    return np.column_stack([x[i * delay : i * delay + n_templates] for i in range(order)])


def _radius_counts(templates: np.ndarray, radius: float, block: int = 128) -> np.ndarray:
    n = templates.shape[0]
    counts = np.empty(n, dtype=np.int32)
    for start in range(0, n, block):
        chunk = templates[start : start + block]
        dist = np.max(np.abs(chunk[:, None, :] - templates[None, :, :]), axis=-1)
        counts[start : start + chunk.shape[0]] = np.sum(dist <= radius, axis=1)
    return counts


def _tolerance(x: np.ndarray, tolerance: float | None) -> float:
    if tolerance is None:
        return float(0.2 * np.std(x, ddof=0))
    if tolerance < 0:
        raise ParameterError("Entropy tolerance must be non-negative.")
    return float(tolerance)


def sample_entropy(x: np.ndarray, *, order: int = 2, tolerance: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if order < 1:
        raise ParameterError("sample entropy order must be at least 1.")
    if x.size < order + 2 or not np.isfinite(x).all():
        return np.nan
    radius = _tolerance(x, tolerance)
    if radius == 0 and np.std(x) == 0:
        return 0.0
    emb_m = _embed(x, order)[:-1]
    emb_m1 = _embed(x, order + 1)
    if emb_m.shape[0] < 2 or emb_m1.shape[0] < 2:
        return np.nan
    b_counts = _radius_counts(emb_m, radius)
    a_counts = _radius_counts(emb_m1, radius)
    b = np.mean((b_counts - 1) / (emb_m.shape[0] - 1))
    a = np.mean((a_counts - 1) / (emb_m1.shape[0] - 1))
    if b <= 0 or a < 0:
        return np.nan
    # A = 0 and B > 0 is a zero ratio. The entropy is infinite, not missing.
    if a == 0:
        return np.inf
    return float(-np.log(a / b))


def approximate_entropy(x: np.ndarray, *, order: int = 2, tolerance: float | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if order < 1:
        raise ParameterError("approximate entropy order must be at least 1.")
    if x.size < order + 2 or not np.isfinite(x).all():
        return np.nan
    radius = _tolerance(x, tolerance)
    phi = []
    for m in (order, order + 1):
        emb = _embed(x, m)
        counts = _radius_counts(emb, radius).astype(float)
        # Self-matches keep the count at least 1 when radius >= 0.
        phi.append(float(np.mean(np.log(counts / emb.shape[0]))))
    return phi[0] - phi[1]


def svd_entropy(x: np.ndarray, *, order: int = 3, delay: int = 1, normalize: bool = True) -> float:
    """Shannon entropy of the normalised singular values of a delay embedding.

    ``p_i = σ_i / Σσ`` and ``H = -Σ p_i log2(p_i)``. Bao, Liu, and
    Zhang (2011, eq. 9) define that entropy and attribute it to Roberts,
    Penny, and Rezek (1999). Dividing by ``log2(order)`` is a separate
    map onto ``[0, 1]``. There are at most ``order`` singular values,
    so that divisor is not ``log2(order!)``. Squared singular values
    are the eigenvalues of the Gram matrix and are not used.
    """
    x = np.asarray(x, dtype=float)
    if order < 2 or delay < 1:
        raise ParameterError("SVD entropy requires order >= 2 and delay >= 1.")
    if x.size <= (order - 1) * delay or not np.isfinite(x).all():
        return np.nan
    mat = _embed(x, order, delay)
    singular = np.linalg.svd(mat, compute_uv=False)
    total = float(np.sum(singular))
    if total <= 0:
        return np.nan
    p = singular / total
    p = p[p > 0]
    entropy = float(-np.sum(p * np.log2(p)))
    if normalize:
        entropy /= np.log2(order)
    return entropy


def multiscale_sample_entropy(
    x: np.ndarray,
    scales: tuple[int, ...] = (1, 5, 10),
    *,
    order: int = 2,
    r_mode: str = "original",
    r_fraction: float = 0.2,
) -> dict[int, float]:
    """Sample entropy of coarse-grained series.

    ``r_mode="original"`` fixes the radius from the scale-1 series.
    Costa, Goldberger, and Peng (2002, footnote 12) illustrated
    ``r_fraction=0.15``. The default is 0.2, matching sample entropy.
    ``r_mode="scale"`` recomputes the radius after coarse-graining
    and is not that fixed-radius definition.
    """
    x = np.asarray(x, dtype=float)
    if r_mode not in {"original", "scale"}:
        raise ParameterError("r_mode must be 'original' or 'scale'.")
    if r_fraction <= 0:
        raise ParameterError("r_fraction must be positive.")
    fixed = None
    if r_mode == "original":
        if x.size == 0 or not np.isfinite(x).all():
            return {int(s): np.nan for s in scales}
        fixed = float(r_fraction * np.std(x, ddof=0))
    out: dict[int, float] = {}
    for scale in scales:
        scale = int(scale)
        if scale < 1:
            raise ParameterError("MSE scales must be positive integers.")
        n = x.size - (x.size % scale)
        if n < scale or not np.isfinite(x).all():
            out[scale] = np.nan
            continue
        coarse = x[:n].reshape(-1, scale).mean(axis=1)
        if coarse.size < order + 5:
            out[scale] = np.nan
            continue
        tolerance = fixed if r_mode == "original" else float(r_fraction * np.std(coarse, ddof=0))
        out[scale] = sample_entropy(coarse, order=order, tolerance=tolerance)
    return out


def _lz_complexity(sequence: np.ndarray) -> int:
    """Kaspar–Schuster parsing used by the common EEG implementation of LZ76."""
    complexity = 1
    prefix_len = 1
    length = 1
    max_length = 1
    pointer = 0
    n = int(sequence.size)
    while prefix_len + length <= n:
        if sequence[pointer + length - 1] == sequence[prefix_len + length - 1]:
            length += 1
        else:
            max_length = max(length, max_length)
            pointer += 1
            if pointer == prefix_len:
                complexity += 1
                prefix_len += max_length
                pointer = 0
                max_length = 1
            length = 1
    if length != 1:
        complexity += 1
    return int(complexity)


def _symbol_codes(symbols: np.ndarray) -> np.ndarray:
    """Map discrete symbols to compact codes. Equality is preserved."""
    _unique, codes = np.unique(symbols, return_inverse=True)
    return np.asarray(codes)


def lempel_ziv_complexity(sequence: np.ndarray | str, *, normalize: bool = True) -> float:
    """Lempel–Ziv complexity of a discrete sequence.

    Continuous samples are not symbols. Binarise or otherwise discretise
    a voltage trace before calling this function. Integer-valued floats
    are accepted. The ``lzc`` feature does that median split itself.
    """
    if isinstance(sequence, str):
        if sequence == "":
            return np.nan
        symbols = np.fromiter((ord(ch) for ch in sequence), dtype=np.uint32, count=len(sequence))
    else:
        symbols = np.asarray(sequence)
        if symbols.ndim != 1:
            raise ParameterError("LZC sequence must be one-dimensional.")
        if symbols.size == 0:
            return np.nan
        if symbols.dtype.kind == "f":
            if not np.isfinite(symbols).all():
                return np.nan
            if not np.all(symbols == np.floor(symbols)):
                raise ParameterError(
                    "Lempel-Ziv complexity needs discrete symbols. "
                    "Binarise a continuous signal before calling lempel_ziv_complexity."
                )
            symbols = symbols.astype(np.int64)
        elif symbols.dtype.kind not in "biu":
            raise ParameterError("LZC sequence must be numeric or a string.")
        else:
            symbols = symbols.astype(np.int64, copy=False)
        symbols = _symbol_codes(symbols)
    raw = _lz_complexity(symbols)
    if not normalize:
        return float(raw)
    n = int(symbols.size)
    if n < 2:
        return np.nan
    n_unique = int(np.unique(symbols).size)
    base = 2 if n_unique < 2 else n_unique
    return float(raw / (n / math.log(n, base)))
