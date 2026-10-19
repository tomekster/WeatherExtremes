"""PyTorch port of rankboot.py: per-point thresholds, in-base bootstrap (rank form), OLS and Theil-Sen,
batched over grid points. x: (P, n_years, 365) tensor; the first n_ref years are REF."""
import math
import torch

def pooled(arr, w):                                   # (..., ny, 365) -> (..., ny, 365, w)
    h = w // 2
    idx = (torch.arange(365, device=arr.device)[:, None] + torch.arange(-h, h + 1, device=arr.device)) % 365
    return arr[..., idx]

def _qparams(p, N):
    h = p * (N - 1); a = math.floor(h)
    return a, h - a

def thresholds(ref, w, percs):                        # (P, ny, 365) -> (nq, P, 365), numpy 'linear'
    v = pooled(ref, w).permute(0, 2, 1, 3).reshape(ref.shape[0], 365, -1)
    v = torch.sort(v, dim=-1).values; N = v.shape[-1]
    out = []
    for p in percs:
        a, f = _qparams(p, N)
        hi = v[..., min(a + 1, N - 1)]
        out.append(v[..., a] + f * (hi - v[..., a]))
    return torch.stack(out)

def in_base_rates(ref, xin, w, percs):
    """ref (P, ny, 365); xin (P, ny, 365) in-base analysis values of the same years (row j <-> ref year j).
    Returns (nq, P, ny, 365) Zhang et al. (2005) bootstrap exceedance rates."""
    P, ny, _ = ref.shape; N = ny * w
    S = torch.sort(pooled(ref, w), dim=-1).values.permute(0, 2, 1, 3).contiguous()   # (P, 365, ny, w)
    q = xin.permute(0, 2, 1)                                                          # (P, 365, nj)
    qe = q[:, :, None, :].expand(P, 365, ny, ny).contiguous()                         # (P, 365, y, j)
    le = torch.searchsorted(S, qe, right=True)                                        # values <= x_j in year y
    n_gt = w - le
    jm = torch.eye(ny, dtype=torch.bool, device=ref.device)                           # (y, j): y == j
    C0 = n_gt.masked_fill(jm, 0).sum(2)                                               # (P, 365, j)
    inf = torch.tensor(float("inf"), dtype=ref.dtype, device=ref.device)
    up = torch.gather(S, -1, le.clamp(max=w - 1))
    up = torch.where((le < w) & ~jm, up, inf).amin(2)                                 # smallest value > x, years != j
    lo = torch.gather(S, -1, (le - 1).clamp(min=0))
    lo = torch.where((le > 0) & ~jm, lo, -inf).amax(2)                                # largest value <= x, years != j
    C = C0[:, :, None, :] + n_gt                                                      # (P, 365, k, j)
    out = []
    for p in percs:
        a, f = _qparams(p, N)
        tA, tB = N - 2 - a, N - 1 - a
        mid = torch.nan_to_num(lo + f * (up - lo), nan=float("inf"))
        ex = (C <= tA) | ((C == tB) & (q >= mid)[:, :, None, :])
        ex = ex.masked_fill(jm, False)
        out.append((ex.sum(2).to(ref.dtype) / (ny - 1)).permute(0, 2, 1))           # (P, j, 365)
    return torch.stack(out)

def ols_theilsen(years, counts):                      # counts (B, n) -> ols, se, ts (B,)
    n = years.numel(); t = years - years.mean(); stt = (t * t).sum()
    b = (counts * t).sum(-1) / stt
    res = counts - counts.mean(-1, keepdim=True) - b[:, None] * t
    se = torch.sqrt((res ** 2).sum(-1) / (n - 2) / stt)
    i, j = torch.triu_indices(n, n, 1, device=counts.device)
    s = torch.sort((counts[:, j] - counts[:, i]) / (years[j] - years[i]), dim=-1).values
    m = s.shape[-1]
    ts = s[:, m // 2] if m % 2 else 0.5 * (s[:, m // 2 - 1] + s[:, m // 2])
    return b, se, ts

def seasonal_cycle(ref, n_harmonics=None):
    """ref (P, ny, 365) -> (P, 1, 365). n_harmonics=None: per-DOY mean over the reference years (Brunner & Voigt 2024);
    otherwise the least-squares fit of a constant plus n_harmonics annual harmonics to that mean, which equals the fit
    to all reference days because every DOY has the same number of years."""
    m = ref.mean(1)
    if n_harmonics is None:
        return m[:, None, :]
    d = torch.arange(365, dtype=ref.dtype, device=ref.device) * (2 * math.pi / 365)
    H = torch.stack([torch.ones_like(d)] + [f(k * d) for k in range(1, n_harmonics + 1) for f in (torch.sin, torch.cos)], 1)
    coef = torch.linalg.pinv(H) @ m.T                                                # (2K+1, P), least squares
    return (H @ coef).T[:, None, :]

def compute_point_batch(x, years, n_ref, pools, percs, n_harmonics=None, ref_start=0, bootstrap=True):
    """x (P, n_years, 365); REF is years ref_start .. ref_start + n_ref - 1 of the analysis period.
    Returns dict of (P, len(pools), len(percs)) tensors: ols, se, ts (per year), mean yearly count 'mean', and
    ols_oob, ts_oob: the same fits using only the years outside REF (no in-base years, so no in-base step).
    n_harmonics selects the seasonal cycle (see seasonal_cycle); bootstrap=False compares in-base years with the
    full-reference thresholds (conventional procedure)."""
    rs = slice(ref_start, ref_start + n_ref)
    ref = x[:, rs]; sc = seasonal_cycle(ref, n_harmonics)
    ref_a = ref - sc; an_a = x - sc
    P = x.shape[0]
    oob = torch.ones(x.shape[1], dtype=torch.bool, device=x.device); oob[rs] = False
    keys = ("ols", "se", "ts", "mean", "ols_oob", "ts_oob")
    res = {k: torch.empty(P, len(pools), len(percs), dtype=x.dtype, device=x.device) for k in keys}
    for pi, w in enumerate(pools):
        thr = thresholds(ref_a, w, percs)
        rates = in_base_rates(ref_a, an_a[:, rs], w, percs) if bootstrap else None
        for qi in range(len(percs)):
            ex = (an_a >= thr[qi][:, None, :]).to(x.dtype)
            if bootstrap:
                ex[:, rs] = rates[qi]
            counts = ex.sum(-1)
            b, se, ts = ols_theilsen(years, counts)
            res["ols"][:, pi, qi], res["se"][:, pi, qi], res["ts"][:, pi, qi] = b, se, ts
            res["mean"][:, pi, qi] = counts.mean(-1)
            if oob.sum() >= 3:
                b, _, ts = ols_theilsen(years[oob], counts[:, oob])
            else:
                b = ts = torch.full_like(b, float("nan"))
            res["ols_oob"][:, pi, qi], res["ts_oob"][:, pi, qi] = b, ts
    return res
