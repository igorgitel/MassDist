"""
BF_analysis.py  --  the measurement layer for Majorant_v2

The simulation engine, BF_v_no_resampling_v2.py, produces arrays.  Turning those
arrays into a number is a separate job with its own failure modes, and this module
is that job in one place: loading a saved run, choosing the estimator, attaching an
uncertainty to it, finding the range over which a power law actually holds, and
propagating the counting error all the way through to the growth exponent.

NOTHING HERE TOUCHES THE ENGINE.  Every uncertainty below is reconstructed from what
`simulate` already stores, so no run has to be repeated and `simulate` itself needs
no change.  That is possible because `dndm` and `iso_counts` are both linear
rescalings of raw particle counts, and the scale factors are recorded in `meta`:

    dndm[k,b]      = n[k,b] * (w/V) / widths[b]        ->  n = dndm * widths * V / w
    iso_counts[a,b] = (raw counts) * w                 ->  n = iso_counts / w

----------------------------------------------------------------------
TWO ERRORS, AND WHY BOTH ARE REPORTED
----------------------------------------------------------------------
The Poisson error, sigma/F = 1/sqrt(C) with C the total counts, is a FLOOR.  It is
what you would get if every snapshot were an independent draw and the only noise
were counting noise.

The empirical error, the scatter of dndm across snapshots divided by sqrt(K_eff), is
what you actually have.  It needs no distributional assumption and it automatically
contains everything Poisson misses: that consecutive snapshots share particles, that
the population itself fluctuates, that the steady state is only statistically steady.

K_eff is the number of INDEPENDENT snapshots, estimated from the lag-1
autocorrelation as K (1-rho)/(1+rho).  Using K instead of K_eff is the standard way
to under-report an error bar by a factor of a few.

The two are returned side by side.  When the empirical error sits at the Poisson
floor the sampling is as good as counting allows; when it sits far above, the run is
telling you that the snapshots are correlated or the state is drifting.

----------------------------------------------------------------------
FROM THE SPECTRUM TO b
----------------------------------------------------------------------
The chain is short and every link is exact:

  1. counts in an (age, mass) cell        c_ab            Poisson, sigma = sqrt(c)
  2. mean mass of an isochrone            <m>_a = S1/S0   S1 = sum_b c_ab m_b
                                                          S0 = sum_b c_ab
     d<m>/dc_b = (m_b - <m>)/S0, so
        var(<m>) = sum_b c_b (m_b - <m>)^2 / S0^2
     which is exactly the standard error of the mean, sigma_m / sqrt(S0), with
     sigma_m the spread of masses inside that isochrone.
  3. a particle is counted once per snapshot it survives in its age bin, so S0
     counts APPEARANCES, not independent particles.  The correction is
        N_eff = S0 / n_rep,     n_rep = age-bin width / snapshot spacing,
     clipped at 1.  Skipping it is the second standard way to under-report.
  4. b from a WEIGHTED least squares of ln<m> against ln tau, which returns
     sigma_b analytically together with chi2/dof.

chi2/dof is the useful part.  It is near one when a single power law describes the
window and blows up when the window has swallowed the crossover at m_inj or the
truncation at m_sink -- which makes it an objective test of the fitting range
instead of an argument about it.
"""

import numpy as np

import BF_v_no_resampling_v2 as BF

# Re-exported from the engine so a notebook imports ONE module.  These are the
# analysis helpers that live there for historical reasons; they are unchanged.
local_slope         = BF.local_slope
find_inertial_range = BF.find_inertial_range
guard_band          = BF.guard_band
fit_powerlaw        = BF.fit_powerlaw
superpose           = BF.superpose
predict             = BF.predict
iso_mean_mass_raw   = BF.iso_mean_mass      # counts-only version, no error bar
growth_fit          = BF.growth_fit


# ======================================================================
#  LOADING
# ======================================================================

def load(path):
    """Read a run written by add_last_run.  Thin wrapper, kept so notebooks import
    one module rather than two."""
    return BF.load_run(path)


def describe(run, name=""):
    """One block of text: configuration, cost, what was stored alongside."""
    m = run["meta"]
    a = m.get("analysis", {})
    out = ["=" * 78, "%s  <-  %s" % (name or m.get("saved_as", "?"), m.get("saved_as", "?")),
           "  %-16s %s / %s, kernel %s (lambda = %s)"
           % ("configuration", m["process"], m["system"], m["kernel"], m["lambda"]),
           "  %-16s m_i = %.4g, N = %g" % ("initial", m["ic"]["m"], m["ic"]["N"]),
           "  %-16s rate = %.4g at m = %.4g" % ("injection", m["injection_rate"],
                                                m["injection_mass"]),
           "  %-16s %s" % ("sink", m["sink_mass"]),
           "  %-16s %s" % ("stopped by", m.get("stop_reason")),
           "  %-16s events = %.4g, acc = %.3f, mass drift = %+.1e"
           % ("cost", float(run["events"][-1]), float(run["acceptance"]),
              float(run["mass_drift"])),
           "  %-16s %d snapshots, %d mass bins"
           % ("arrays", np.asarray(run["dndm"]).shape[0], np.asarray(run["centers"]).size)]
    if a:
        out.append("  stored analysis:")
        for k in sorted(a):
            v = a[k]
            out.append("      %-22s %s" % (k, ("%+.4g" % v) if isinstance(v, float) else v))
    return "\n".join(out)


# ======================================================================
#  RAW COUNTS  --  the thing every error bar is built from
# ======================================================================

def scale_factors(run):
    """(w, V, widths): everything needed to go between dndm and raw counts."""
    m = run["meta"]
    w = float(m.get("weight", 1.0))
    V = float(m.get("volume", 1.0))
    return w, V, np.asarray(run["widths"], float)


def raw_counts(run):
    """
    Particle counts per (snapshot, mass bin), recovered from dndm.

    dndm = n * (w/V) / width, so n = dndm * width * V / w.  Exact, not an estimate:
    the engine stores the rescaled histogram and the scale is in meta.
    """
    w, V, widths = scale_factors(run)
    return np.asarray(run["dndm"], float) * widths[None, :] * V / w


def steady_mask(run, frac=0.5, verbose=False):
    """
    Which snapshots belong to the steady state.

    The sink gate is the criterion: it fires once the cascade has demonstrably
    delivered particles across the whole inertial range, which is exactly the point
    after which the run is stationary.  Averaging across the transient before it
    would smear the spectrum rather than sharpen it.  Without a gate, fall back on
    the last `frac` of the run and say so.
    """
    t = np.asarray(run["t"], float)
    t0 = float(run["iso_t_begin"]) if "iso_t_begin" in run else np.nan
    if np.isfinite(t0):
        ss = t >= t0
    else:
        ss = np.zeros(t.size, bool)
        ss[int((1.0 - frac) * t.size):] = True
        if verbose:
            print("   no sink gate recorded -- averaging the last %.0f%% of snapshots"
                  % (100 * frac))
    if ss.sum() < 1:
        ss[-1] = True
    return ss


def effective_snapshots(series):
    """
    Number of INDEPENDENT samples in a correlated series, K (1-rho)/(1+rho).

    Consecutive snapshots share particles, so the naive K over-states the averaging.
    rho is the lag-1 autocorrelation; rho = 0 gives K back, rho -> 1 gives 1.
    """
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    K = x.size
    if K < 4:
        return float(max(K, 1))
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return float(K)
    rho = float((x[:-1] * x[1:]).sum() / denom)
    rho = min(max(rho, 0.0), 0.98)
    return float(np.clip(K * (1.0 - rho) / (1.0 + rho), 1.0, K))


# ======================================================================
#  THE SPECTRUM, WITH AN ERROR BAR
# ======================================================================

def steady_spectrum(run, frac=0.5, verbose=False):
    """
    The estimator for an OPEN run and its uncertainty.

    The steady state IS the instantaneous spectrum, so dt never enters and every
    post-gate snapshot is a draw from the same distribution.  Averaging them is free
    -- they are already in the file -- and this returns the average together with
    both error estimates.

    Returns a dict:
        F          averaged dN/dm
        sigma      the error bar to plot: empirical, floored at Poisson
        sigma_emp  scatter across snapshots / sqrt(K_eff)
        sigma_pois F / sqrt(total counts)
        counts     total raw counts per bin over the averaged snapshots
        K, K_eff   snapshots averaged, and how many of them were independent
    """
    D = np.asarray(run["dndm"], float)
    n = raw_counts(run)
    ss = steady_mask(run, frac=frac, verbose=verbose)
    Ds, ns = D[ss], n[ss]
    K = int(ss.sum())

    F = Ds.mean(axis=0)
    C = ns.sum(axis=0)                                   # total counts per bin

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pois = np.where(C > 0, F / np.sqrt(np.maximum(C, 1e-300)), np.nan)

    if K >= 4:
        # one K_eff per bin: correlation is not the same at the top and the bottom
        Keff = np.array([effective_snapshots(Ds[:, b]) for b in range(Ds.shape[1])])
        sigma_emp = Ds.std(axis=0, ddof=1) / np.sqrt(np.maximum(Keff, 1.0))
    else:
        Keff = np.full(Ds.shape[1], float(K))
        sigma_emp = np.full(F.shape, np.nan)

    sigma = np.where(np.isfinite(sigma_emp),
                     np.maximum(sigma_emp, np.where(np.isfinite(sigma_pois), sigma_pois, 0.0)),
                     sigma_pois)
    sigma = np.where(F > 0, sigma, np.nan)

    return dict(F=F, sigma=sigma, sigma_emp=sigma_emp, sigma_pois=sigma_pois,
                counts=C, K=K, K_eff=Keff, mask=ss)


def superposed_spectrum(run):
    """The CLOSED-system estimator: the dt-weighted superposition.  No steady state
    exists there, so the observable is the age integral and the snapshot cadence,
    not the averaging, is what has to be right (see note [10] in the engine)."""
    F = BF.superpose(run["dndm"], run["t"])
    n = raw_counts(run)
    dt = np.diff(np.asarray(run["t"], float), prepend=float(run["t"][0]))
    # the weighted sum of Poisson counts: var = sum (dt_k)^2 * n_k, in dndm units
    w, V, widths = scale_factors(run)
    scale = (w / V) / widths
    var = ((dt[:, None] ** 2) * n).sum(axis=0) * scale ** 2
    with np.errstate(invalid="ignore"):
        sigma = np.where(F > 0, np.sqrt(var), np.nan)
    return dict(F=F, sigma=sigma, sigma_pois=sigma, sigma_emp=np.full(F.shape, np.nan),
                counts=n.sum(axis=0), K=int(n.shape[0]),
                K_eff=np.full(F.shape, float(n.shape[0])), mask=np.ones(n.shape[0], bool))


def spectrum(run, frac=0.5, verbose=False):
    """Pick the estimator from the system recorded in meta.  Open -> the averaged
    instantaneous spectrum; closed -> the superposition."""
    if run["meta"]["system"] == "open":
        return steady_spectrum(run, frac=frac, verbose=verbose)
    return superposed_spectrum(run)


# ======================================================================
#  ISOCHRONES, WITH AN ERROR BAR
# ======================================================================

def iso_mean_mass(run, correct_repeats=True, groups=None):
    """
    <m>(tau) from the accumulated (age, mass) histogram, with its uncertainty.

    var(<m>) = sum_b c_b (m_b - <m>)^2 / S0^2 -- the standard error of the mean, with
    the spread taken over the mass distribution inside that isochrone.

    `correct_repeats` divides the counts by how many snapshots a particle spends
    inside one age bin, because the histogram counts appearances rather than
    independent particles.  n_rep = (age bin width) / (snapshot spacing), clipped at
    one.  Without it the error bar is too small by sqrt(n_rep), which here is a
    factor of a few.

    `groups` merges age bins before anything is computed: a list of index arrays,
    one per merged isochrone.  The counts of an age bin are accumulated additively
    by the engine, so summing adjacent bins is not an approximation -- it is exactly
    the histogram that a coarser `iso_age_edges` would have produced.  That identity
    is what makes the rebinned growth law a check on the fine-binned one rather than
    a different measurement: the same particles, resampled in age.

    Returns dict with tau, mbar, sigma, counts, n_rep (and tau_lo, tau_hi).
    """
    w, V, widths = scale_factors(run)
    c = np.asarray(run["iso_counts"], float) / w          # raw counts per (age, mass)
    m = np.asarray(run["centers"], float)
    e = np.asarray(run["iso_age_edges"], float)

    if groups is None:
        lo, hi = e[:-1], e[1:]
    else:
        groups = [np.atleast_1d(np.asarray(g, int)) for g in groups]
        if len(groups) == 0:
            nan = np.zeros(0)
            return dict(tau=nan, mbar=nan, sigma=nan, counts=nan, n_rep=nan,
                        tau_lo=nan, tau_hi=nan)
        c = np.array([c[g].sum(axis=0) for g in groups])   # counts add.  Exactly.
        lo = np.array([e[g.min()] for g in groups])
        hi = np.array([e[g.max() + 1] for g in groups])
    tau = np.sqrt(lo * hi)

    S0 = c.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mbar = np.where(S0 > 0, (c * m[None, :]).sum(axis=1) / np.maximum(S0, 1e-300), np.nan)
        spread2 = np.where(S0 > 0,
                           (c * (m[None, :] - mbar[:, None]) ** 2).sum(axis=1), np.nan)

    n_rep = np.ones_like(tau)
    if correct_repeats:
        ss = steady_mask(run)
        t = np.asarray(run["t"], float)[ss]
        if t.size > 1:
            dt_snap = float(np.median(np.diff(t)))
            if dt_snap > 0:
                n_rep = np.clip((hi - lo) / dt_snap, 1.0, None)

    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = np.where(S0 > 0, np.sqrt(spread2 * n_rep) / np.maximum(S0, 1e-300), np.nan)

    return dict(tau=tau, mbar=mbar, sigma=sigma, counts=S0, n_rep=n_rep,
                tau_lo=lo, tau_hi=hi)


# ======================================================================
#  FITTING
# ======================================================================

def wls_powerlaw(x, y, sy=None, mask=None):
    """
    Weighted least squares of ln y against ln x -- i.e. y = A x^p with errors.

    Returns p, its analytic sigma, A, chi2/dof and the number of points.  chi2/dof is
    the part worth reading: near one the window is described by a single power law,
    far above one it is not, whatever the correlation coefficient says.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask is not None:
        k &= mask
    if sy is not None:
        sy = np.asarray(sy, float)
        k &= np.isfinite(sy) & (sy > 0)
    if k.sum() < 3:
        return dict(p=np.nan, sigma_p=np.nan, A=np.nan, chi2_dof=np.nan,
                    rms_resid=np.nan, n=int(k.sum()))

    X, Y = np.log(x[k]), np.log(y[k])
    W = np.ones_like(X) if sy is None else 1.0 / (sy[k] / y[k]) ** 2

    Sw = W.sum(); Sx = (W * X).sum(); Sy = (W * Y).sum()
    Sxx = (W * X * X).sum(); Sxy = (W * X * Y).sum()
    det = Sw * Sxx - Sx * Sx
    if det == 0:
        return dict(p=np.nan, sigma_p=np.nan, A=np.nan, chi2_dof=np.nan,
                    rms_resid=np.nan, n=int(k.sum()))

    p = (Sw * Sxy - Sx * Sy) / det
    a = (Sxx * Sy - Sx * Sxy) / det
    sigma_p = np.sqrt(Sw / det)
    resid = Y - (a + p * X)
    dof = max(k.sum() - 2, 1)
    chi2_dof = float((W * resid ** 2).sum() / dof)
    rms = float(np.sqrt((resid ** 2).mean()))
    if sy is None:
        # Unweighted: there is no chi2, only scatter.  Scale sigma by it, and report
        # the number as nan so nobody reads a goodness-of-fit into it.
        sigma_p *= np.sqrt(chi2_dof)
        chi2_dof = np.nan
    return dict(p=float(p), sigma_p=float(sigma_p), A=float(np.exp(a)),
                chi2_dof=chi2_dof, rms_resid=rms, n=int(k.sum()))


def anchor_amplitude(m, F, m_lo, m_hi, alpha):
    """A such that A m^alpha passes through the data over [m_lo, m_hi]: the
    least-squares intercept at FIXED slope, so the drawn line carries the measured
    index and nothing else, and the theory line drawn the same way differs from it
    in slope alone."""
    m = np.asarray(m, float); F = np.asarray(F, float)
    if not (np.isfinite(m_lo) and np.isfinite(m_hi) and np.isfinite(alpha)):
        return np.nan
    k = np.isfinite(F) & (F > 0) & (m >= m_lo) & (m <= m_hi)
    if k.sum() < 3:
        return np.nan
    return 10.0 ** np.mean(np.log10(F[k]) - alpha * np.log10(m[k]))


# ======================================================================
#  PLATEAUX
# ======================================================================

def spectrum_plateau(run, half=3, tol=0.30, min_decades=0.8, spec=None):
    """The inertial range of the spectrum: the longest flat run of
    d log F / d log m.  Thin wrapper on the engine's finder, so the notebook has one
    import."""
    s = spec if spec is not None else spectrum(run)
    return BF.find_inertial_range(run["centers"], s["F"], half=half, tol=tol,
                                  min_decades=min_decades)


def growth_plateau(tau, mbar, counts=None, half=3, tol=0.50, min_bins=5,
                   min_slope=0.30, n_min=1e3):
    """
    The self-similar stretch of <m>(tau): the longest flat run of
    d log<m> / d log tau whose mean is an actual growth rate.

    BF.find_inertial_range cannot be used unmodified here.  It returns the LONGEST
    flat run, and the longest flat run in <m>(tau) is the dead shelf at <m> = m_inj
    where the isochrones have not started moving -- sixty bins over three decades of
    tau at slope zero, against a dozen bins over half a decade for the real thing.
    It is the same trap as the frozen shelf below min_frag_mass in closed
    fragmentation.  The single extra condition, |mean slope| >= min_slope, states that
    a shelf is not a growth law, and rejects it without touching anything real.

    The absolute value is what lets the same finder serve fragmentation, where the
    isochrones run DOWN in mass and the exponent is negative.  What is being rejected
    is a rate of zero, not a sign; requiring the mean slope itself to be positive
    would have returned nothing at all on every fragmentation run and quietly fallen
    back on drawing every age bin.
    """
    tau = np.asarray(tau, float); mbar = np.asarray(mbar, float)
    ok = np.isfinite(mbar) & (mbar > 0)
    if counts is not None:
        ok &= np.asarray(counts, float) > n_min
    x, y = tau[ok], mbar[ok]
    if x.size < min_bins + 2:
        return dict(b=np.nan, scatter=np.nan, decades=0.0, n_bins=0, tau_lo=np.nan,
                    tau_hi=np.nan, m_lo=np.nan, m_hi=np.nan)

    _, G = BF.local_slope(x, y, half=half)
    good = np.isfinite(G)
    best, i, n = None, 0, x.size
    while i < n:
        if not good[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and good[j + 1]:
            seg = G[i:j + 2]
            if seg.max() - seg.min() > 2 * tol:
                break
            j += 1
        seg = G[i:j + 1]
        if seg.size >= min_bins and abs(seg.mean()) >= min_slope:
            dec = np.log10(x[j] / x[i])
            if best is None or dec > best[0]:
                best = (dec, i, j)
        i = j + 1

    if best is None:
        return dict(b=np.nan, scatter=np.nan, decades=0.0, n_bins=0, tau_lo=np.nan,
                    tau_hi=np.nan, m_lo=np.nan, m_hi=np.nan)
    dec, i, j = best
    seg = G[i:j + 1]
    return dict(b=float(seg.mean()), scatter=float(seg.std()), decades=float(dec),
                n_bins=int(j - i + 1), tau_lo=float(x[i]), tau_hi=float(x[j]),
                m_lo=float(y[i]), m_hi=float(y[j]))


def growth_law(run, correct_repeats=True, **kw):
    """
    The whole chain in one call: counts -> <m>(tau) with sigma -> plateau ->
    weighted fit -> b with its uncertainty.

    Returns the isochrone dict, the plateau dict and the fit dict, plus the mask of
    age bins that entered the fit.
    """
    iso = iso_mean_mass(run, correct_repeats=correct_repeats)
    pl = growth_plateau(iso["tau"], iso["mbar"], iso["counts"], **kw)
    k = (np.isfinite(iso["mbar"]) & (iso["tau"] >= pl["tau_lo"])
         & (iso["tau"] <= pl["tau_hi"]))
    fit = wls_powerlaw(iso["tau"], iso["mbar"], iso["sigma"], mask=k)
    return iso, pl, fit, k


# ======================================================================
#  DRAWING
# ======================================================================

def iso_groups(run, pl=None, iso=None, n_curves=8, min_counts=1e3,
               inside_plateau=True, correct_repeats=True):
    """
    Merge the age bins into a handful of wide ones and return them as index groups.

    Two things go wrong when every age bin is drawn as its own isochrone.  Below one
    collision time nothing has happened yet, so dozens of bins pile up as identical
    curves at the injection mass; above the plateau the bins are so thin that the
    curve is shot noise.  Both are cured by choosing the window from the physics and
    then widening the bins until each one has something in it.

    The window is the growth-law plateau -- the stretch of ages over which
    d log<m>/d log tau is flat.  Those are the isochrones that are actually
    self-similar, so the picture then shows exactly the age bins the exponent was
    measured on, and nothing else.  Set inside_plateau=False to keep every bin that
    holds counts.

    The widening is greedy from the young end: a group keeps taking bins until it is
    at least (span / n_curves) wide in log tau AND holds at least min_counts.  Where
    counts are plentiful that stops after one or two bins; near the top of the
    cascade, where they are not, the groups widen on their own.  Merging is exact --
    see iso_mean_mass -- so nothing is smoothed away that was ever measured; only the
    age resolution is spent, and it is spent where it was worthless anyway.

    Returns (groups, info) with info.rejected listing the bins left out, so a caller
    can draw them faintly instead of hiding them.
    """
    empty = dict(rejected=np.array([], int), candidates=np.array([], int),
                 pl=pl, span=0.0, mode="none")
    if "iso_counts" not in run:
        return [], empty
    e = np.asarray(run["iso_age_edges"], float)
    tau = np.sqrt(e[:-1] * e[1:])
    cnt = np.asarray(run["iso_counts"], float).sum(axis=1)
    have = np.flatnonzero(cnt > 0)
    if have.size == 0:
        return [], empty

    mode = "all bins"
    cand = have
    if inside_plateau:
        if pl is None:
            if iso is None:
                iso = iso_mean_mass(run, correct_repeats=correct_repeats)
            pl = growth_plateau(iso["tau"], iso["mbar"], iso["counts"])
        if np.isfinite(pl.get("tau_lo", np.nan)):
            sel = have[(tau[have] >= pl["tau_lo"]) & (tau[have] <= pl["tau_hi"])]
            if sel.size >= 2:
                cand, mode = sel, "growth plateau"
    # A plateau that could not be found is not a reason to draw nothing.
    if cand.size < 2:
        cand, mode = have, "all bins (no plateau found)"

    # widths are measured on the EDGES, so "how wide is this group" is the width of
    # the age band it actually covers, not the gap between the first and last centre
    span = float(np.log10(e[cand[-1] + 1] / e[cand[0]]))
    target = span / max(int(n_curves), 1) if span > 0 else np.inf

    groups, i = [], 0
    while i < cand.size:
        j = i
        while j + 1 < cand.size:
            wide = np.log10(e[cand[j] + 1] / e[cand[i]]) >= target
            full = cnt[cand[i:j + 1]].sum() >= min_counts
            if wide and full:
                break
            j += 1
        groups.append(cand[i:j + 1])
        i = j + 1
    # a starved last group is folded back rather than drawn as noise
    if len(groups) > 1 and cnt[groups[-1]].sum() < min_counts:
        groups[-2] = np.concatenate([groups[-2], groups[-1]])
        groups.pop()

    used = np.concatenate(groups) if groups else np.array([], int)
    info = dict(rejected=np.setdiff1d(have, used), candidates=cand, pl=pl,
                span=span, mode=mode)
    return groups, info


def plateau_line(pl, fit=None, b_theory=None, tag=""):
    """One line stating the window the picture and the fit share, ready to print."""
    if pl is None or not np.isfinite(pl.get("b", np.nan)):
        return "%s  no growth plateau found" % tag
    s = ("%s  plateau: <m> from %.4g to %.4g, tau from %.4g to %.4g, "
         "%d age bins over %.2f dec, b = %.3f +- %.3f"
         % (tag, pl["m_lo"], pl["m_hi"], pl["tau_lo"], pl["tau_hi"],
            pl["n_bins"], pl["decades"], pl["b"], pl["scatter"]))
    if fit is not None and np.isfinite(fit.get("p", np.nan)):
        s += " | weighted fit %.4f +- %.4f" % (fit["p"], fit["sigma_p"])
    if b_theory:
        s += " | theory %.3f" % b_theory
    return s


def pick_isochrones(run, n_show=None, mode="plateau", min_frac=1e-4,
                    min_counts=1e3, pl=None, iso=None):
    """
    Which isochrones to draw.  Returns (groups, tau, n_candidates), where each group
    is an array of age-bin indices to be merged into one curve; a group of length one
    is an ordinary isochrone, so the old (index array) return still works downstream.

    mode='plateau' -- the default -- takes the window from the growth law and widens
    the bins, which is the only selection here that is not a cosmetic threshold.
    mode='counts' is the old behaviour with its one bug fixed: it now ranks age bins
    by how many PARTICLES they hold, not by the sum of dN/dm, which is a sum of
    densities dominated by the narrow bins at the injection scale and falls twelve
    orders of magnitude across the age grid where the counts fall four.  That
    mismatch is what silently discarded the top two decades of the cascade.
    """
    if "iso_dndm" not in run:
        return [], np.array([]), 0
    e = np.asarray(run["iso_age_edges"], float)
    tau = np.sqrt(e[:-1] * e[1:])
    cnt = np.asarray(run["iso_counts"], float).sum(axis=1)
    if not np.any(cnt > 0):
        return [], tau, 0

    if mode == "plateau":
        groups, info = iso_groups(run, pl=pl, iso=iso,
                                  n_curves=(8 if not n_show else int(n_show)),
                                  min_counts=min_counts)
        return groups, tau, int(info["candidates"].size)

    usable = np.flatnonzero(cnt > min_frac * cnt.max())
    if usable.size == 0:
        return [], tau, 0
    if n_show and n_show < usable.size:
        usable = usable[np.unique(np.linspace(0, usable.size - 1,
                                              int(n_show)).astype(int))]
    return [np.array([k]) for k in usable], tau, int(usable.size)


def draw_isochrones(ax, run, idx, tau, tau_scale=1.0, cbar_label=r"age $\tau$",
                    min_count=3.0, show_rejected=True, rejected=None):
    """
    Isochrones coloured by age.  `idx` may be plain bin indices or groups of them; a
    group is summed, which is what a wider age bin means.  Mass bins holding fewer
    than min_count particles are blanked, because at the ends of an isochrone the
    histogram is one or two counts wide and draws a spiky tail that is shot noise.

    The age bins that were NOT selected are drawn faintly rather than dropped.  That
    matters: the selection is made by the growth-law plateau, and a picture that
    showed only what the plateau kept would be confirming the window with the window.
    Drawn in grey underneath, the choice can be argued with.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    if len(idx) == 0:
        return
    groups = [np.atleast_1d(np.asarray(g, int)) for g in idx]
    c = np.asarray(run["centers"], float)
    cnt = np.asarray(run["iso_counts"], float)
    dn = np.asarray(run["iso_dndm"], float)
    nsnp = max(float(run["iso_snapshots"]), 1.0)

    if show_rejected:
        if rejected is None:
            used = np.concatenate(groups)
            alive = np.flatnonzero(cnt.sum(axis=1) > 0)
            rejected = np.setdiff1d(alive, used)
        for k in np.atleast_1d(rejected):
            y = dn[k] / nsnp
            ok = (y > 0) & (cnt[k] >= min_count)
            if ok.sum() < 2:
                continue      # a single populated mass bin draws no line, only clutter
            ax.loglog(c, np.where(ok, y * c ** 2, np.nan), lw=.7, alpha=.55,
                      color="0.78", zorder=0)

    tg = np.array([np.average(tau[g], weights=np.maximum(cnt[g].sum(axis=1), 1e-300))
                   for g in groups])
    lo, hi = tg.min() * tau_scale, tg.max() * tau_scale
    norm = LogNorm(vmin=lo, vmax=max(hi, lo * 1.0001))
    cm = plt.cm.viridis
    for g, tk in zip(groups, tg):
        cg = cnt[g].sum(axis=0)
        y = dn[g].sum(axis=0) / nsnp
        ok = (y > 0) & (cg >= min_count)
        ax.loglog(c, np.where(ok, y * c ** 2, np.nan), lw=1.6, alpha=.95,
                  color=cm(norm(tk * tau_scale)), zorder=1)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm); sm.set_array([])
    cb = ax.figure.colorbar(sm, ax=ax, pad=.02, fraction=.05)
    cb.set_label(cbar_label, fontsize=7); cb.ax.tick_params(labelsize=6)


def compensated_ylim(ax, m, F, lo=3e-5, hi=4.0):
    """Frame a compensated panel on the steady state, not on the faintest isochrone."""
    y = np.asarray(F, float) * np.asarray(m, float) ** 2
    top = np.nanmax(np.where(np.isfinite(y) & (y > 0), y, np.nan))
    if np.isfinite(top) and top > 0:
        ax.set_ylim(top * lo, top * hi)


def plot_spectrum(ax, run, spec=None, plateau=None, compensate=True, errorbars=True,
                  color="k", label="measured"):
    """Spectrum with error bars, compensated as m^2 dN/dm by default.  Every fit is
    done on the raw dN/dm; the compensation is applied at draw time only."""
    s = spec if spec is not None else spectrum(run)
    c = np.asarray(run["centers"], float)
    f = c ** 2 if compensate else np.ones_like(c)
    k = np.isfinite(s["F"]) & (s["F"] > 0)
    if errorbars:
        ax.errorbar(c[k], (s["F"] * f)[k], yerr=(s["sigma"] * f)[k], fmt="o", ms=3,
                    lw=0, elinewidth=.8, capsize=1.5, color=color, zorder=4, label=label)
    else:
        ax.loglog(c[k], (s["F"] * f)[k], "o", ms=3, color=color, zorder=4, label=label)
    ax.set_xscale("log"); ax.set_yscale("log")
    if plateau is not None and np.isfinite(plateau.get("m_lo", np.nan)):
        xs = np.logspace(np.log10(plateau["m_lo"]), np.log10(plateau["m_hi"]), 30)
        A = anchor_amplitude(c, s["F"], plateau["m_lo"], plateau["m_hi"], plateau["alpha"])
        ax.plot(xs, A * xs ** plateau["alpha"] * (xs ** 2 if compensate else 1), "-",
                lw=2.2, color="C3", zorder=5,
                label=r"plateau $\alpha=%.2f$" % plateau["alpha"])
    ax.set_xlabel("m")
    ax.set_ylabel(r"$m^2\,dN/dm$" if compensate else "dN/dm")
    return s


def plot_growth(axs, run, iso=None, pl=None, fit=None, k=None, b_theory=None,
                color="C0", tag=""):
    """
    Two panels: <m>(tau) with error bars and every age bin visible, and the local
    slope beside it so the plateau can be seen rather than trusted.
    """
    if iso is None:
        iso, pl, fit, k = growth_law(run)
    tau, mbar, sig = iso["tau"], iso["mbar"], iso["sigma"]
    have = np.isfinite(mbar) & (mbar > 0) & (iso["counts"] > 0)

    a = axs[0]
    a.errorbar(tau[have], mbar[have], yerr=sig[have], fmt="o", ms=4, mfc="none",
               mec="0.6", mew=.9, lw=0, elinewidth=.7, ecolor="0.75",
               label="all age bins (%d)" % have.sum())
    if k is not None and k.sum():
        a.errorbar(tau[k], mbar[k], yerr=sig[k], fmt="o", ms=4, color=color, lw=0,
                   elinewidth=.9, capsize=1.5, label="on the plateau (%d)" % k.sum())
    if fit is not None and np.isfinite(fit["p"]):
        a.plot(tau[have], fit["A"] * tau[have] ** fit["p"], "--", lw=1, color="0.4",
               label="fit extrapolated")
        a.plot(tau[k], fit["A"] * tau[k] ** fit["p"], "-", lw=2.2, color="C3",
               label=r"fit $\tau^{%.3f\pm%.3f}$" % (fit["p"], fit["sigma_p"]))
    if b_theory and np.isfinite(b_theory) and k is not None and k.sum():
        a.plot(tau[k], mbar[k][0] * (tau[k] / tau[k][0]) ** b_theory, ":", lw=1.6,
               color="C1", label=r"theory $\tau^{%.1f}$" % b_theory)
    a.set_xscale("log"); a.set_yscale("log")
    # frame on the data: the extrapolated fit spans twenty decades and would own the axis
    a.set_ylim(np.nanmin(mbar[have]) / 3.0, np.nanmax(mbar[have]) * 3.0)
    a.set_xlabel(r"age $\tau$"); a.set_ylabel(r"$\langle m\rangle$")
    a.set_title("%s — growth law, all data" % tag, fontsize=9)
    a.legend(fontsize=6, loc="upper left")

    lt, lm = np.log(tau), np.log(mbar)
    idx = np.flatnonzero(have)
    slope = np.full(tau.size, np.nan)
    for j in range(1, idx.size - 1):
        slope[idx[j]] = ((lm[idx[j + 1]] - lm[idx[j - 1]]) /
                         (lt[idx[j + 1]] - lt[idx[j - 1]]))
    a2 = axs[1]
    ok = np.isfinite(slope)
    a2.semilogx(mbar[ok], slope[ok], "o-", ms=3.5, lw=.8, color="0.55",
                label="local slope")
    if k is not None:
        a2.semilogx(mbar[ok & k], slope[ok & k], "o", ms=4.5, color=color,
                    label="plateau")
    if b_theory and np.isfinite(b_theory):
        a2.axhline(b_theory, ls="--", lw=1, color="C1",
                   label="theory $b=%.1f$" % b_theory)
    if fit is not None and np.isfinite(fit["p"]):
        a2.axhline(fit["p"], ls="-", lw=1, color="C3",
                   label=r"fit $b=%.3f\pm%.3f$" % (fit["p"], fit["sigma_p"]))
        if pl is not None and np.isfinite(pl["m_lo"]):
            a2.axvspan(min(pl["m_lo"], pl["m_hi"]), max(pl["m_lo"], pl["m_hi"]),
                       color="C1", alpha=.10)
    a2.set_xlabel(r"$\langle m\rangle$")
    a2.set_ylabel(r"$d\log\langle m\rangle/d\log\tau$")
    a2.set_ylim(-0.5, (b_theory if b_theory else 6) * 1.35)
    a2.set_title("where the slope settles", fontsize=9)
    a2.legend(fontsize=6, loc="upper left")
    return iso, pl, fit, k


def growth_compare(run, groups=None, correct_repeats=True, **kw):
    """
    The growth law measured twice on the same particles: once on the age grid as
    recorded, once on the merged age bins the isochrone panel is drawn from.

    This is the check that the rebinning is honest.  Merging age bins is exactly
    what a coarser `iso_age_edges` would have produced, so b must come back the
    same; if it does not, the widened bins are straddling curvature and the picture
    is showing something the fit never saw.  The two numbers, not one, are the
    result.

    Returns dict with the fine and rebinned isochrone dicts, plateaus, fits and masks.
    """
    iso = iso_mean_mass(run, correct_repeats=correct_repeats)
    pl = growth_plateau(iso["tau"], iso["mbar"], iso["counts"], **kw)
    k = (np.isfinite(iso["mbar"]) & (iso["tau"] >= pl["tau_lo"])
         & (iso["tau"] <= pl["tau_hi"]))
    fit = wls_powerlaw(iso["tau"], iso["mbar"], iso["sigma"], mask=k)

    if groups is None:
        groups, _ = iso_groups(run, pl=pl, iso=iso, correct_repeats=correct_repeats)
    iso_g = iso_mean_mass(run, correct_repeats=correct_repeats, groups=groups)
    fit_g = (wls_powerlaw(iso_g["tau"], iso_g["mbar"], iso_g["sigma"])
             if iso_g["tau"].size >= 3 else
             dict(p=np.nan, sigma_p=np.nan, A=np.nan, chi2_dof=np.nan,
                  rms_resid=np.nan, n=0))
    return dict(iso=iso, pl=pl, fit=fit, k=k, groups=groups, iso_g=iso_g, fit_g=fit_g)


def plot_growth_compare(axs, run, cmp=None, groups=None, b_theory=None,
                        color="C0", tag=""):
    """
    The two growth laws side by side: the original fine age bins and the merged ones
    the isochrones were drawn from.  Left, <m>(tau) with both fits; right, the local
    slope of both, so a disagreement shows up as a shape rather than as a number.
    """
    if cmp is None:
        cmp = growth_compare(run, groups=groups)
    iso, pl, fit, k = cmp["iso"], cmp["pl"], cmp["fit"], cmp["k"]
    iso_g, fit_g = cmp["iso_g"], cmp["fit_g"]
    tau, mbar, sig = iso["tau"], iso["mbar"], iso["sigma"]
    have = np.isfinite(mbar) & (mbar > 0) & (iso["counts"] > 0)

    a = axs[0]
    a.errorbar(tau[have], mbar[have], yerr=sig[have], fmt="o", ms=3.5, mfc="none",
               mec="0.72", mew=.8, lw=0, elinewidth=.6, ecolor="0.82",
               label="original bins, all (%d)" % have.sum())
    if k.sum():
        a.errorbar(tau[k], mbar[k], yerr=sig[k], fmt="o", ms=4, color=color, lw=0,
                   elinewidth=.9, capsize=1.5, label="original, on plateau (%d)" % k.sum())
        if np.isfinite(fit["p"]):
            a.plot(tau[k], fit["A"] * tau[k] ** fit["p"], "-", lw=2.2, color="C3",
                   label=r"original: $b=%.3f\pm%.3f$" % (fit["p"], fit["sigma_p"]))
    tg, mg, sg = iso_g["tau"], iso_g["mbar"], iso_g["sigma"]
    if tg.size:
        a.errorbar(tg, mg, yerr=sg, fmt="s", ms=8, mfc="none", mec="C2", mew=1.6,
                   lw=0, elinewidth=1.1, ecolor="C2", capsize=2,
                   label="rebinned (%d curves)" % tg.size)
        if np.isfinite(fit_g["p"]):
            a.plot(tg, fit_g["A"] * tg ** fit_g["p"], "--", lw=1.8, color="C2",
                   label=r"rebinned: $b=%.3f\pm%.3f$" % (fit_g["p"], fit_g["sigma_p"]))
    if b_theory and np.isfinite(b_theory) and k.sum():
        a.plot(tau[k], mbar[k][0] * (tau[k] / tau[k][0]) ** b_theory, ":", lw=1.6,
               color="C1", label=r"theory $\tau^{%.1f}$" % b_theory)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_ylim(np.nanmin(mbar[have]) / 3.0, np.nanmax(mbar[have]) * 3.0)
    a.set_xlabel(r"age $\tau$"); a.set_ylabel(r"$\langle m\rangle$")
    a.set_title("%s — original vs rebinned" % tag, fontsize=9)
    a.legend(fontsize=6, loc="upper left")

    def _slope(x, y):
        s = np.full(np.size(x), np.nan)
        i = np.flatnonzero(np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
        for j in range(1, i.size - 1):
            s[i[j]] = ((np.log(y[i[j + 1]]) - np.log(y[i[j - 1]])) /
                       (np.log(x[i[j + 1]]) - np.log(x[i[j - 1]])))
        return s

    a2 = axs[1]
    sf = _slope(tau, mbar); ok = np.isfinite(sf)
    a2.semilogx(mbar[ok], sf[ok], "o-", ms=3.5, lw=.8, color="0.6",
                label="original bins")
    if k.sum():
        a2.semilogx(mbar[ok & k], sf[ok & k], "o", ms=4.5, color=color, label="plateau")
    sg2 = _slope(tg, mg); okg = np.isfinite(sg2)
    if okg.any():
        a2.semilogx(mg[okg], sg2[okg], "s--", ms=7, mfc="none", mec="C2", mew=1.5,
                    lw=1.1, color="C2", label="rebinned")
    if b_theory and np.isfinite(b_theory):
        a2.axhline(b_theory, ls="--", lw=1, color="C1",
                   label="theory $b=%.1f$" % b_theory)
    if np.isfinite(fit["p"]):
        a2.axhline(fit["p"], ls="-", lw=1, color="C3", label="original fit")
    if np.isfinite(fit_g["p"]):
        a2.axhline(fit_g["p"], ls=":", lw=1.4, color="C2", label="rebinned fit")
    if np.isfinite(pl["m_lo"]):
        a2.axvspan(min(pl["m_lo"], pl["m_hi"]), max(pl["m_lo"], pl["m_hi"]),
                       color="C1", alpha=.10)
    a2.set_xlabel(r"$\langle m\rangle$")
    a2.set_ylabel(r"$d\log\langle m\rangle/d\log\tau$")
    a2.set_ylim(-0.5, (b_theory if b_theory else 6) * 1.35)
    a2.set_title("the same slope, sampled twice", fontsize=9)
    a2.legend(fontsize=6, loc="upper left")
    return cmp


# ======================================================================
#  SAVING  --  used by the run notebooks, kept here so there is one module
# ======================================================================

def check_grid(edges, *masses):
    """
    Refuse to start if a characteristic mass falls outside the bin grid.

    The bin-pair majorant is built from bin UPPER CORNERS, and mass_to_bin CLAMPS
    anything above the top edge into the last bin -- whose corner is then smaller
    than the particle itself -- so K(m,m) exceeds its own majorant and the engine
    raises "Majorant violated" on the very first pair.  With a constant kernel that
    is impossible; with any lambda > 0 it is immediate, and the traceback points at
    the majorant rather than at the grid.
    """
    e = np.asarray(edges, float)
    bad = [float(m) for m in masses if m is not None and not (e[0] < m < e[-1])]
    if bad:
        raise ValueError("masses %s fall outside the grid (%.3g, %.3g) -- widen EDGES"
                         % (bad, e[0], e[-1]))


def _jsonable(v):
    if v is None or isinstance(v, (str, bool)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def add_last_run(run, name, analysis=None, drop_particles=True, runs_dir="runs"):
    """
    Write one finished run to runs/<name>.npz, replacing the previous one.

    Everything simulate() returns is kept except `final_mass` and `final_inj_time`:
    those carry the whole live population, one float per particle, and no figure
    needs them because the spectra and the isochrone histogram are stored in their
    own right.  `analysis` is merged into meta, which save_run stores as JSON, so it
    survives load without any engine change; `stop_reason` is copied there too
    because save_run drops top-level strings.
    """
    import os
    payload = {k: v for k, v in run.items()
               if not (drop_particles and k in ("final_mass", "final_inj_time"))}
    meta = dict(run["meta"])
    meta["stop_reason"] = run.get("stop_reason", meta.get("stop_reason"))
    meta["saved_as"] = name
    meta["particles_dropped"] = bool(drop_particles)
    if analysis:
        meta["analysis"] = {k: _jsonable(v) for k, v in analysis.items()}
    payload["meta"] = meta

    os.makedirs(runs_dir, exist_ok=True)
    path = os.path.join(runs_dir, name + ".npz")
    BF.save_run(path, payload)
    a = meta.get("analysis", {})
    if a and not np.isfinite(a.get("alpha_plateau", 0.0)):
        print("   !! plateau is nan -- saved anyway, but this run has no inertial range")
    print("saved -> %s   (%.2f MB, stop = %s)"
          % (path, os.path.getsize(path) / 1e6, meta.get("stop_reason")))
    return path
