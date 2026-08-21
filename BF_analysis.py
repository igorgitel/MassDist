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

# The analysis helpers below (local_slope, find_inertial_range, save_run, ...) are
# identical in every engine, so take whichever is on the path -- newest first.
try:
    import BF_warm_start_v5 as BF
except ImportError:
    try:
        import BF_v_no_resampling_v4 as BF
    except ImportError:
        try:
            import BF_v_no_resampling_v3 as BF
        except ImportError:
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
# [v5] present only on the warm-start engine; None on v2/v3/v4 so a notebook can
# test for them instead of failing on import.
sample_powerlaw     = getattr(BF, "sample_powerlaw", None)
steady_N_for_mass   = getattr(BF, "steady_N_for_mass", None)
ENGINE_NAME         = getattr(BF, "__name__", "?")


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
    # [v5] a warm start must announce itself: every statement about ancestry on
    # such a run is conditional on the honest fraction, and every statement
    # about alpha is conditional on the seed having been the WRONG slope.
    if m.get("seeded"):
        s = m.get("seed") or {}
        out.append("  %-16s alpha = %+.4f on [%.3g, %.3g], N = %s, <m> = %.4g"
                   % ("SEEDED (v5)", s.get("alpha", np.nan), s.get("m_lo", np.nan),
                      s.get("m_hi", np.nan), s.get("N", "?"), s.get("mean_mass", np.nan)))
    if "honest_num" in run and np.size(run["honest_num"]):
        out.append("  %-16s %.3f by number, %.3f by mass  (1.0 = no seed left)"
                   % ("honest at end", float(np.asarray(run["honest_num"])[-1]),
                      float(np.asarray(run["honest_mass"])[-1])))
    if "wait_mean" in run:
        out.append("  %-16s %.4g stays closed in %d bins"
                   % ("waiting law", float(np.asarray(run["wait_events"]).sum()),
                      int((np.asarray(run["wait_events"]) > 0).sum())))
    # [v5] ПРОИСХОЖДЕНИЕ.  Перезапуски составляются, и через месяц по одному
    # только alpha уже не восстановить, с какого чекпойнта прогон вырос и
    # сколько физического времени коробка прожила до него.
    if m.get("restarted_from"):
        out.append("  %-16s %s" % ("RESTARTED FROM", m["restarted_from"]))
        out.append("  %-16s %.4g до этого прогона + %.4g в этом = %.4g всего"
                   % ("t накоплено", m.get("t_origin", 0.0),
                      float(run.get("final_t_phys", np.nan)),
                      m.get("t_origin", 0.0) + float(run.get("final_t_phys", np.nan))))
    if a.get("checkpoint"):
        out.append("  %-16s %s (%.2f промывки), дрейф <m> %+.3f, alpha %+.4f"
                   % ("ЧЕКПОЙНТ",
                      "СТАЦИОНАРЕН" if a.get("stationary") else "точка продолжения",
                      a.get("flushes", np.nan), a.get("drift_mbar", np.nan),
                      a.get("drift_alpha", np.nan)))
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

    THE CRITERION IS THE SINK COUNTER, `n_out`, which is recorded per snapshot: the
    gate fires once the cascade has demonstrably delivered particles across the whole
    inertial range, and that is exactly the point after which the run is stationary.

    It is deliberately NOT the gate TIME.  A run with zero_time_at_tracer_start
    reports t measured from the gate, so t is negative through the fill -- and a
    stored gate time written on the old clock then lies outside the t array entirely,
    every snapshot fails the test, and the estimator silently falls back to averaging
    ONE snapshot.  That happened: a six-decade run with 202 snapshots, 184 of them
    past the gate, was reduced to K = 1.  A counter cannot be shifted, so it cannot
    fail that way.

    Falls back to the gate time, then to the last `frac` of the run, and says so.
    """
    t = np.asarray(run["t"], float)
    meta = run.get("meta", {})
    gate = float(meta.get("iso_start_sink", 0) or 0)
    n_out = np.asarray(run["n_out"], float) if "n_out" in run else None

    ss = None
    if n_out is not None and gate > 0:
        ss = n_out >= gate
        if ss.sum() >= 1 and verbose:
            print("   steady mask: %d of %d snapshots past the sink gate (n_out >= %.3g)"
                  % (ss.sum(), ss.size, gate))
    if ss is None or ss.sum() < 1:
        t0 = float(run["iso_t_begin"]) if "iso_t_begin" in run else np.nan
        if np.isfinite(t0) and t0 <= t.max():
            ss = t >= t0
        else:
            ss = np.zeros(t.size, bool)
            ss[int((1.0 - frac) * t.size):] = True
            if verbose:
                print("   no usable sink gate -- averaging the last %.0f%% of snapshots"
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


# ======================================================================
#  GENERATIONS  --  the split count, carried by every particle
# ======================================================================
#
#  WHAT THIS REPLACES.  v2/v3 measured the number of splits along a path with tagged
#  tracer chains.  The number does not need a path: it is a property of the PARTICLE,
#  one int32, so v4 carries it on all of them.  The tree count, the shelf at m_inj and
#  the censoring by the sink -- the three things that limited the tracers -- are gone,
#  because a particle contributes while it is alive and its generation does not care
#  how long it waited or whether it survives a full descent.
#
#  THE TWO WEIGHTINGS ARE NOT THE SAME, and this is the one thing to get right.
#  gen_counts counts BOTH fragments, so it is NUMBER weighted and one step of a
#  uniform split gives <ln xi> = -1, Var = 1.  gen_mass weights by mass, which is what
#  a tracer did when it followed a fragment with probability xi, and gives -1/2 and
#  1/4.  Mass weighting is the default here for a practical reason as well as a
#  principled one: the packet moves half as fast, so it stays clear of the sink for
#  twice as many generations.  Measured over three decades: number weighting departs
#  from the line at g = 4, mass weighting holds to g = 10.


def steady_mean_mass(alpha, m_lo, m_hi):
    """
    <m> = M/N for a truncated power law F ~ m^alpha on [m_lo, m_hi].  Sizes the initial
    condition: the box must hold N_ss * <m> of mass, so N0 = N_ss * <m> / m_inj.

    Not cosmetic.  The clock is dt = 2V/(wR) with R ~ N^2, so the physical time spent
    filling goes as 1/N0 and the deterministic injector delivers q*t bodies during it.
    Measured at three decades: N0 = 4 injects 705 bodies during the fill, N0 = 180
    injects 200, N0 = 1000 injects 38.  At the low end the run is a decaying cascade
    wearing the costume of an open one.
    """
    a_, lo, hi = float(alpha), float(m_lo), float(m_hi)

    def _int(e):
        return (np.log(hi / lo) if abs(e + 1.0) < 1e-12
                else (hi ** (e + 1.0) - lo ** (e + 1.0)) / (e + 1.0))

    return _int(a_ + 1.0) / _int(a_)


# ======================================================================
#  [v5]  THE WAITING LAW  --  b measured as a TIME, not as a density
# ======================================================================
#
#  <dt>(m) is the mean time a particle sits at mass m between successive
#  events that change its mass.  The engine accumulates it two ways
#  (engine note [14]):
#
#    'exposure'  occupancy / events.  Int N_b(t) dt over the number of
#                stays that ended in b.  Unbiased under censoring; use it.
#    'interval'  mean of the completed intervals.  Censored at large m,
#                which flattens the slope and INFLATES b.  Kept as the
#                cross-check: exposure and interval agreeing means the
#                run is long compared with the longest waiting time.
#
#  WHAT IT IS AND IS NOT INDEPENDENT OF.  Against a power-law background
#  the disruption rate of a body of mass m is, by homogeneity of K,
#
#      nu(m) = Int_{f m}^{m} K(m,m') n(m') dm'  ~  m^(lambda + alpha + 1),
#
#  so the slope of <dt> and alpha carry the SAME exponent information:
#
#      1/b = -(1 + lambda + alpha)          [waiting law]
#      1/b =   2 + alpha                    [flux closure]
#
#  and both have db/dalpha = b^2.  Nobody escapes the amplification --
#  b = 6 is a large number manufactured out of a small one, and that is
#  physics, not a defect in the estimator.  QUOTE 1/b, WITH ITS ERROR,
#  AND LET b BE THE DERIVED NUMBER.
#
#  What the waiting law does buy, and it is worth having:
#    - it is a TIME, measured directly, not an exponent inferred from a
#      density plus a flux argument.  The two routes above differ by
#      (1 - lambda)/2 - (2 + alpha) evaluated off the fixed point, so
#      comparing them TESTS the closure instead of assuming it;
#    - it is MEMORYLESS, so it is valid on a seeded population from t = 0
#      -- which is the entire reason the warm start exists;
#    - it is LOCAL: only the bins next to the sink are censored, not the
#      whole trajectory, which is what killed tau(g) in v4.


def waiting_band(run, pad_decades=(1.0, 0.15)):
    """The default fitting window for <dt>(m), and it is DELIBERATELY ASYMMETRIC.

    Measured on a three-decade fragmentation run, the local slope of <dt>:

        m/m_sink < ~5     slope -2.7, -1.1, -0.8, then +1.1  -- garbage
        m/m_sink ~ 7..700 slope 0.10 .. 0.30, mean ~0.17     -- the law
        top bin           slope -0.16                        -- injection pile

    The sink is a HARD truncation and it reaches about a decade up: a particle
    one bin above it is destroyed on its next event whatever the kinetics say,
    so <dt> there measures the boundary.  The injection end is a source at a
    single mass and contaminates only the bin it lands in.  Padding the two
    ends equally therefore throws away good data at the top and keeps bad data
    at the bottom -- which is exactly the mistake that turned a clean 1/b =
    0.174 into 0.220.

    pad_decades: scalar, or (pad at the sink end, pad at the injection end).
    """
    meta = run["meta"]
    m_inj = float(meta.get("injection_mass") or meta["ic"]["m"])
    m_sink = float(meta.get("sink_mass") or 0.0)
    try:
        p_lo, p_hi = float(pad_decades[0]), float(pad_decades[1])
    except (TypeError, IndexError):
        p_lo = p_hi = float(pad_decades)
    if m_sink <= 0 or not np.isfinite(m_sink):
        c = np.asarray(run["centers"], float)
        return float(c.min()), float(c.max())
    if m_sink < m_inj:                       # fragmentation: sink below
        return m_sink * 10.0 ** p_lo, m_inj / 10.0 ** p_hi
    return m_inj * 10.0 ** p_hi, m_sink / 10.0 ** p_lo   # coagulation: sink above


def waiting_law(run, estimator="exposure", gated=None, band=None,
                pad_decades=(1.0, 0.15), min_events=200.0, weighted=False,
                verbose=False, _cross=True):
    """
    Fit <dt>(m) = A m^(1/b) and return 1/b, b and the diagnostics.

    band          (m_lo, m_hi) to fit over.  Default: waiting_band(), which is
                  asymmetric for the reason given there.
    gated         True  -> the accumulator that started at the steady-state
                           gate; False -> the one that started at t = 0.
                  None (default) -> gated if it holds enough events, and the
                  choice is reported.
    min_events    a bin needs this many ended stays to enter the fit.
    weighted      False by DEFAULT, and that is not laziness.  The Poisson
                  error on a bin with 2e4 events is 0.7%, far below the real
                  bin-to-bin scatter, which is systematic (residual curvature
                  from the boundaries) and not statistical.  Weighting by the
                  Poisson error therefore hands the fit to whichever bin has
                  the most counts -- always the one nearest the sink, always
                  the worst one.  Measured on the same run: unweighted gives
                  1/b = 0.174 +- 0.006 with chi2/dof from the scatter;
                  weighted gives anything from 0.13 to 0.22 with chi2/dof
                  between 3 and 260 depending on where the window ends.
                  Switch it on only after chi2/dof has come down to order 1.

    Returns a dict with p = 1/b and sigma_p (from the scatter), b and
    sigma_b = b^2 sigma_p, `p_spread` -- the SYSTEMATIC, i.e. how much 1/b
    moves when the window is trimmed -- and the other estimator's slope, so
    censoring can be read off rather than assumed.
    """
    if "wait_mean" not in run:
        raise ValueError("this run carries no waiting law "
                         "(engine older than v5, or track_waiting=False)")
    c = np.asarray(run["centers"], float)
    meta = run["meta"]
    m_inj = float(meta.get("injection_mass") or meta["ic"]["m"])
    m_sink = float(meta.get("sink_mass") or 0.0)

    # --- which accumulator ------------------------------------------------
    def _pair(g):
        sfx = "_gated" if g else ""
        return (np.asarray(run["wait_mean" + sfx], float),
                np.asarray(run["wait_sem" + sfx], float),
                np.asarray(run["wait_events" + sfx], float))

    if gated is None:
        _, _, n_g = _pair(True)
        _, _, n_u = _pair(False)
        gated = bool(n_g.sum() >= 0.5 * n_u.sum() and n_g.sum() > 0)
    y, sy, n = _pair(gated)
    if estimator == "interval":
        sfx = "_gated" if gated else ""
        y = np.asarray(run["wait_mean_interval" + sfx], float)
        sy = np.asarray(run.get("wait_sem_interval", np.full_like(y, np.nan)), float)
        n = np.asarray(run["wait_n" + ("_gated" if gated else "")], float)
    elif estimator != "exposure":
        raise ValueError("estimator must be 'exposure' or 'interval'")

    # --- the fitting window ----------------------------------------------
    if band is None:
        band = waiting_band(run, pad_decades)
    band = (min(band), max(band))
    mask = (c >= band[0]) & (c <= band[1]) & (n >= float(min_events)) \
        & np.isfinite(y) & (y > 0)

    fit = wls_powerlaw(c, y, sy=(sy if weighted else None), mask=mask)
    p, sp = fit["p"], fit["sigma_p"]

    # --- SYSTEMATIC: how much does 1/b move if the window is trimmed? -----
    # The statistical error above is the scatter about one particular window.
    # It says nothing about the choice of window, which is the larger error on
    # a shallow slope.  Trim a quarter and a half decade off each end and
    # report the full spread; if that exceeds sigma_p, the window is the
    # dominant uncertainty and quoting sigma_p alone is dishonest.
    trials = []
    for dlo in (0.0, 0.25, 0.5):
        for dhi in (0.0, 0.25):
            bb = (band[0] * 10.0 ** dlo, band[1] / 10.0 ** dhi)
            mk = (c >= bb[0]) & (c <= bb[1]) & (n >= float(min_events)) \
                & np.isfinite(y) & (y > 0)
            if mk.sum() >= 5:
                q = wls_powerlaw(c, y, sy=(sy if weighted else None), mask=mk)["p"]
                if np.isfinite(q):
                    trials.append(q)
    p_spread = float(np.max(trials) - np.min(trials)) if len(trials) > 1 else np.nan
    b = 1.0 / p if (np.isfinite(p) and p != 0.0) else np.nan
    # db/d(1/b) = -b^2.  The SIGN of the amplification does not matter, its
    # size does: at b = 6 a 1% error in the slope is a 6% error in b.
    sb = (b ** 2) * sp if np.isfinite(b) and np.isfinite(sp) else np.nan

    # --- the other estimator, for the censoring check ---------------------
    # _cross=False on the inner call, or the two estimators call each other for
    # ever.  One level deep is all the cross-check needs.
    other = "interval" if estimator == "exposure" else "exposure"
    p_other = np.nan
    if _cross:
        try:
            p_other = waiting_law(run, estimator=other, gated=gated, band=band,
                                  pad_decades=pad_decades, min_events=min_events,
                                  weighted=weighted, _cross=False)["p"]
        except Exception:
            p_other = np.nan

    # `n` stays what wls_powerlaw meant by it -- the number of BINS in the fit.
    # The per-bin event counts go in under their own name; overloading `n` here
    # is how the report ended up trying to print an array with %d.
    out = dict(fit)
    out.update(p=p, sigma_p=sp, b=b, sigma_b=sb, mask=mask, m=c, dt=y, sem=sy,
               n_events=n, band=tuple(band), gated=bool(gated), estimator=estimator,
               p_other=p_other, other=other, p_spread=p_spread, weighted=bool(weighted),
               sigma_b_sys=(b ** 2) * p_spread if np.isfinite(b) and np.isfinite(p_spread)
               else np.nan,
               decades=(np.log10(c[mask].max() / c[mask].min()) if mask.sum() > 1
                        else np.nan))
    if verbose:
        print(waiting_report(run, out))
    return out


def waiting_report(run, wl=None, alpha=None, sigma_alpha=None):
    """Text block: 1/b and b from the waiting law, next to b from the closure,
    with the censoring check and the honest fraction.  Everything a reader needs
    to decide whether to believe the number."""
    wl = wl if wl is not None else waiting_law(run)
    lam = run["meta"].get("lambda")
    L = ["-" * 74,
         "WAITING LAW   <dt>(m) ~ m^(1/b)      [%s estimator, %s]"
         % (wl["estimator"], "gated" if wl["gated"] else "ungated, from t=0"),
         "  window        %.3g .. %.3g   (%d bins, %.2f decades)"
         % (wl["band"][0], wl["band"][1], wl["n"], wl["decades"]),
         "  1/b           %+.5f +- %.5f (stat) +- %.5f (window)   chi2/dof = %s"
         % (wl["p"], wl["sigma_p"], wl["p_spread"],
            "n/a" if not np.isfinite(wl["chi2_dof"]) else "%.2f" % wl["chi2_dof"]),
         "  b             %.3f +- %.3f (stat) +- %.3f (window)"
         % (wl["b"], wl["sigma_b"], wl["sigma_b_sys"]),
         "                (sigma_b = b^2 sigma_(1/b): the amplification is physics,"
         " not a defect)"]
    if np.isfinite(wl["p_other"]):
        d = wl["p_other"] - wl["p"]
        L.append("  cross-check   %s estimator gives 1/b = %+.5f  (%+.5f, %s)"
                 % (wl["other"], wl["p_other"], d,
                    "consistent" if abs(d) <= 2 * wl["sigma_p"]
                    else "DIFFERENT -> censoring, run longer"))
    if lam is not None and np.isfinite(lam):
        L.append("  theory        1/b = (1-lambda)/2 = %+.5f   ->  b = %.3f"
                 % (0.5 * (1 - lam), 2.0 / (1.0 - lam)))
    if alpha is not None and np.isfinite(alpha):
        # 1/b = 2 + alpha  (flux closure) and 1/b = -(1+lambda+alpha) (kinetic).
        # Printing both is the point: they agree only ON the fixed point, so the
        # gap between them measures how far this run still is from it.
        L.append("  from alpha    alpha = %+.4f%s" %
                 (alpha, "" if sigma_alpha is None else " +- %.4f" % sigma_alpha))
        L.append("                closure  1/b = 2+alpha        = %+.5f  ->  b = %.3f"
                 % (2 + alpha, 1.0 / (2 + alpha) if (2 + alpha) != 0 else np.nan))
        if lam is not None and np.isfinite(lam):
            k = -(1.0 + lam + alpha)
            L.append("                kinetic  1/b = -(1+lam+alpha) = %+.5f  ->  b = %.3f"
                     % (k, 1.0 / k if k != 0 else np.nan))
            L.append("                measured minus kinetic     = %+.5f"
                     % (wl["p"] - k))
    hn = run.get("honest_num")
    if hn is not None and np.size(hn):
        L.append("  honest        %.3f of the population has a real history"
                 % float(np.asarray(hn)[-1]))
    if run["meta"].get("seeded"):
        s = run["meta"].get("seed") or {}
        L.append("  SEEDED        alpha_seed = %+.4f on [%.3g, %.3g], N = %s"
                 % (s.get("alpha", np.nan), s.get("m_lo", np.nan),
                    s.get("m_hi", np.nan), s.get("N", "?")))
        L.append("                the measured alpha is only meaningful if it has"
                 " MOVED off this value")
    mi, mo = float(np.asarray(run["M_in"])[-1]), float(np.asarray(run["M_out"])[-1])
    L.append("  mass balance  M_out/M_in = %.3f   (1.0 = the box is flushed)"
             % (mo / mi if mi > 0 else np.nan))
    L.append("-" * 74)
    return "\n".join(L)


def time_avg_spectrum(run, gated=None):
    """dN/dm averaged over the run rather than sampled at snapshots.

    The engine's occupancy integral Int N_b(t) dt is exact and free (it is the
    denominator of the waiting law), so this spectrum is smooth where the
    snapshot average is ragged, and it is weighted by TIME rather than by
    however the snapshots happened to fall.  Returns (m, F) or (None, None) on
    a pre-v5 run."""
    if "dndm_time_avg" not in run:
        return None, None
    if gated is None:
        gated = bool(np.isfinite(run.get("wait_t_span_gated", np.nan))
                     and float(run["wait_t_span_gated"]) > 0)
    key = "dndm_time_avg_gated" if gated else "dndm_time_avg"
    return np.asarray(run["centers"], float), np.asarray(run[key], float)


def seed_spectrum(alpha, m_lo, m_hi, N, rng=None):
    """N masses from n(m) ~ m^alpha on [m_lo, m_hi].  Re-exported from the engine
    so a notebook can draw the seed it is about to hand to simulate()."""
    if sample_powerlaw is None:
        raise ValueError("the engine on the path has no seeding (need v5)")
    return sample_powerlaw(alpha, m_lo, m_hi, N, rng)


def honesty(run):
    """The seeded fraction as a time series: how much of the run is usable for
    anything that depends on ancestry (generations, isochrones, ages).  On a cold
    run this is 1.0 everywhere by construction."""
    if "honest_num" not in run:
        return None
    return {"t": np.asarray(run["t"], float),
            "n_out": np.asarray(run["n_out"], float),
            "num": np.asarray(run["honest_num"], float),
            "mass": np.asarray(run["honest_mass"], float),
            "seeded": bool(run["meta"].get("seeded", False))}


# ======================================================================
#  [v5]  СТАЦИОНАРНОСТЬ  --  когда можно снимать чекпойнт
# ======================================================================
#
#  ДВА ПОПУЛЯРНЫХ КРИТЕРИЯ НЕ РАБОТАЮТ, и это измерено, а не предположено.
#
#  M_out/M_in.  Из точного тождества M_out = M_in + M_sys(0) - M_sys(t)
#  следует M_out/M_in = 1 + dM_sys/M_in, и это стремится к единице просто
#  потому, что M_in растёт без предела.  Критерий говорит одно: прогон
#  длиннее заполнения.  Про ФОРМУ спектра он не говорит ничего.
#
#  dn_out/devents.  Выходит на единицу после пятой части одной промывки и
#  дальше стоит.  Это баланс ЧИСЛА, а число живёт у стока и равновесится
#  на t_turn, тогда как форма живёт наверху и равновесится на tau_res --
#  в <m>/m_sink раз дольше (на шести декадах в 45 раз).
#
#  Замерено на холодном шестидекадном прогоне, в промывках коробки
#  (одна промывка = M_sys/m_sink событий):
#
#      промывок  live/1e6   <m>     alpha    dn_out/dev   M_out/M_in
#        0.18      14.3     16.2   -1.890       0.96
#        0.46       9.4     22.5   -1.881       1.01
#        0.73       8.1     24.4   -1.879       1.02
#        1.01       6.9     26.6   -1.862       1.02
#        1.29       5.9     30.6   -1.829       1.02         1.016
#
#  Оба «критерия» отрапортовали успех на 0.2 промывки.  При этом <m> росло
#  по 14 за промывку и alpha ехало, причём в конце быстрее, чем в начале.
#  Работают только <m> и alpha, сравненные между половинами хвоста.


def stationarity(run, tail=0.5, tol_mbar=0.03, tol_alpha=0.010,
                 tol_msys=0.03, band_pad=1.0, verbose=False):
    """ДОПУСКИ ЗАДАНЫ НА ОДНУ ПРОМЫВКУ, а не на окно сравнения.  Это не деталь,
    а единственное, что делает тест осмысленным на коротком прогоне.

    Сдвиг между половинами хвоста тем меньше, чем короче прогон -- окно сжимается
    вместе с ним.  Сравнивать такой сдвиг с фиксированным порогом значит объявлять
    стационарным всё, что достаточно коротко.  Замерено на двух холодных прогонах
    одной и той же конфигурации:

        длина 1.31 промывки:  сдвиг <m> +13.0%  на разносе 0.33 промывки
        длина 0.29 промывки:  сдвиг <m>  +2.8%  на разносе 0.073 промывки

    Второй прошёл бы порог 3% и получил бы «СТАЦИОНАРЕН» при <m> = 16.8 против
    предельных 45 и M_out/M_in = 0.32.  Поделённые на разнос, оба дают ОДНУ И ТУ
    ЖЕ скорость -- 39% на промывку, -- и оба честно объявляются едущими."""
    """Стационарен ли прогон настолько, чтобы снимать с него чекпойнт.

    Берётся ХВОСТ прогона (последняя доля `tail` снимков), делится пополам, и
    сравниваются средние по половинам.  Сравнение половин хвоста, а не начала с
    концом: нас интересует, движется ли система СЕЙЧАС, а не насколько далеко
    она ушла от начального условия.

    Проверяется:
      mbar   <m> = M_sys/N        -- самый острый индикатор формы
      alpha  наклон на охранной полосе, снимок за снимком
      msys   полная масса         -- производная, а не кумулятивное отношение
      dnout  dn_out/devents       -- слабый, оставлен для полноты картины

    Возвращает dict с посуточными числами, вердиктом `ok` и текстом `report`.
    Допуски по умолчанию: 3% на <m> и M_sys, 0.010 на alpha -- последнее
    примерно вчетверо меньше типичного разброса плато, то есть требование
    строже, чем точность, с которой alpha вообще меряется.
    """
    e = np.asarray(run["events"], float)
    if e.size < 8:
        return {"ok": False, "report": "снимков меньше восьми -- судить не о чем",
                "n": int(e.size)}
    lv = np.asarray(run["live"], float)
    Ms = np.asarray(run["M_sys"], float)
    no = np.asarray(run["n_out"], float)
    c = np.asarray(run["centers"], float)
    D = np.asarray(run["dndm"], float)
    meta = run["meta"]
    m_inj = float(meta.get("injection_mass") or meta["ic"]["m"])
    m_sink = float(meta.get("sink_mass") or 0.0)
    lo, hi = (m_sink, m_inj) if m_sink < m_inj else (m_inj, m_sink)
    band = (c >= lo * 10.0 ** band_pad) & (c <= hi / 10.0 ** band_pad)

    mbar = np.where(lv > 0, Ms / np.maximum(lv, 1), np.nan)
    al = np.full(e.size, np.nan)
    for i in range(e.size):
        k = band & (D[i] > 0)
        if k.sum() >= 6:
            al[i] = np.polyfit(np.log(c[k]), np.log(D[i][k]), 1)[0]

    i0 = max(int(e.size * (1.0 - tail)), 0)
    idx = np.arange(i0, e.size)
    half = i0 + (e.size - i0) // 2
    A, B = np.arange(i0, half), np.arange(half, e.size)
    if A.size < 2 or B.size < 2:
        return {"ok": False, "report": "хвост слишком короткий для двух половин",
                "n": int(e.size)}

    #  ОШИБКА СДВИГА, и без неё вердикт был бы гаданием.  Снимки СИЛЬНО
    #  коррелированы -- соседние делят почти всю популяцию, -- поэтому наивная
    #  ошибка по числу снимков занижена в разы.  effective_snapshots даёт
    #  K(1-rho)/(1+rho) по автокорреляции на лаге один; на реальных прогонах
    #  из 168 снимков независимых оказывается около четырёх.
    def _cmp(x, rel):
        a, b = np.nanmean(x[A]), np.nanmean(x[B])
        sa = np.nanstd(x[A]) / np.sqrt(max(effective_snapshots(x[A]), 1.0))
        sb = np.nanstd(x[B]) / np.sqrt(max(effective_snapshots(x[B]), 1.0))
        sd = float(np.hypot(sa, sb))
        if rel:
            d = (b - a) / a if np.isfinite(a) and a != 0 else np.nan
            sd = sd / abs(a) if np.isfinite(a) and a != 0 else np.nan
        else:
            d = b - a
        return a, b, d, sd

    mb_a, mb_b, d_mb, s_mb = _cmp(mbar, True)
    ms_a, ms_b, d_ms, s_ms = _cmp(Ms, True)
    al_a, al_b, d_al, s_al = _cmp(al, False)
    dn = (no[-1] - no[i0]) / max(e[-1] - e[i0], 1.0)

    #  РАЗНОС половин в промывках -- на него и делится сдвиг.  Берутся средние
    #  позиции половин, а не их края: сдвиг между средними и относится к
    #  расстоянию между средними.
    flush0 = Ms[0] / m_sink if m_sink > 0 else np.nan
    span = (float(e[B].mean()) - float(e[A].mean())) / flush0 if np.isfinite(flush0) else np.nan
    if not np.isfinite(span) or span <= 0:
        span = np.nan
    d_mb, s_mb = d_mb / span, s_mb / span
    d_ms, s_ms = d_ms / span, s_ms / span
    d_al, s_al = d_al / span, s_al / span

    flush = flush0
    #  ТРИ исхода, а не два.  Сдвиг может быть (а) мал -- сошлось; (б) велик И
    #  значим -- едет; (в) велик, но в пределах собственной ошибки -- сказать
    #  нечего, нужна статистика.  Без третьего случая тест на коротком прогоне
    #  уверенно объявлял бы «едет» по шуму.
    checks = [("<m>", d_mb, s_mb, tol_mbar), ("alpha", d_al, s_al, tol_alpha),
              ("M_sys", d_ms, s_ms, tol_msys)]

    def _verdict(v, sd, t):
        if not np.isfinite(v):
            return "?"
        if abs(v) <= t:
            return "ok"
        return "ЕДЕТ" if (np.isfinite(sd) and abs(v) > 2.0 * sd) else "шум?"

    verd = [_verdict(v, sd, t) for _, v, sd, t in checks]
    ok = all(x == "ok" for x in verd)
    noisy = any(x == "шум?" for x in verd)

    L = ["-" * 70,
         "СТАЦИОНАРНОСТЬ  (две половины последних %.0f%% снимков)" % (100 * tail),
         "  длина прогона   %.3g событий = %.2f промывки коробки" % (e[-1], e[-1] / flush)
         if np.isfinite(flush) else "  длина прогона   %.3g событий" % e[-1],
         "  разнос половин  %.4f промывки -- на него поделены все сдвиги" % span,
         "  %-8s %11s %11s %18s  %-7s %s"
         % ("", "первая", "вторая", "сдвиг/промывку", "порог", "вердикт")]
    for (name, v, sd, t), vd in zip(checks, verd):
        a, b = ((mb_a, mb_b) if name == "<m>" else
                (al_a, al_b) if name == "alpha" else (ms_a, ms_b))
        L.append("  %-8s %11.4g %11.4g %18s  %-7.3g %s"
                 % (name, a, b, "%+.4f +- %.4f" % (v, sd), t, vd))
    L.append("  %-8s %11.3f %11s %11s  %-8s %s"
             % ("dn/dev", dn, "", "", "1.00",
                "(слабый критерий: садится на единицу задолго до формы)"))
    mi = float(np.asarray(run["M_in"])[-1]); mo = float(np.asarray(run["M_out"])[-1])
    L.append("  %-8s %11.3f %11s %11s  %-8s %s"
             % ("M_o/M_i", mo / mi if mi > 0 else np.nan, "", "", "1.00",
                "(слабый: -> 1 автоматически, см. заметку выше)"))
    L.append("  ВЕРДИКТ: %s"
             % ("СТАЦИОНАРЕН -- годится как старт измерения" if ok else
                ("НЕ СТАЦИОНАРЕН -- продолжать" if not noisy else
                 "НЕ СТАЦИОНАРЕН, но часть сдвигов в пределах шума -- "
                 "нужна длина, а не вердикт")))
    if noisy:
        L.append("  («шум?»: сдвиг больше допуска, но меньше двух своих ошибок --")
        L.append("   отличить движение от флуктуации на этой длине нельзя.)")
    else:
        L.append("  (ошибка сдвига считана по НЕЗАВИСИМЫМ снимкам: соседние делят")
        L.append("   популяцию, и наивная ошибка по их числу занижена в разы.)")
    # ---- сколько ещё, если не сошлось ----------------------------------
    #  Модель: <m>(n) = M_inf - A exp(-n/n0) по промывкам n.  Тогда ДРЕЙФ,
    #  измеренный на окне, сам затухает как exp(-n/n0) с тем же n0.  Значит n0
    #  достаётся сравнением дрейфа на двух последовательных окнах, а число
    #  промывок до допуска есть n0 * ln(d/tol).  Линейная экстраполяция тут не
    #  годится в принципе: при постоянной скорости дрейфа сходимость не
    #  наступает никогда, и «ещё столько-то» было бы выдумкой.
    est = np.nan
    if not ok and np.isfinite(d_mb) and abs(d_mb) > tol_mbar and np.isfinite(flush):
        t0 = max(int(e.size * (1.0 - min(tail * 1.5, 0.9))), 0)
        cut = np.array_split(np.arange(t0, e.size), 3)
        if all(c_.size >= 2 for c_ in cut):
            mm = [float(np.nanmean(mbar[c_])) for c_ in cut]
            d1 = (mm[1] - mm[0]) / mm[0]
            d2 = (mm[2] - mm[1]) / mm[1]
            dn_fl = float(e[cut[2]].mean() - e[cut[1]].mean()) / flush
            if np.isfinite(d1) and np.isfinite(d2) and d1 * d2 > 0 and abs(d2) < abs(d1):
                n0 = dn_fl / np.log(abs(d1) / abs(d2))
                est = n0 * np.log(abs(d_mb) / tol_mbar)
                L.append("  оценка: дрейф <m> затухает с масштабом %.2f промывки," % n0)
                L.append("          до допуска %.0f%% ещё около %.1f промывки = %.3g событий"
                         % (100 * tol_mbar, est, est * flush))
            else:
                L.append("  оценка: дрейф <m> НЕ затухает на хвосте (%+.3f -> %+.3f) --"
                         % (d1, d2))
                L.append("          прогон далеко от стационара, срок назвать нельзя")
    L.append("-" * 70)
    rep = "\n".join(L)
    if verbose:
        print(rep)
    return {"ok": bool(ok), "report": rep, "flushes": float(e[-1] / flush) if np.isfinite(flush) else np.nan,
            "mbar": (mb_a, mb_b, d_mb, s_mb), "alpha": (al_a, al_b, d_al, s_al),
            "msys": (ms_a, ms_b, d_ms, s_ms), "dn_dev": float(dn),
            "verdicts": dict(zip(("mbar", "alpha", "msys"), verd)),
            "noisy": bool(noisy),
            "flushes_left": float(est),
            "mbar_series": mbar, "alpha_series": al, "events": e}


def generations(run, weight="mass"):
    """
    Unpack the generation histogram.  weight is 'mass' or 'number'.

    Returns dict(g, H, centers, widths, x, num, tau, n_eff) where H[g] is the mass (or
    number) distribution of generation g over the mass grid, and x = ln(m/m_inj).
    """
    if "gen_counts" not in run:
        raise ValueError("this run carries no generation histogram (engine older than v4)")
    H = np.asarray(run["gen_mass" if weight == "mass" else "gen_counts"], float)
    c = np.asarray(run["centers"], float)
    minj = float(run["meta"].get("injection_mass") or run["meta"]["ic"]["m"])
    return {"g": np.arange(H.shape[0]), "H": H, "centers": c,
            "widths": np.asarray(run["widths"], float), "x": np.log(c / minj),
            "num": np.asarray(run["gen_num"], float),
            "tau": np.asarray(run["gen_tau"], float),
            "tau_live": np.asarray(run.get("gen_tau_live", run["gen_tau"]), float),
            "reach_n": np.asarray(run.get("gen_reach_n", run["gen_num"]), float),
            "m_inj": minj, "m_sink": float(run["meta"].get("sink_mass") or 0.0),
            "weight": weight, "snapshots": float(run.get("gen_snapshots", 0.0))}


def gen_moments(gen, min_count=500.0, clear_sink=2.0):
    """
    <x> and Var(x) per generation, with a VALIDITY MASK.

    The mask is the whole point.  The sink truncates the packet from below, so a
    generation whose distribution has reached it is measuring the boundary, not the
    walk -- its mean saturates and its variance collapses towards zero.  A generation
    counts as usable while

        <x> - clear_sink * sigma  >  ln(m_sink / m_inj),

    i.e. while the packet is still `clear_sink` standard deviations clear of the sink.
    Fitting through the saturated tail is the documented way to get a clean-looking
    slope that is simply wrong.
    """
    H, x = gen["H"], gen["x"]
    xs = np.log(gen["m_sink"] / gen["m_inj"]) if gen["m_sink"] > 0 else -np.inf
    n = H.sum(axis=1)
    mu = np.full(H.shape[0], np.nan); va = np.full(H.shape[0], np.nan)
    for g in range(H.shape[0]):
        if n[g] < min_count:
            continue
        p = H[g] / n[g]
        mu[g] = float((p * x).sum())
        va[g] = float((p * (x - mu[g]) ** 2).sum())
    ok = np.isfinite(mu) & (mu - clear_sink * np.sqrt(np.maximum(va, 0)) > xs)
    ok[0] = False                      # generation 0 has not moved yet

    #  SECOND CUT, and it matters for tau(g) rather than for the moments.  Every split
    #  makes two particles out of one, so the number that REACH generation g DOUBLES
    #  with g -- it is not flat, and treating it as flat is wrong.  What is flat is the
    #  ratio reach_n[g] / reach_n[g-1] = 2, and it stays at 2 exactly until the sink
    #  starts removing fragments before they can split again.  Measured on a
    #  three-decade test: 1.93, 1.96, 2.03, 2.09, 2.05, then 1.94, 1.76, 1.60, 1.43 --
    #  the sink bites at g = 7.  Beyond that the particles that reached g are the FAST
    #  ones, a survivor bias that compresses tau(g) and drags b down.
    rn = np.asarray(gen.get("reach_n", n), float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.full(rn.size, np.nan)
        ratio[1:] = rn[1:] / np.where(rn[:-1] > 0, rn[:-1], np.nan)
    ok_flux = np.nan_to_num(ratio, nan=0.0) > 1.8
    return {"g": gen["g"], "mu": mu, "var": va, "n": n, "ok": ok,
            "ok_flux": ok & ok_flux, "reach_n": rn, "reach_ratio": ratio,
            "x_sink": xs}


def gen_growth(gen, mom=None, min_count=500.0, s_max=120.0):
    """
    b from <tau>(g): the mean time to reach generation g, over the whole population.

    Same statement as the tracer version, on 10^4 times the sample.  Splits happen at
    rate nu(m) ~ m^(beta-1) and <ln m> falls by <ln xi> per split, so with
    beta - 1 = -1/b,

        tau(g) = T (1 - exp(-g / (2b / |<ln xi>| / 2)))   ->  scale = b / |<ln xi>|.

    With the MASS weighting <ln xi> = -1/2 and the scale is 2b, exactly as for the
    tracers; with NUMBER weighting <ln xi> = -1 and the scale is b.  `resid` against
    `resid_linear` is the verdict: a tau(g) that is barely curved does not determine b.
    """
    if mom is None:
        mom = gen_moments(gen, min_count=min_count)
    #  tau(g) is fitted on the flux-flat range only -- see gen_moments.  Outside it the
    #  sample at each g is the fast tail and the curve is compressed.
    k = mom.get("ok_flux", mom["ok"]) & np.isfinite(gen["tau"])
    if k.sum() < 4:
        return {"b": np.nan, "T": np.nan, "g": gen["g"][k], "tau": gen["tau"][k],
                "resid": np.nan, "resid_linear": np.nan, "scale": np.nan}
    gf, tf = gen["g"][k].astype(float), gen["tau"][k]
    sg = np.linspace(0.5, s_max, int(4 * s_max) + 1)
    rs = []
    for sv in sg:
        h = 1.0 - np.exp(-gf / sv)
        T = np.sum(tf * h) / np.sum(h * h)
        rs.append(np.std(tf - T * h))
    rs = np.array(rs); sb = float(sg[np.argmin(rs)])
    h = 1.0 - np.exp(-gf / sb); T = float(np.sum(tf * h) / np.sum(h * h))
    lin = np.polyfit(gf, tf, 1)
    mu_step = 0.5 if gen["weight"] == "mass" else 1.0
    return {"b": sb * mu_step, "scale": sb, "T": T, "g": gf, "tau": tf,
            "resid": float(rs.min() / tf.std()),
            "resid_linear": float(np.std(tf - np.polyval(lin, gf)) / tf.std())}


def step_stats(frag_split_width, weight="mass"):
    """
    (<ln xi>, Var(ln xi)) for ONE split, exactly, by quadrature.

    weight='mass'   -- the piece is met with probability xi (p ~ xi): -1/2, 1/4 at w=1/2.
    weight='number' -- both pieces counted equally:                   -1,   1    at w=1/2.

    No free parameter, which is what makes <x> and Var(x) against g the sharpest test
    in the analysis.  The skewness of one step is -2 whenever the piece can be
    arbitrarily small, because ln of it then has an exponential tail; only "always the
    heavier" is bounded and comes out nearly symmetric.  Skewness after g steps falls
    as 1/sqrt(g), so it is a small-g transient, not an obstruction.
    """
    xi = np.linspace(0.5 - frag_split_width, 0.5 + frag_split_width, 200001)[1:-1]
    tz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    p = (xi if weight == "mass" else np.ones_like(xi))
    p = p / tz(p, xi)
    mu = float(tz(p * np.log(xi), xi))
    return mu, float(tz(p * np.log(xi) ** 2, xi) - mu ** 2)


def add_last_run(run, name, analysis=None, drop_particles=True, runs_dir="runs"):
    """
    Write one finished run to runs/<name>.npz, replacing the previous one.

    Everything simulate() returns is kept except `final_mass`, `final_inj_time` and
    `final_age`: those carry the whole live population, one float per particle, and no
    figure needs them because the spectra and the isochrone histogram are stored in
    their own right.  (`final_age` arrived with the v3 spin-up, where the triple
    (final_mass, final_age) IS the initial condition of the next run -- so if that is
    what you are saving, pass drop_particles=False and save it under its own name;
    it does not belong in a run archive that the analysis notebooks reload.)

    EVERYTHING TRACER-RELATED IS KEPT: `tracer_hist`, `tracer_done`, `tracer_t0_all`,
    `tracer_m0_all`.  The full trajectory log is the point of the feature and it is
    chain-indexed, not particle-indexed, so it does not scale with N.  `analysis` is merged into meta, which save_run stores as JSON, so it
    survives load without any engine change; `stop_reason` is copied there too
    because save_run drops top-level strings.
    """
    import os
    payload = {k: v for k, v in run.items()
               if not (drop_particles and k in ("final_mass", "final_inj_time",
                                                "final_age", "final_gen",
                                                "final_gen_time"))}
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
