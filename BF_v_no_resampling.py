"""
BF_v_no_resampling.py  --  Majorant_v2

*** BASELINE VARIANT: NO RESAMPLING, NO THINNING. ***
One simulated particle is one physical particle, w == 1 for the whole run.
Everything else -- majorant, clock, snapshot modes, analysis helpers --
is identical to BF.py, so the two can be compared line for line.
======================================================================
ONE event-driven Monte Carlo engine for the mass-space cascade, covering
all four configurations with the same code path:

        process = 'coagulation' | 'fragmentation'
        system  = 'closed'      | 'open'

The physics that differs between the four cases enters only through
(a) what an accepted event does to the two selected particles, and
(b) the boundary conditions (injection + absorbing sink, or none).
Everything else -- majorant construction, pair selection, the clock,
the bookkeeping -- is shared.

----------------------------------------------------------------------
METHOD AND THE TRICKS USED, WITH CITATIONS
----------------------------------------------------------------------

[1] Majorant Collision Frequency (MCF) / majorant-kernel thinning.
    Instead of computing the exact pair rate for every pair (O(N^2)),
    a majorant  fmaj >= K  is used to drive a dominating Poisson
    process; events are then thinned with acceptance K/fmaj.
      - Ivanov & Rogasinsky, Sov. J. Numer. Anal. Math. Modelling 3,
        453 (1988)                              [MCF scheme in DSMC]
      - Garcia, van den Broeck, Aertsens & Serneels,
        Physica A 143, 535 (1987)               [majorant for coagulation]
      - Gillespie, J. Atmos. Sci. 32, 1977 (1975)
                                                [exact stochastic coalescence]

[2] Bin-pair majorant table.  The majorant is not a single global
    number but a matrix fmaj[b,c] over pairs of mass bins, built once
    from the bin upper corners.  This keeps the acceptance ratio O(1)
    in every bin pair even when the mass range spans many decades, and
    -- crucially -- keeps the m^lambda scaling of the kernel inside the
    clock.  A single running global f_max instead tracks the extreme of
    the mass distribution, collapses the acceptance, and silently
    removes the kernel dependence from dt.
      - Eibeck & Wagner, SIAM J. Sci. Comput. 22, 802 (2000)
        doi:10.1137/S1064827599353488           [binned / efficient version]

[3] O(1) bin membership with swap-with-last.
    bins[b] is a python list of particle indices; pos_in_bin[i] is the
    position of i inside that list.  Removal swaps the last element
    into the hole, so add/remove are O(1) with no reallocation.
      - standard "swap-and-pop" idiom; see e.g. Goodson & Kraft,
        J. Comput. Phys. 183, 210 (2002) for the same idea used with a
        binary tree for O(log N) selection.

[4] Incremental rate maintenance.
    T[b] = sum_c fmaj[b,c] N_c is updated in O(B) by a single vector
    operation when one bin count changes, instead of recomputing the
    full matrix-vector product.  Selection is done with a vectorised
    cumsum + searchsorted rather than a python loop, which is the
    dominant cost otherwise.  For very large B a Fenwick tree gives
    O(log B); see Goodson & Kraft (2002).

[5] THE CLOCK  --  the one thing that is easy to get wrong.

    (i)  ORDERED vs UNORDERED PAIRS.  The maintained total
             R = sum_b N_b (T_b - fmaj[b,b]) = sum_{i != j} fmaj_ij
         counts every unordered pair TWICE.  The physical majorant rate
         runs over unordered pairs, so
                       dt = 2 / R,        NOT  1 / R.
         Pair selection is unaffected (both orderings lead to the same
         unordered pair, and the factors cancel); only the unit of time
         is.  Getting this wrong rescales t by exactly 2 and breaks any
         comparison with the analytic solution.

    (ii) PARTICLE WEIGHTS.  Here w is a CONSTANT (default 1): one
         simulated particle is one physical particle.  It is still
         written into the step,
                       dt = 2 * V / (w * R),
         so that the formula matches BF.py exactly and a constant w != 1
         simply rescales the density.  Nothing in this file ever changes
         w, so the clock cannot be corrupted by weight bookkeeping --
         which is the whole reason this baseline exists.

    (iii) dt must be QUADRATIC in the particle number, because the
          process is binary.  A step linear in N describes a unary
          process and gives the wrong N(t).

[6] Deterministic (residual) injection.
    Injection uses a fractional accumulator rather than a Poisson draw,
    which removes shot noise at the injection scale.  This is a
    modelling choice and is stated in the output metadata.

[7] Isochrone (age) bookkeeping.
    Every particle carries its injection time.  For FRAGMENTATION the
    age of a fragment is unambiguous: fragments are the descendants of
    exactly one parent.  For COAGULATION a merged particle has TWO
    ancestors and an inheritance rule must be chosen; it is exposed as
    a parameter (`age_rule`) precisely because the isochrone analysis
    depends on it and the choice must be reported and tested.

[8] LOCALITY OF THE FRAGMENTATION RULE  --  a physics trap, not a bug.

    If "the heavier of the pair breaks, whatever hit it", then the rate
    at which a particle of mass m0 is disrupted is

        nu(m0) ~ integral over m' < m0 of F(m') K(m0,m') dm'
               ~ integral of x^alpha dx  near x = m'/m0 -> 0,

    which DIVERGES at the lower limit whenever alpha < -1.  The drift is
    then set by the smallest particles in the system, i.e. by the sink
    scale, not by m0.  Consequences: the mean-field closure is void, the
    drift becomes dm0/dtau ~ -C m0 (beta = 1, exponential decay), and the
    measured index locks onto -2 regardless of the kernel -- a spurious
    "universality" that is really a boundary effect.

    Physically the cure is the same one Dohnanyi (1969) built in: a
    catastrophic-disruption threshold.  A grain of dust does not shatter
    a boulder.  Requiring  m_small >= f * m_large  with f = O(0.1-1)
    restricts the integral to x in [f,1], which is scale free, restores
    locality, and recovers alpha = -(3+lambda)/2.

    Note that a CLOSED system is immune: its self-similar packet is
    narrow, so all partners are already of order m0 and the rule is local
    by construction.  This is why the closed runs agree with the theory
    even at f = 0, and the open ones do not.

[9] NO RESAMPLING, NO THINNING  --  and what that costs.

    This variant deliberately omits the weight machinery.  Every
    simulated particle stands for exactly one physical particle, w == 1
    is fixed, and therefore:

      + Sum(w) and Sum(w*m) are conserved to machine precision at every
        step; `mass_drift` is identically zero and any nonzero value is a
        genuine bug, not a statistical excursion.
      + There is no clock subtlety at all: dt = 2V/R with w = 1.
      + Nothing can bias the result, because nothing is resampled.

      - The RANGE is hard-capped.  In a closed box N = N_i m_i/m0, so
        keeping ~100 particles alive requires m0 <= N_i/100:
              N_i = 1e6  ->  4 decades
              N_i = 1e8  ->  6 decades (and ~3 GB of arrays)
        A coagulation run pushed past that ends with a handful of
        particles -- in a test at N_i = 2e5 it finished with ONE particle
        alive, and the usable plateau shrank from 5.6 decades to 2.7.
      - Fragmentation is worse in the other direction: N grows as
        m_i/m0, so three decades of cascade multiplies the population a
        thousandfold and memory, not physics, ends the run.

    Use this file when the priority is an unimpeachable baseline: exact
    conservation, no weights, no schemes to defend.  Use BF.py, which
    adds multiplication and down-sampling, when the priority is range --
    ten decades there cost 4.5e5 events and returned alpha = -1.005
    against a theoretical -1.000.  The two agree wherever both can run,
    which is the point of keeping both.

[10] SNAPSHOT PLACEMENT  --  the other half of reaching many decades.

    The closed-system observable is a Riemann sum
    F(m) = sum_k (dN/dm)(m,t_k) dt_k, and the sampling of t has TWO
    separate jobs which must not be confused:

      PLACEMENT decides what is RESOLVED.  For a given m the integrand
        peaks at m0 ~ m, so every decade of m needs snapshots at the
        matching decade of m0.
      WEIGHT decides what is COMPUTED.  It must be the actual dt between
        consecutive snapshots, whatever the placement.

    Three placements, all with the correct dt weight:

      'events'  -- events per unit m0 fall as 1/m0^2, so dt between
        event-uniform snapshots explodes.  Measured: 21 snapshots, the
        LAST one carrying 94% of the total weight; the sum degenerates
        into a single self-similar packet and alpha comes out anywhere
        from -1.4 to -2.6 instead of -1.  Worse, the plateau finder still
        reports a clean 1-decade plateau at -1.375 +- 0.14: an internally
        consistent, entirely wrong measurement.

      't_phys'  -- correct over a NARROW range (1.3 decades gave
        alpha = -0.965...-1.023 against -1.000), but fatal over a wide
        one: since t ~ m0^(1-lambda), uniform t is nearly uniform m0, so
        over ten decades with 2216 snapshots the first sample already
        sits at m0 = 5e5 and the lower FIVE decades get no sample at all.
        F(m) at small m is then assembled from the flat cores of far
        larger packets, n ~ m0^-2 Phi(m/m0 -> 0), which is independent of
        m: alpha collapses to ~0.  Measured: -0.04 +- 0.16.

      'log_m0'  -- snapshot whenever m0 has moved by 10^(1/k), giving k
        samples per decade of m0 everywhere.  Ten decades at k = 30 costs
        300 snapshots, seven times fewer than the failing uniform-t run,
        and places them where the integrand actually lives.  This is the
        placement to use for any run spanning more than ~2 decades.

    Note that resampling [9] and placement [10] cure DIFFERENT diseases:
    resampling buys the RANGE (without it N = N_i m_i/m0 dies at ~4
    decades), placement buys a CORRECT MEASUREMENT over that range.  A
    run with [9] and without [10] has flawless dynamics -- m0 agreed with
    the exact Smoluchowski solution to 1e-14 over ten decades -- and a
    meaningless spectrum.

----------------------------------------------------------------------
ANALYTIC PREDICTIONS THIS CODE IS MEANT TO TEST
----------------------------------------------------------------------
With kernel homogeneity  K(a m1, a m2) = a^lambda K(m1,m2),  and

        dm0/dtau  ~  m0^beta        (drift)
        m0        ~  tau^b,   b = 1/(1-beta)        (growth)
        F(m)      ~  m^alpha, alpha = -(1+beta)     (spectrum)

    CLOSED (packet amplitude ~ m0^-2, mass conserving):
        beta = lambda,          b = 1/(1-lambda),  alpha = -(1+lambda)

    OPEN (background amplitude ~ m0^alpha, self-consistent):
        beta = (1+lambda)/2,    b = 2/(1-lambda),  alpha = -(3+lambda)/2

    Fragmentation gives the SAME beta and the SAME alpha; only the sign
    of the drift flips, so m0 ~ (tau_* - tau)^b instead of tau^b.

    Both branches meet at lambda = 1 (beta = 1), where the drift becomes
    exponential and alpha = -2.  That point is simultaneously the
    gelation threshold and the only case in which closed and open agree.

    Constant kernel, lambda = 0 :  closed alpha = -1,   open alpha = -3/2
    Geometric kernel, lambda=2/3:  closed alpha = -5/3, open alpha = -11/6
----------------------------------------------------------------------
"""

import numpy as np
import json
import inspect
import re
from pathlib import Path
import time as _time


# ======================================================================
#  KERNELS  --  passed in as a parameter; lambda recorded for the theory
# ======================================================================

def kernel_constant(m1, m2):
    """lambda = 0."""
    return 1.0


def kernel_geometric(m1, m2):
    """Geometric cross section, sigma ~ (r1+r2)^2 with r ~ m^(1/3).  lambda = 2/3."""
    return (m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)) ** 2


def kernel_sum(m1, m2):
    """lambda = 1/3."""
    return m1 ** (1.0 / 3.0) + m2 ** (1.0 / 3.0)


def kernel_product(m1, m2):
    """lambda = 2  --  ABOVE the gelation threshold; the stationary cascade does not exist."""
    return m1 * m2


def kernel_additive(m1, m2):
    """lambda = 1  --  exactly the marginal / gelation-threshold kernel; drift is exponential."""
    return m1 + m2


#: homogeneity degree of each kernel, used only to state the prediction
KERNEL_LAMBDA = {
    "kernel_constant":  0.0,
    "kernel_geometric": 2.0 / 3.0,
    "kernel_sum":       1.0 / 3.0,
    "kernel_additive":  1.0,
    "kernel_product":   2.0,
}


# ======================================================================
#  ANALYTIC PREDICTION
# ======================================================================

def predict(system, lam):
    """
    Return the analytic prediction for a given configuration.

    Parameters
    ----------
    system : {'closed','open'}
    lam    : kernel homogeneity degree lambda

    Returns
    -------
    dict with
        beta  : drift exponent,      |dm0/dtau| ~ m0^beta
        b     : growth exponent,     m0 ~ tau^b   (or (tau_*-tau)^b)
        alpha : spectrum exponent,   F(m) ~ m^alpha
    Identical for coagulation and fragmentation -- only the sign of the
    drift differs, not the exponents.
    """
    if system == "closed":
        beta = lam                      # partner density ~ m0^-1  (mass conserving packet)
    elif system == "open":
        beta = (1.0 + lam) / 2.0        # partner density ~ m0^alpha (stationary background)
    else:
        raise ValueError("system must be 'closed' or 'open'")

    alpha = -(1.0 + beta)
    b = np.inf if abs(1.0 - beta) < 1e-12 else 1.0 / (1.0 - beta)
    return {"beta": beta, "b": b, "alpha": alpha, "lambda": lam, "system": system}


# ======================================================================
#  THE ENGINE
# ======================================================================

def simulate(
    *,
    process,                    # 'coagulation' | 'fragmentation'
    system,                     # 'closed'      | 'open'
    kernel,                     # callable K(m1,m2) >= 0, non-decreasing in both arguments
    edges,                      # 1D strictly increasing mass-bin edges
    ic,                         # {'m': m0, 'N': N0}
    # ---- open-system boundary conditions (ignored when system='closed') ----
    injection_rate=0.0,         # particles per unit physical time
    injection_mass=None,        # mass of an injected particle; default ic['m']
    sink_mass=None,             # ABSORBING boundary.  coagulation: remove products >= sink
                                #                     fragmentation: remove fragments <= sink
    # ---- passive limiter (closed fragmentation needs one to terminate) ----
    min_frag_mass=None,         # particles below this never fragment (mass stays conserved)
    frag_min_ratio=0.0,         # LOCALITY CONTROL, fragmentation only.  The heavier particle
                                # breaks only if the impactor satisfies  m_small >= f * m_large.
                                # f = 0 reproduces "any impactor shatters any target", which is
                                # NON-LOCAL against a power-law background (see note [8]) and
                                # drives the drift from the sink scale instead of from m0.
                                # Use f of order 0.1-1 for a scale-free (local) cascade.
    # ---- stopping ----
    max_time=np.inf,
    max_events=np.inf,
    stop_max_mass=None,         # stop when m_max >= this (closed coagulation)
    stop_min_mass=None,         # stop when m_mean <= this (closed fragmentation)
    max_stall_tries=2_000_000,  # stop if this many majorant attempts yield no event
                                # (happens once everything sits below min_frag_mass)
    # ---- snapshots ----
    snapshot_mode="events",     # 'events' | 't_phys' | 'log_m0'   (see note [10])
                                #   'log_m0' is the one to use beyond ~2 decades
    snapshot_stride=1e4,        # events / time units / SNAPSHOTS PER DECADE of m0
    snapshot_first_at_start=True,
    # ---- isochrones ----
    iso_age_edges=None,         # 1D age-bin edges; None disables
    iso_t_start=0.0,            # start accumulating isochrones only after this time
    age_rule="min",             # coagulation age inheritance: 'min'|'mass_weighted'|'heavier'
    # ---- misc ----
    weight=1.0,                 # CONSTANT weight per simulated particle (see notes [5ii], [9])
    volume=1.0,
    rng=None,
    verbose=True,
    verbose_every=1,
):
    """
    Event-driven majorant-frequency Monte Carlo for coagulation or
    fragmentation, in a closed or an open system.

    Returns a dict of arrays; see the bottom of this function.
    """

    # ------------------------------------------------------------------
    # 0.  Validate and set up
    # ------------------------------------------------------------------
    if process not in ("coagulation", "fragmentation"):
        raise ValueError("process must be 'coagulation' or 'fragmentation'")
    if system not in ("closed", "open"):
        raise ValueError("system must be 'closed' or 'open'")
    if age_rule not in ("min", "mass_weighted", "heavier"):
        raise ValueError("age_rule must be 'min', 'mass_weighted' or 'heavier'")

    if system == "closed":
        # A closed system has no source and no absorbing boundary: total mass
        # is exactly conserved.  min_frag_mass is a PASSIVE floor -- particles
        # below it simply stop fragmenting -- so mass conservation is untouched.
        injection_rate = 0.0
        sink_mass = None
    else:
        if injection_rate <= 0.0:
            raise ValueError("an open system needs injection_rate > 0")
        if sink_mass is None:
            raise ValueError("an open system needs an absorbing sink_mass")

    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing, len >= 2")
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])       # geometric (log) centres
    B = edges.size - 1

    m_init = float(ic["m"])
    N0 = int(ic["N"])
    if injection_mass is None:
        injection_mass = m_init

    w = float(weight)                               # physical particles per sim particle
    V = float(volume)

    if rng is None:
        rng = np.random.default_rng()
    rnd = rng.random
    rint = rng.integers

    # ------------------------------------------------------------------
    # 1.  Particle arrays.  The live prefix [0:live] is the active set.
    # ------------------------------------------------------------------
    cap = max(N0, 1024)
    mass = np.empty(cap, dtype=np.float64)
    inj_time = np.empty(cap, dtype=np.float64)
    bin_of = np.empty(cap, dtype=np.int32)
    pos_in_bin = np.empty(cap, dtype=np.int32)

    mass[:N0] = m_init
    inj_time[:N0] = 0.0
    live = N0

    bins = [[] for _ in range(B)]                   # bin -> list of particle indices

    def ensure_capacity(need):
        """Grow all per-particle arrays together, keeping the live prefix."""
        nonlocal mass, inj_time, bin_of, pos_in_bin
        if need <= mass.size:
            return
        new_cap = max(need, int(mass.size * 1.6) + 1)
        for name in ("mass", "inj_time", "bin_of", "pos_in_bin"):
            old = locals_ref[name]
            new = np.empty(new_cap, dtype=old.dtype)
            new[:live] = old[:live]
            locals_ref[name] = new
        mass = locals_ref["mass"]
        inj_time = locals_ref["inj_time"]
        bin_of = locals_ref["bin_of"]
        pos_in_bin = locals_ref["pos_in_bin"]

    locals_ref = {"mass": mass, "inj_time": inj_time,
                  "bin_of": bin_of, "pos_in_bin": pos_in_bin}

    # ------------------------------------------------------------------
    # 2.  Bin index, and O(1) bin membership  [trick 3]
    # ------------------------------------------------------------------
    def mass_to_bin(m):
        b = int(np.searchsorted(edges, m, side="right") - 1)
        return 0 if b < 0 else (B - 1 if b >= B else b)

    def bin_add(i, b):
        bin_of[i] = b
        pos_in_bin[i] = len(bins[b])
        bins[b].append(i)

    def bin_remove(i):
        b = int(bin_of[i]); p = int(pos_in_bin[i])
        last = bins[b][-1]
        bins[b][p] = last
        pos_in_bin[last] = p
        bins[b].pop()

    for i in range(live):
        bin_add(i, mass_to_bin(mass[i]))

    N_bin = np.array([len(bins[b]) for b in range(B)], dtype=np.float64)

    # ------------------------------------------------------------------
    # 3.  Static bin-pair majorant table  [trick 2]
    #     fmaj[b,c] = K(upper_b, upper_c) bounds K on the whole bin pair,
    #     PROVIDED K is non-decreasing in both arguments.  Violations are
    #     checked at run time rather than assumed.
    # ------------------------------------------------------------------
    upper = edges[1:]
    fmaj = np.empty((B, B), dtype=np.float64)
    for b in range(B):
        for c in range(B):
            val = float(kernel(float(upper[b]), float(upper[c])))
            if val < 0.0:
                raise ValueError("kernel must be non-negative")
            fmaj[b, c] = val
    diag = np.diag(fmaj).copy()

    # ------------------------------------------------------------------
    # 4.  Rate bookkeeping  [trick 4]
    #     T[b]        = sum_c fmaj[b,c] N_c
    #     rate_row[b] = N_b (T[b] - fmaj[b,b])       (ordered pairs, i != j)
    #     R           = sum_b rate_row[b]  = sum_{i != j} fmaj_ij
    #                                      = 2 * sum_{i<j} fmaj_ij      [note 5i]
    # ------------------------------------------------------------------
    T = fmaj @ N_bin
    rate_row = N_bin * (T - diag)
    R = float(rate_row.sum())

    def apply_delta(b, dN):
        """One bin count changed by dN: update T, rate_row and R in O(B), vectorised."""
        nonlocal R
        if dN == 0:
            return
        N_bin[b] += dN
        T[:] += dN * fmaj[:, b]
        np.multiply(N_bin, T - diag, out=rate_row)
        R = float(rate_row.sum())

    # --- vectorised selection: cumsum + searchsorted instead of a python loop ---
    def sample_row():
        cum = np.cumsum(rate_row)
        return int(np.searchsorted(cum, rnd() * cum[-1], side="right"))

    def sample_col(b):
        q = N_bin * fmaj[b]
        q[b] = (N_bin[b] - 1.0) * fmaj[b, b]        # exclude the self-pair
        s = q.sum()
        if s <= 0.0:
            return None
        cum = np.cumsum(q)
        return int(np.searchsorted(cum, rnd() * s, side="right"))

    # ------------------------------------------------------------------
    # 5.  Particle removal / relocation
    # ------------------------------------------------------------------
    def delete_particle(j):
        """Remove particle j: drop from its bin, then swap-with-last in the arrays."""
        nonlocal live
        b_j = int(bin_of[j])
        bin_remove(j)
        apply_delta(b_j, -1)
        last = live - 1
        if j != last:
            mass[j] = mass[last]
            inj_time[j] = inj_time[last]
            b_last = int(bin_of[last]); p_last = int(pos_in_bin[last])
            bins[b_last][p_last] = j
            bin_of[j] = b_last
            pos_in_bin[j] = p_last
        live -= 1

    def move_bin(i, b_new):
        b_old = int(bin_of[i])
        if b_new == b_old:
            return
        bin_remove(i); apply_delta(b_old, -1)
        bin_add(i, b_new); apply_delta(b_new, +1)

    def add_particle(m, t_born):
        """Append one particle; returns its index."""
        nonlocal live
        ensure_capacity(live + 1)
        i = live
        mass[i] = m
        inj_time[i] = t_born
        b = mass_to_bin(m)
        bin_add(i, b)
        apply_delta(b, +1)
        live += 1
        return i


    # ------------------------------------------------------------------
    # 6.  Age inheritance for coagulation  [trick 7]
    # ------------------------------------------------------------------
    def inherit_age(t1, t2, m1, m2):
        if age_rule == "min":
            return min(t1, t2)                       # oldest ancestor wins
        if age_rule == "heavier":
            return t1 if m1 >= m2 else t2
        return (m1 * t1 + m2 * t2) / (m1 + m2)       # mass-weighted

    # ------------------------------------------------------------------
    # 7.  State, counters, snapshot storage
    # ------------------------------------------------------------------
    t_phys = 0.0
    events = 0                 # accepted events
    tries = 0                  # majorant attempts
    tries_since_event = 0      # stall guard: see stop condition below
    inj_residual = 0.0         # [trick 6]
    M_sys = w * N0 * m_init    # physical mass currently in the system
    M_in = 0.0
    M_out = 0.0
    m_max = m_init
    m_min = m_init
    sink_events = 0
    M_ref = M_sys                        # reference mass; drift must stay identically zero

    rows = {k: [] for k in ("t", "events", "live", "N_phys", "m_mean", "m_max",
                            "m_min", "M_sys", "M_in", "M_out", "weight", "dndm")}

    use_iso = iso_age_edges is not None
    if use_iso:
        iso_age_edges = np.asarray(iso_age_edges, dtype=float)
        iso_counts = np.zeros((iso_age_edges.size - 1, B), dtype=np.float64)
        iso_snaps = 0
    else:
        iso_counts = None
        iso_snaps = 0

    def snapshot():
        nonlocal iso_snaps
        N_phys = w * live
        rows["t"].append(t_phys)
        rows["events"].append(events)
        rows["live"].append(live)
        rows["N_phys"].append(N_phys)
        rows["m_mean"].append(M_sys / N_phys if N_phys > 0 else np.nan)
        rows["m_max"].append(m_max)
        rows["m_min"].append(m_min)
        rows["M_sys"].append(M_sys)
        rows["M_in"].append(M_in)
        rows["M_out"].append(M_out)
        rows["weight"].append(w)
        # dN/dm in PHYSICAL units: counts * w / bin width / volume
        rows["dndm"].append(N_bin * (w / V) / widths)

        if use_iso and t_phys >= iso_t_start and live > 0:
            ages = t_phys - inj_time[:live]
            H, _, _ = np.histogram2d(ages, mass[:live], bins=[iso_age_edges, edges])
            iso_counts[:] += H * w
            iso_snaps += 1

    if snapshot_first_at_start:
        snapshot()

    if snapshot_mode not in ("events", "t_phys", "log_m0"):
        raise ValueError("snapshot_mode must be 'events', 't_phys' or 'log_m0'")
    # log_m0: fire whenever m0 has moved by a factor 10^(1/snapshot_stride)
    log_step = 1.0 / float(snapshot_stride) if snapshot_mode == "log_m0" else None
    m0_last  = (M_sys / (w * live)) if live > 0 else m_init
    next_thr = (events if snapshot_mode == "events" else t_phys) + float(snapshot_stride)

    t_wall0 = _time.perf_counter()
    n_snap_printed = 0

    # ==================================================================
    # 8.  MAIN LOOP
    # ==================================================================
    while True:
        # ---- stopping conditions -------------------------------------
        if live < 2 or R <= 0.0:
            stop_reason = "population exhausted"
            break
        if t_phys >= max_time:
            stop_reason = "max_time"
            break
        if events >= max_events:
            stop_reason = "max_events"
            break
        if stop_max_mass is not None and m_max >= stop_max_mass:
            stop_reason = "stop_max_mass"
            break
        if stop_min_mass is not None and rows["m_mean"] and rows["m_mean"][-1] <= stop_min_mass:
            stop_reason = "stop_min_mass"
            break
        # Stall guard.  A passive floor (min_frag_mass) or a collapsed acceptance can
        # leave the majorant firing while no event is ever accepted; without this the
        # loop spins forever, since max_events is then never reached.
        if tries_since_event > max_stall_tries:
            stop_reason = "acceptance collapsed (stall guard)"
            break

        # ---- THE CLOCK  [note 5] -------------------------------------
        # R counts ORDERED pairs, hence the factor 2; w converts the
        # simulated sub-volume to the physical one.  Quadratic in N by
        # construction, because R ~ N^2.
        dt = 2.0 * V / (w * R)
        t_phys += dt
        tries += 1
        tries_since_event += 1

        # ---- deterministic injection  [trick 6] ----------------------
        if injection_rate > 0.0:
            inj_residual += injection_rate * dt / w      # in SIMULATED particles
            k = int(inj_residual)
            if k > 0:
                inj_residual -= k
                ensure_capacity(live + k)
                for _ in range(k):
                    add_particle(float(injection_mass), t_phys)
                M_sys += w * k * injection_mass
                M_in += w * k * injection_mass
                if injection_mass > m_max:
                    m_max = float(injection_mass)
                if injection_mass < m_min:
                    m_min = float(injection_mass)

        # ---- pick a bin pair, then two particles ---------------------
        b = sample_row()
        c = sample_col(b)
        if c is None:
            continue
        if b != c:
            ib, jc = bins[b], bins[c]
            if not ib or not jc:
                continue
            i = ib[rint(0, len(ib))]
            j = jc[rint(0, len(jc))]
        else:
            ib = bins[b]
            if len(ib) < 2:
                continue
            p1 = rint(0, len(ib))
            p2 = rint(0, len(ib) - 1)
            if p2 >= p1:
                p2 += 1
            i, j = ib[p1], ib[p2]

        # ---- thin against the bin-pair majorant  [trick 1] -----------
        m1 = float(mass[i]); m2 = float(mass[j])
        f_now = float(kernel(m1, m2))
        f_cap = float(fmaj[b, c])
        if f_now > f_cap * (1.0 + 1e-12):
            raise RuntimeError(
                f"Majorant violated: K({m1:.4g},{m2:.4g})={f_now:.4g} > "
                f"fmaj[{b},{c}]={f_cap:.4g}.  The corner bound assumes a kernel "
                f"non-decreasing in both arguments."
            )
        if f_cap <= 0.0 or rnd() >= f_now / f_cap:
            continue                                     # null (thinned) event

        # ==============================================================
        #  ACCEPTED EVENT -- the only place the four cases differ
        # ==============================================================
        if process == "coagulation":
            mn = m1 + m2
            inj_time[i] = inherit_age(inj_time[i], inj_time[j], m1, m2)
            mass[i] = mn
            move_bin(i, mass_to_bin(mn))
            delete_particle(j)
            events += 1; tries_since_event = 0
            if mn > m_max:
                m_max = mn
            # absorbing sink at LARGE mass (open systems only)
            if sink_mass is not None and mn >= sink_mass:
                # index i may have been relocated by delete_particle(j)
                i_now = i if i < live else j
                delete_particle(i_now)
                M_sys -= w * mn
                M_out += w * mn
                sink_events += 1

        else:  # ---------------- fragmentation ----------------------
            # the more massive of the pair breaks; the other is untouched
            ip = i if m1 >= m2 else j
            mp = max(m1, m2)
            ms = min(m1, m2)
            if min_frag_mass is not None and mp < min_frag_mass:
                continue                                 # passive floor: no event
            if frag_min_ratio > 0.0 and ms < frag_min_ratio * mp:
                continue                                 # impactor too small to disrupt [note 8]
            xi = rnd()
            ma, mb = xi * mp, (1.0 - xi) * mp
            t_born = float(inj_time[ip])                 # unambiguous: one parent
            mass[ip] = ma
            move_bin(ip, mass_to_bin(ma))
            i_new = add_particle(mb, t_born)
            events += 1; tries_since_event = 0
            if ma < m_min: m_min = ma
            if mb < m_min: m_min = mb
            # absorbing sink at SMALL mass (open systems only).
            # Delete the higher index first so the swap-with-last cannot
            # invalidate the other index.
            if sink_mass is not None:
                doomed = []
                if mb <= sink_mass: doomed.append((i_new, mb))
                if ma <= sink_mass: doomed.append((ip, ma))
                for idx, mm in sorted(doomed, key=lambda z: -z[0]):
                    delete_particle(idx)
                    M_sys -= w * mm
                    M_out += w * mm
                    sink_events += 1

        # ---- snapshots ----------------------------------------------
        if snapshot_mode == "log_m0":
            # PLACEMENT: uniform in log m0, so every decade is resolved [note 10].
            # The WEIGHT used later is still the actual dt between snapshots.
            m0_now = (M_sys / (w * live)) if live > 0 else np.nan
            fire = (np.isfinite(m0_now) and m0_now > 0
                    and abs(np.log10(m0_now / m0_last)) >= log_step)
            if fire:
                m0_last = m0_now
            progress = 1.0 if fire else 0.0
            next_thr = 1.0
        else:
            progress = events if snapshot_mode == "events" else t_phys
        while progress >= next_thr:
            snapshot()
            if snapshot_mode == "log_m0":
                n_snap_printed += 1
                break
            next_thr += float(snapshot_stride)
            n_snap_printed += 1
            if verbose and (n_snap_printed % verbose_every == 0):
                print(f"[{_time.strftime('%H:%M:%S')}] "
                      f"t={t_phys:.6g} | events={events:.3e} | live={live} | "
                      f"<m>={rows['m_mean'][-1]:.4g} | m_max={m_max:.3g} | "
                      f"sink={sink_events} | acc={events/max(tries,1):.3f} | "
                      f"cpu={_time.perf_counter()-t_wall0:.1f}s")
            progress = events if snapshot_mode == "events" else t_phys
    else:
        stop_reason = "loop exit"

    snapshot()   # always record the final state

    # ------------------------------------------------------------------
    # 9.  Pack results
    # ------------------------------------------------------------------
    out = {k: np.asarray(v, dtype=float) for k, v in rows.items() if k != "dndm"}
    out["dndm"] = np.vstack(rows["dndm"]) if rows["dndm"] else np.zeros((0, B))
    out["centers"] = centers
    out["edges"] = edges
    out["widths"] = widths
    out["final_mass"] = mass[:live].copy()
    out["final_inj_time"] = inj_time[:live].copy()
    out["final_t_phys"] = float(t_phys)
    out["tries"] = float(tries)
    out["acceptance"] = float(events / max(tries, 1))
    out["stop_reason"] = stop_reason
    out["final_weight"] = float(w)          # constant by construction
    # with no resampling this is exactly zero; anything else is a bug
    out["mass_drift"] = float((M_sys - M_in + M_out - M_ref) / max(M_ref, 1e-300))
    if use_iso:
        out["iso_age_edges"] = iso_age_edges
        out["iso_counts"] = iso_counts
        out["iso_dndm"] = iso_counts / widths[None, :]
        out["iso_snapshots"] = float(iso_snaps)

    out["meta"] = {
        "process": process,
        "system": system,
        "kernel": getattr(kernel, "__name__", "kernel"),
        "lambda": KERNEL_LAMBDA.get(getattr(kernel, "__name__", ""), None),
        "ic": {"m": m_init, "N": N0},
        "injection_rate": float(injection_rate),
        "injection_mass": float(injection_mass),
        "sink_mass": None if sink_mass is None else float(sink_mass),
        "min_frag_mass": None if min_frag_mass is None else float(min_frag_mass),
        "frag_min_ratio": float(frag_min_ratio),
        "age_rule": age_rule,
        "weight": w,
        "volume": V,
        "clock": "dt = 2*V/(w*R), R over ordered pairs",
        "resampling": "none - baseline variant, w constant",
        "injection_scheme": "deterministic residual (no Poisson noise)",
        "stop_reason": stop_reason,
        "max_stall_tries": int(max_stall_tries),
    }
    return out


# ======================================================================
#  ANALYSIS HELPERS
# ======================================================================

def superpose(dndm, times):
    """
    Age-integrated spectrum  F(m) = sum_t (dN/dm)(m,t) * dt.

    This is the correct estimator for a CLOSED system, where no
    stationary state exists and the observable spectrum is the
    superposition of independently evolving parcels with a stationary
    age distribution.  For an OPEN system the steady state is the
    instantaneous spectrum and this is NOT what you want -- use the
    last snapshot instead.
    """
    D = np.asarray(dndm, float)
    t = np.asarray(times, float).ravel()
    dt = np.diff(t, prepend=t[0])
    return (D * dt[:, None]).sum(axis=0)


def fit_powerlaw(x, y, xmin, xmax):
    """Least-squares fit y ~ A x^alpha over [xmin,xmax]; returns dict."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x >= xmin) & (x <= xmax) & np.isfinite(x) & np.isfinite(y) & (y > 0)
    if m.sum() < 3:
        return {"alpha": np.nan, "A": np.nan, "r2": np.nan, "n": int(m.sum())}
    lx, ly = np.log10(x[m]), np.log10(y[m])
    s, c = np.polyfit(lx, ly, 1)
    pred = c + s * lx
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {"alpha": float(s), "A": float(10 ** c),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "n": int(m.sum())}


def local_slope(m, F, half=3, min_val=0.0):
    """
    Local logarithmic slope  Gamma(m) = d log F / d log m,  computed by least
    squares on a sliding window of (2*half+1) bins.

    This is the honest way to look at a measured spectrum: a genuine inertial
    range shows up as a PLATEAU in Gamma, and the contaminated ends show up as
    the places where Gamma bends.  Fitting a single power law across the whole
    array averages the plateau together with the bends and returns a number
    that describes neither.  (The same diagnostic is used as the lower subpanel
    of every spectrum figure in Laor & Gitelman, Phys. Rev. E 113, 044135.)

    Returns (m_valid, Gamma) with NaN where the window is not usable.
    """
    m = np.asarray(m, float); F = np.asarray(F, float)
    ok = np.isfinite(m) & np.isfinite(F) & (F > min_val) & (m > 0)
    lm, lF = np.log10(m), np.log10(np.where(ok, F, np.nan))
    G = np.full(m.size, np.nan)
    for i in range(m.size):
        a, b = max(0, i - half), min(m.size, i + half + 1)
        w = ok[a:b]
        if w.sum() >= max(4, half + 1):
            G[i] = np.polyfit(lm[a:b][w], lF[a:b][w], 1)[0]
    return m, G


def find_inertial_range(m, F, half=4, tol=0.30, min_decades=0.8, min_val=0.0):
    """
    Locate the inertial range automatically as the LONGEST contiguous run of
    bins over which the local slope stays within `tol` of that run's own mean.

    Parameters
    ----------
    tol          : allowed scatter of Gamma inside the plateau (dex per dex)
    min_decades  : reject a plateau narrower than this (a two-bin "plateau" is
                   not a power law)

    Returns dict(m_lo, m_hi, alpha, scatter, decades, n_bins) -- or NaNs if no
    plateau qualifies, which is itself the correct answer when the run has no
    inertial range yet.
    """
    m, G = local_slope(m, F, half=half, min_val=min_val)
    good = np.isfinite(G)
    best = None
    i = 0
    n = m.size
    while i < n:
        if not good[i]:
            i += 1; continue
        j = i
        while j + 1 < n and good[j + 1]:
            seg = G[i:j + 2]
            if seg.max() - seg.min() > 2 * tol:
                break
            j += 1
        seg = G[i:j + 1]
        if seg.size >= 3:
            dec = np.log10(m[j] / m[i])
            if dec >= min_decades and (best is None or dec > best[0]):
                best = (dec, i, j)
        i = j + 1
    if best is None:
        return {"m_lo": np.nan, "m_hi": np.nan, "alpha": np.nan,
                "scatter": np.nan, "decades": 0.0, "n_bins": 0}
    dec, i, j = best
    seg = G[i:j + 1]
    return {"m_lo": float(m[i]), "m_hi": float(m[j]),
            "alpha": float(np.mean(seg)), "scatter": float(np.std(seg)),
            "decades": float(dec), "n_bins": int(j - i + 1)}


def guard_band(m_low_scale, m_high_scale, pad_decades=0.5):
    """
    The a-priori inertial range: strip `pad_decades` from each end of the
    interval between the two characteristic masses of the problem.

        closed system : (m_inj , max m0 reached)
        open system   : (m_inj , m_sink)

    Use it as an independent cross-check on `find_inertial_range`: the two
    should agree.  If they do not, the run has not developed a cascade over
    the range you assumed.
    """
    lo = m_low_scale * 10.0 ** pad_decades
    hi = m_high_scale / 10.0 ** pad_decades
    return (lo, hi) if hi > lo else (np.nan, np.nan)


def iso_mean_mass(iso_counts, centers, age_edges):
    """
    Mean mass of each isochrone,  <m>(tau), from the accumulated 2D
    (age, mass) histogram.  This is the direct measurement of the
    growth law  m0(tau)  in an OPEN system.
    """
    C = np.asarray(iso_counts, float)
    n = C.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mbar = (C * centers[None, :]).sum(axis=1) / n
    tau = np.sqrt(np.asarray(age_edges)[:-1] * np.asarray(age_edges)[1:])
    return tau, mbar, n


def growth_fit(t, m0, t_star=None, mask=None):
    """
    Fit the growth (or decay) law.

    Coagulation:    m0 ~ tau^b            -> pass t_star=None
    Fragmentation:  m0 ~ (t_star - t)^b   -> pass t_star (e.g. the final time)

    Returns dict with b, r2, and the R^2 of an exponential fit for
    comparison -- if the exponential wins, the clock is wrong.
    """
    t = np.asarray(t, float); m0 = np.asarray(m0, float)
    if mask is None:
        mask = np.isfinite(t) & np.isfinite(m0) & (m0 > 0)
    x = (t_star - t[mask]) if t_star is not None else t[mask]
    y = m0[mask]
    good = x > 0
    x, y = x[good], y[good]
    if x.size < 4:
        return {"b": np.nan, "r2_power": np.nan, "r2_exp": np.nan, "n": int(x.size)}
    b, c = np.polyfit(np.log(x), np.log(y), 1)
    r2p = float(np.corrcoef(np.log(x), np.log(y))[0, 1] ** 2)
    r2e = float(np.corrcoef(x, np.log(y))[0, 1] ** 2)
    return {"b": float(b), "A": float(np.exp(c)),
            "r2_power": r2p, "r2_exp": r2e, "n": int(x.size)}


# ---------------------------------------------------------------------
#  Exact reference solutions for the constant kernel (lambda = 0),
#  closed system.  Used to validate the clock -- see note [5].
# ---------------------------------------------------------------------

def exact_closed_constant(t, m_init, n_init, K1=1.0, process="coagulation"):
    """
    Smoluchowski with K = K1 = const in a closed box.

        coagulation:    n(t) = n0 / (1 + n0 K1 t / 2),  m0 = m_init (1 + n0 K1 t/2)
        fragmentation:  n(t) = n0 / (1 - n0 K1 t / 2),  m0 = m_init (1 - n0 K1 t/2)

    In BOTH cases m0(t) is a straight line on LINEAR axes.  If a run
    instead gives a straight line for ln m0 versus t, the particle
    weights are missing from the time step.
    """
    t = np.asarray(t, float)
    s = 0.5 * n_init * K1 * t
    return m_init * (1.0 + s) if process == "coagulation" else m_init * (1.0 - s)


# ======================================================================
#  I/O
# ======================================================================

def _kernel_tag(f):
    name = getattr(f, "__name__", "kernel")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def save_run(path, out):
    """Save a run to .npz; metadata goes in as a JSON blob."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: np.asarray(v) for k, v in out.items()
         if k != "meta" and not isinstance(v, str)}
    d["meta_json"] = np.frombuffer(
        json.dumps(out["meta"], ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **d)
    return str(path)


def load_run(path):
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            if k == "meta_json":
                out["meta"] = json.loads(bytes(z[k]).decode("utf-8"))
            else:
                out[k] = z[k]
    return out