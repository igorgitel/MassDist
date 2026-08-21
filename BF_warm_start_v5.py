"""
BF_warm_start_v5.py  --  Majorant_v2

v5 = v4 + a WARM START.  Three additions, nothing else:

  (a) SEEDED INITIAL CONDITION.  ic={'steady': {...}} draws the initial
      population from a power law instead of a delta at m_inj, so the run
      does not have to spend one full residence time building the cascade
      before any measurement is honest.  See note [12].

  (b) AN INHERITED HONESTY MARK.  Seeded bodies carry gen = -1, and -1 is
      contagious: any descendant of a seeded body, through any number of
      splits or mergers, stays -1 and is excluded from the generation
      statistics.  Only bodies injected after t = 0, and their descendants,
      have an honest split count.  See note [13].

  (c) THE WAITING LAW.  <dt>(m), the mean time between successive
      mass-changing events for a particle sitting at mass m, accumulated
      per mass bin.  <dt> ~ m^(1/b), and unlike tau(g) it is MEMORYLESS --
      it needs no history, so seeded particles contribute to it from the
      first event.  This is the point of the whole version: an estimate of
      b that does not go through alpha, where db/dalpha = b^2 = 36 destroys
      the precision.  See note [14].

With ic={'m':..,'N':..} the dynamics are bit-identical to v4 at the same
seed: no random number is drawn that v4 did not draw, in the same order.

ONE event-driven Monte Carlo engine for the mass-space cascade, covering
all four configurations with the same code path:

        process = 'coagulation' | 'fragmentation'
        system  = 'closed'      | 'open'

The physics that differs between the four cases enters only through
(a) what an accepted event does to the two selected particles, and
(b) the boundary conditions (injection + absorbing sink, or none).
Everything else -- majorant construction, pair selection, the clock,
the bookkeeping -- is shared.

BASELINE VARIANT: NO RESAMPLING, NO THINNING.  One simulated particle is
one physical particle, w == 1 for the whole run, so Sum(w) and Sum(w*m)
are conserved to machine precision and `mass_drift` is identically zero.
That is what *conservative* means in the names of the runs this engine
writes.  What it costs is range -- see note [9].

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
    from the bin upper corners.  What that buys is COST and a CHECKABLE
    BOUND -- not correctness, because the majorant cancels out of the
    physical rate exactly:

        (w R / 2V) * (fmaj_ij / R) * (K_ij / fmaj_ij)  =  w K_ij / (2V),

    i.e. ticks per unit time, times the chance the tick picked that pair,
    times the acceptance.  fmaj cancels twice, once in the selection
    weight and once in the acceptance, and summing over unordered pairs
    returns the Smoluchowski rate.  ANY valid majorant gives the same
    physics; a bad one only wastes trials.  This holds because the clock
    is advanced on every TRIAL, not on every event -- see note [5iv].

    A single global f_max = K(m_max,m_max) is therefore legitimate and
    unbiased.  It is still the wrong choice, for three reasons:
      - acceptance.  A typical pair is accepted with probability
        <K>/f_max ~ (m0/m_max)^lambda, i.e. 10^(-lambda D) across D
        decades: for the geometric kernel over four decades that is
        2e-3, some five hundred trials per event against 1.05 with the
        table, whose corner overshoot is only 10^(0.1 lambda) per
        argument at 0.1-dex bins.
      - the stall guard misfires.  max_stall_tries counts trials without
        an accepted event; at an acceptance of 2e-3 its default 2e6 is
        only ~4000 events of honest work, so a healthy run dies with the
        diagnosis "acceptance collapsed".
      - a LAGGING maximum is a genuine bias.  A running global maximum
        not refreshed before some particle grows past it gives fmaj < K,
        and the thinning is then silently wrong.  The corner table is
        static and every trial checks K <= fmaj explicitly, so the same
        situation raises instead of biasing.

    For the binned / efficient version of this construction see
      - Eibeck & Wagner, SIAM J. Sci. Comput. 22, 802 (2000)
        doi:10.1137/S1064827599353488

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
         so that a constant w != 1 simply rescales the density and the
         formula stays identical to the weighted variant of this engine.
         Nothing in this file ever changes w, so the clock cannot be
         corrupted by weight bookkeeping -- which is the whole reason
         this baseline exists.

    (iii) dt must be QUADRATIC in the particle number, because the
          process is binary.  A step linear in N describes a unary
          process and gives the wrong N(t).

    (iv) TIME ADVANCES ON EVERY TRIAL, not on every event.  t_phys is
         incremented right after dt is formed, some fifty lines before
         the acceptance test, so null (thinned) trials move the clock
         too.  That is exactly what makes the cancellation in note [2]
         work, and moving the increment inside the accepted branch is
         the one edit that turns the majorant from a cost into a BIAS:
         time would then advance by 2V/(wR) per accepted event, the
         reported rate would be R instead of R*<K/fmaj> = sum K, and
         with a global f_max one would get dt_eff ~ 1/(N^2 m_max^lambda)
         -- the kernel entering through the EXTREME of the distribution
         instead of through the typical mass.  In an open system m_max
         is pinned at the sink and does not grow, so the m0 dependence
         leaves the clock entirely, the drift degenerates to
         dm0/dt ~ m0 (exponential, beta = 1), and the constant-flux
         argument then returns alpha = -2 for EVERY kernel.  That is a
         third, purely numerical route to the same -2 fingerprint that
         note [8] produces physically.

[6] Deterministic (residual) injection.
    Injection uses a fractional accumulator rather than a Poisson draw,
    which removes shot noise at the injection scale.  This is a
    modelling choice and is stated in the output metadata.

[7] AGE AND TRAJECTORY BOOKKEEPING.

    Two mechanisms sit side by side here.  Ages label an ENSEMBLE; tracers follow a
    TRAJECTORY.  They answer different questions and are read differently.

    ---- AGES -------------------------------------------------------------------

    Every particle carries `inj_time`, the moment its own clock was last set; the
    histogram bins tau = t_phys - inj_time.  Accumulation does not begin until the
    sink has swallowed `iso_start_sink` physical particles (default 10) -- the one
    steady-state statement that needs no retuning when N_ss, the kernel or the mass
    range changes.  A CLOSED system has no sink, so the default gate is silently
    disarmed there; only an explicitly passed value still raises.

    COAGULATION.  A merged particle has two ancestors, so an inheritance rule is
    needed: `age_rule` ('min'|'max'|'mass_weighted'|'heavier').  Clocks are set at
    injection only, always on a particle of mass m_inj, so every age class starts
    from the same mass and <m>(tau) is well posed.  Report the rule and test its
    sensitivity -- 'min' and 'heavier' weight the tail of the merger tree
    differently.

    FRAGMENTATION.  A fragment has one parent, so nothing is ambiguous -- but that
    is not the same as nothing being chosen.  `frag_age_rule` picks which piece
    keeps the parent's clock:

        'inherit'   both do.  No clock is ever reset, so an age class is the whole
                    descendant tree of one injected body -- a FAMILY.  Diagnostic:
                    the median age equals the maximum age, because essentially
                    every live particle descends from t = 0.
        'heavier'   heavier keeps it, lighter is reborn at t_phys.
        'lighter'   mirror image; a sensitivity test, not physics.
        'both_new'  both reborn; removes the age axis on purpose, so a surviving
                    <m>(tau) trend is a bug.

    Read an age class as an ENSEMBLE, not as a path.  Splitting multiplies carriers,
    so a class collects the whole tree at once and its width in log m is set by the
    spread of birth masses within it, not by tau.  That is what tracers are for.

    ---- GENERATIONS -------------------------------------------------------------

    [v4] Every particle carries `gen`, the number of splits between it and the body
    that ENTERED the system, and `gen_time`, the moment that body entered.  A snapshot
    bins the live population by (gen, mass) exactly as the isochrones bin it by
    (age, mass).

    WHY THIS REPLACES THE TRACERS.  What a tracer chain was ever for is the number of
    splits along a path: x = ln(m/m0) is a sum of ln(xi) over splits, so <x> and Var(x)
    are linear in that number with slopes that are pure numbers, and everything about
    the trajectory follows.  But that number does not need a tagged path -- it is a
    property of the PARTICLE and it fits in one int32.  So it is carried by every
    particle instead of by a few thousand tagged ones, and the statistics improve by
    four orders of magnitude for 16 MB at N = 4e6.

    Three problems disappear with the tracers rather than being solved:
      - the number of independent TREES, which no tracer setting could raise above
        n_out/R (tens at six decades), stops being the limit: a generation histogram is
        built from the whole population;
      - the shelf at m_inj, which held 92% of the tracer dwell time in a measured run,
        no longer decides what is sampled -- a particle's generation does not care how
        long it waited;
      - censoring by the sink is gone, because a particle is counted while it is alive
        and nothing has to survive a full descent to contribute.

    AND THE CLOCK COMES FREE.  Accumulating the age alongside the count gives <tau>(g),
    the mean time to reach generation g over the whole population.  That is the curve a
    tracer log gave as <tau>(k), and it carries b the same way:

        dg/dtau ~ nu(m) ~ m^(beta-1),  <ln m> falls by <ln xi> per split, so
        tau(g) = T (1 - exp(-g/(2b)))   with <ln xi> = -1/2 at w = 1/2.

    WHAT IS LOST.  A generation histogram is an ENSEMBLE at fixed g, not a path: it
    cannot say which particle came from which, so correlations BETWEEN successive steps
    of one trajectory are not measurable from it.  For the multiplicative picture that
    is no loss, because independence of the steps is exactly what makes <x> and Var(x)
    linear in g -- and that linearity IS the test.

    THE SPLIT IS SIZE-BIASED ONLY ALONG A PATH, NOT HERE.  A tracer followed a fragment
    with probability xi and therefore saw <ln xi> = -1/2.  A generation histogram counts
    BOTH fragments, so the number-weighted step applies: <ln xi> = -1 and Var = 1 for a
    uniform split.  Same tree, different weighting -- use the right pair of numbers.
    Weighting the histogram by mass recovers -1/2 and 1/4.

    ASYMMETRY.  Whenever the followed piece can be arbitrarily small, ln of it has an
    exponential tail and the skewness of one step is exactly -2, whatever the rule --
    which is why "random piece", "always lighter" and "the tracer" all measured -1.99.
    Only "always the heavier" is bounded, xi in [1/2,1], and comes out nearly symmetric
    at -0.24.  Skewness after g steps falls as -2/sqrt(g), so it is a small-g transient
    and not an obstruction; at g = 28 it is -0.38.

    INHERITANCE.  Fragmentation: both fragments take gen+1 and keep gen_time, so
    tau = t - gen_time is the time since the ANCESTOR entered, whatever frag_age_rule
    does to inj_time (which still serves the isochrones and is a separate field on
    purpose).  Coagulation: the product takes max(gen_i, gen_j) + 1 and the older
    gen_time.  Injection and the initial condition: gen = 0, gen_time = now.

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

    This note and note [7] are DIFFERENT diseases and must not be
    conflated: f is about which collisions are allowed to break a body,
    frag_age_rule is about which piece keeps the clock afterwards.  f
    changes the dynamics and moves alpha; the age rule changes nothing
    but the label on a particle and moves only b.

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
    conservation, no weights, no schemes to defend.  Use the weighted
    engine, which adds multiplication and down-sampling, when the
    priority is range.

    HISTORICAL MEASUREMENTS, recorded here so they are not mistaken for
    claims about the present code: on the weighted engine ten decades
    cost 4.5e5 events and returned alpha = -1.005 against a theoretical
    -1.000, and the two engines agreed wherever both could run.  That
    comparison has not been repeated since; re-run it before citing it.

[10] SNAPSHOT PLACEMENT  --  the other half of reaching many decades.

    THIS NOTE IS ABOUT CLOSED RUNS ONLY.  In an OPEN system a genuine
    steady state exists, the estimator is the average of the
    instantaneous post-gate snapshots, and dt never enters it at all --
    so placement is free there and snapshot_mode='events' is harmless.
    Everything below concerns the closed superposition, where dt is a
    weight in the sum.  (BF_analysis.py makes the same split; see the
    comment at superposed_spectrum.)

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
        placement to use for any closed run spanning more than ~2
        decades.

    Note that resampling [9] and placement [10] cure DIFFERENT diseases:
    resampling buys the RANGE (without it N = N_i m_i/m0 dies at ~4
    decades), placement buys a CORRECT MEASUREMENT over that range.  A
    run with [9] and without [10] has flawless dynamics -- m0 agreed with
    the exact Smoluchowski solution to 1e-14 over ten decades -- and a
    meaningless spectrum.

[11] FRAGMENT SPLIT WIDTH  --  the one parameter that BIASES alpha.

    xi is drawn uniformly on (0.5 - w, 0.5 + w) with w =
    frag_split_width, and the pieces are xi*m and (1-xi)*m.  w = 0.5 is
    the uniform split on (0,1) exactly.

    Narrowing w sharpens the mass-age relation but stops the generations
    from overlapping: they merge only after ~(mu/2sigma)^2 splits while
    the cascade takes 6/mu of them, so below about w = 0.25 the spectrum
    develops a PICKET FENCE at the mean step.  At w = 0.05 the
    modulation is a factor 2.3 per 0.3 dex; at w = 0.25 it is 13%.

    This matters more than its size suggests, because it is a SYSTEMATIC
    on alpha, not a variance: a plateau fitted straight through the comb
    returns a shifted index with a small error bar.  Everything else in
    this file that goes wrong adds scatter; this shifts the answer.

    Do not go below 0.25 without looking at the raw histogram first.
    With frag_age_rule = 'heavier' there is no reason to go below it at
    all -- the tracer is narrow by construction, so the job that a small
    w was doing is now done by the age rule, and w can go back to 0.5.

----------------------------------------------------------------------
FAILURE MAP  --  which symptom points at which note
----------------------------------------------------------------------
The traps above produce overlapping fingerprints.  Read this before
concluding that a number is physics:

  alpha -> -2 for EVERY kernel
        clock advanced per event instead of per trial ............ [5iv]
        non-local fragmentation, f = 0 ............................. [8]
        (the two are indistinguishable from the spectrum alone;
         check the acceptance and the value of frag_min_ratio)

  alpha shifted, spectrum visibly combed
        frag_split_width below ~0.25 .............................. [11]

  alpha clean, plateau finder reports a tight 1-decade plateau on a
  CLOSED run that is nonetheless wrong
        snapshot placement 'events' ............................... [10]

  an age class as wide as the whole inertial range, whatever
  frag_age_rule is set to (median age equal to maximum age under
  'inherit')
        expected: an age class is an ensemble, not a path.  Measure a
        trajectory with tracers instead ............................. [7]

  a generation histogram with everything in gen = 0
        the population has not been ground yet, or the run is shorter
        than one waiting time at m_inj.  With frag_min_ratio > 0 a body
        at the top has few partners of comparable mass and the wait
        there is most of the descent ......................... [7], [8]

  <x> or Var(x) not linear in gen, through the origin
        the splits are correlated, i.e. the multiplicative picture is
        wrong -- or gen_max is clipping and the tail is folded into the
        last bin.  Check gen_overflow before concluding physics ..... [7]

  alpha clean but b too low
        the growth-law fit window reaches down into the shelf near
        m_inj, where <m>(tau) has not left the injection mass.  This
        is an ANALYSIS bug, not an engine one: fit the plateau in
        dlog<m>/dlog tau, do not reuse the spectrum's guard band.

  alpha too shallow AND b too low AND the last mass bin rising
        pile-up at the top of the grid: the flux has nowhere to go.
        Keep s(T) at least two decades below max(edges), or add an
        absorbing boundary and drop the last bin from every fit.

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

    Constant kernel,  lambda = 0  :  closed alpha = -1,    b = 1
                                     open   alpha = -3/2,  b = 2
    Geometric kernel, lambda = 2/3:  closed alpha = -5/3,  b = 3
                                     open   alpha = -11/6, b = 6

    WHY b IS THE HARD ONE.  Write the spectrum as n(m) ~ m^-tau with
    tau = -alpha.  In an open run the mass integral is set by the upper
    cut-off, M_1 ~ s^(2-tau) = J t, so

        b = 1/(2-tau),      db/dtau = 1/(2-tau)^2.

    For the geometric kernel 2 - tau = 1/6, hence db/dtau = 36: a ONE
    PERCENT error in alpha moves b by a full unit.  Expect alpha to
    converge cleanly and b not to, quote b to two significant figures at
    most, and do not read a 5%% discrepancy in b as new physics before
    checking alpha to three decimals.

    SCOPE.  These predictions describe the INERTIAL RANGE only -- not
    the neighbourhoods of the injection scale or the sink -- and assume
    pure coagulation or pure fragmentation, never a mixture.  For
    fragmentation they require a disruption threshold f > 0 (note [8]);
    alpha is then independent of f, which is the point of the threshold
    rather than a coincidence.

    WHAT DEPENDS ON WHAT.  alpha is a property of the steady-state
    spectrum: it is untouched by age bookkeeping, and `age_rule` /
    `frag_age_rule` cannot move it.  b is a property of a tracer
    trajectory and depends on the age rule entirely.  The two are linked
    only through the closure above, so a disagreement between them is
    information about the closure -- or about the fit window -- and not
    automatically an error in either.

[12] THE WARM START  --  why seeding is legitimate and where it is not.

    A cold open run starts from N0 bodies at m_inj and has to BUILD the
    power law.  Building it costs one full mass residence time,

        tau_res = M_sys / (q * m_inj)  =  (<m>/m_sink) * t_turn,

    which at six decades is 45 turnover times.  The v4 run stopped at
    0.28 of ONE descent, which is exactly why tau(g) came back censored
    (b = 1.74 instead of 6) and why M_out/M_in was 0.32 instead of 1.

    Seeding the spectrum removes that cost.  It cannot bias alpha,
    PROVIDED the seed is not the answer: the steady state of an open
    cascade is an attractor, so any seed relaxes to the same alpha, and
    the way to demonstrate that rather than assume it is to seed a slope
    that is DELIBERATELY WRONG -- run alpha_seed = -1.5 and -2.1 and
    check that both walk to -1.83.  A run seeded at -1.8333 that reports
    -1.83 has proved nothing.

    What seeding DOES destroy is history.  A seeded body has no ancestor
    in the simulation, so its split count, its age and its family are all
    meaningless.  Hence note [13].

[13] THE INHERITED HONESTY MARK  --  gen = -1.

    Seeded bodies get gen = -1 instead of 0, and -1 is CONTAGIOUS:

        fragmentation: gen(child)  = -1 if gen(parent) < 0 else gen+1
        coagulation:   gen(merged) = -1 if EITHER parent < 0
        injection:     gen = 0                       (honest by definition)

    so contamination can only spread, never heal.  Everything with
    gen < 0 is excluded from gen_counts, gen_num, gen_tau and the
    overflow count; `honest_num` and `honest_mass` report, at every
    snapshot, what fraction of the population is still honest.  The
    seeded bodies wash out on the residence time, so the honest fraction
    is a direct readout of how far the run has actually got.

    gen_time of a seeded body is -inf, not 0: an age that would be
    silently plausible is worse than one that is obviously absent.

[14] THE WAITING LAW  --  b without alpha.

    For a particle sitting at mass m, let dt be the time until the next
    event that CHANGES ITS MASS (it is disrupted, or it merges).  Its
    mass is constant over that whole interval by construction, so the
    interval belongs unambiguously to one mass bin, and

        <dt>(m) ~ 1/nu(m),   nu(m) = Int K(m,m') n(m') dm'.

    Since one event changes m by a factor of order unity, dm ~ m, and
    with the growth law m ~ t^b, dm/dt ~ m^(1-1/b):

        <dt> ~ dm / (dm/dt) ~ m^(1/b).

    Three properties make this the right observable for a warm start:

      - MEMORYLESS.  It asks nothing about where the particle came from,
        so seeded bodies contribute from the first event.  tau(g) cannot
        do this and neither can the isochrones.
      - LOCAL.  The interval is attributed to the bin the particle
        actually sits in, not to a trajectory that crosses the whole
        range, so the sink censors only the bins next to it.
      - UNBIASED BY CONSTRUCTION.  Every interval is recorded exactly
        once, at the moment it ends.  There is no length bias: intervals
        are not sampled by inspecting the population at snapshot times,
        which would over-represent long ones.  The only loss is
        censoring of the intervals still open when the run stops, and
        that is a boundary effect at the top of the range.

    The price is dynamic range.  1/b = 1/6 means <dt> changes by only
    10^(4/6) = 4.6 across FOUR decades of mass.  A shallow slope needs a
    long arm and clean bins, so fit on a guard band that excludes a
    decade at each end, and read the error bar.

    Both an UNGATED accumulator (from t = 0, the one the warm start is
    for) and a GATED one (after the iso_start_sink gate) are kept, so
    the size of the transient contamination is measurable rather than
    assumed.

[15] RESTART FROM A SAVED STATE  --  the checkpoint.

    ic={'state': {...}} restores mass, gen, gen_time and inj_time from a
    finished run.  Nothing is reset.  That is the entire point: after a
    COLD run every particle already carries an honest split count -- the
    number of breakups since its ancestor entered at m_inj -- so the
    relation between generation and mass (generation g sits near
    m_inj exp(mu g)) is intact, the honest fraction is 1 from the first
    event, and no flush is owed.  Contrast the synthetic power-law seed
    of note [12], where there is no history to inherit and the gen = -1
    machinery of note [13] exists precisely to say so.

    THE TIME OFFSET.  Every inherited timestamp is shifted back by the
    previous leg's end time, so it lands NEGATIVE.  Those particles did
    enter before t = 0, and tau = t_phys - t_anc then returns the FULL
    age across the seam instead of restarting it.  This is what fixes the
    censoring that gave b = 1.74 in v4: there the run was shorter than one
    residence time, so tau(g) simply saturated at t_end.  On a restart a
    body that has lived three residence times arrives carrying that age,
    and the first time it steps to a new generation the full age is
    recorded.  High generations populate immediately and uncensored.

    RESTARTS COMPOSE.  Each leg subtracts its own end time and adds it to
    `t_origin`, so t_origin + final_t_phys is the total age of the box
    however many legs it took.  That makes the practical workflow the
    right one: run two flushes, checkpoint, TEST stationarity, and either
    stop or run two more from the checkpoint.  Guessing the length in
    advance is not necessary and, given the next paragraph, not possible.

    HOW LONG IS ENOUGH.  Measured on a six-decade cold run, against the
    natural unit of one box flush = M_sys/m_sink events:

        flushes   live/1e6   <m>     alpha      dn_out/dev
          0.18      14.3     16.2    -1.890        0.96
          0.46       9.4     22.5    -1.881        1.01
          0.73       8.1     24.4    -1.879        1.02
          1.01       6.9     26.6    -1.862        1.02
          1.29       5.9     30.6    -1.829        1.02

    dn_out/devents reached 1 after a fifth of a flush and stayed there,
    and the cumulative M_out/M_in read 1.016 -- yet <m> was still climbing
    at 14 per flush and alpha was still moving, faster at the end than at
    the start.  NEITHER of those two popular criteria detects a spectrum
    that is still evolving:

        M_out = M_in + M_sys(0) - M_sys(t),  so  M_out/M_in = 1 + dM/M_in

    which tends to 1 for the trivial reason that M_in grows without bound.
    It says the run outlived the fill, nothing more.  Use <m> and alpha,
    compared between the two halves of the tail, and see AN.stationarity.
    Extrapolating <m> towards its 45 at alpha = -11/6 puts convergence at
    three to five flushes, i.e. several times what that run did.
----------------------------------------------------------------------
"""

import numpy as np
import json
import inspect
import warnings
import re
from pathlib import Path
import time as _time


# ======================================================================
#  [v2] Sentinel for "this came from the signature, not from the caller"
# ======================================================================
#
# iso_start_sink is armed by default in v2.  An int subclass is used rather
# than None so that the value behaves as a plain 10 EVERYWHERE -- comparisons,
# arithmetic, json, inspect.signature all see 10 -- while `isinstance(x,
# _Default)` still tells the validator whether the caller typed it.

class _Default(int):
    """Marks a value that came from a signature default.  Behaves as int."""
    __slots__ = ()


_ISO_START_SINK_DEFAULT = _Default(10)


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
#  [v5] SEEDING A POWER-LAW POPULATION                          note [12]
# ======================================================================

def sample_powerlaw(alpha, m_lo, m_hi, N, rng=None):
    """N masses drawn from n(m) ~ m^alpha on [m_lo, m_hi], by inverse CDF.

    For alpha != -1 the normalised CDF is

        F(m) = (m^(a) - m_lo^(a)) / (m_hi^(a) - m_lo^(a)),   a = alpha + 1

    so m = [m_lo^a + u (m_hi^a - m_lo^a)]^(1/a).  The a -> 0 case
    (alpha = -1) is log-uniform and is handled separately, because the
    general formula is 0/0 there and loses all its digits well before it.

    Done in logs rather than in powers: with alpha = -1.83 and six
    decades, m_lo^a spans 10^(-5) of m_hi^a, and the naive form throws
    away half the mantissa on the light end -- exactly the end that
    carries the statistics.
    """
    a = float(alpha) + 1.0
    N = int(N)
    if N <= 0:
        return np.zeros(0, dtype=float)
    if not (0.0 < m_lo < m_hi):
        raise ValueError("need 0 < m_lo < m_hi")
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(N)
    L_lo = np.log(float(m_lo))
    L_hi = np.log(float(m_hi))
    if abs(a) < 1e-9:
        return np.exp(L_lo + u * (L_hi - L_lo))
    # work with the LOG of the CDF endpoints; factor out the larger one
    if a > 0.0:
        # m_hi^a dominates: m = m_hi * [ (1-u) e^{a(L_lo-L_hi)} + u ]^(1/a)
        z = (1.0 - u) * np.exp(a * (L_lo - L_hi)) + u
        return np.exp(L_hi + np.log(z) / a)
    else:
        # m_lo^a dominates: m = m_lo * [ (1-u) + u e^{a(L_hi-L_lo)} ]^(1/a)
        z = (1.0 - u) + u * np.exp(a * (L_hi - L_lo))
        return np.exp(L_lo + np.log(z) / a)


def steady_N_for_mass(alpha, m_lo, m_hi, M_target):
    """How many bodies a power-law seed needs to carry mass M_target.

    <m> = Int m^(alpha+1) dm / Int m^(alpha) dm over [m_lo, m_hi], so
    N = M_target / <m>.  Handy when the cold run's steady state is known
    by its MASS (which is set by the injection rate and the residence
    time) rather than by its number.
    """
    a = float(alpha)
    lo, hi = float(m_lo), float(m_hi)

    def _mom(p):
        q = a + p + 1.0
        if abs(q) < 1e-12:
            return np.log(hi / lo)
        return (hi ** q - lo ** q) / q

    mbar = _mom(1) / _mom(0)
    return max(int(round(float(M_target) / mbar)), 1), float(mbar)


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
    ic,                         # THREE accepted forms:
                                #   {'m': m0, 'N': N0}
                                #       monodisperse, exactly as in v2..v4.
                                #   {'mass': array}
                                #       an explicit list of masses.
                                #   {'steady': {'alpha':a, 'm_lo':.., 'm_hi':..,
                                #               'N':..}}                 [v5]
                                #       N bodies drawn from n(m) ~ m^a on
                                #       [m_lo, m_hi] -- the WARM START, note [12].
                                #       These bodies carry gen = -1 and so do all
                                #       of their descendants, note [13].
                                #       Optionally 'M' instead of 'N' to fix the
                                #       seeded MASS and let N follow.
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
    frag_split_width=0.5,       # HALF-WIDTH of the fragment split, fragmentation only.
                                # xi is drawn uniformly on (0.5 - w, 0.5 + w) and the pieces
                                # are xi*m and (1-xi)*m.  w = 0.5 is the old uniform split on
                                # (0,1) exactly, so this default changes nothing.  Smaller w
                                # narrows the isochrone -- the mass-age relation sharpens --
                                # but the generations stop overlapping and the spectrum
                                # develops a picket fence at the mean step.  Generations
                                # merge only after ~(mu/2sigma)^2 splits while the cascade
                                # takes 6/mu of them, so w below about 0.25 combs the
                                # spectrum: at w = 0.05 the modulation is a factor 2.3 per
                                # 0.3 dex, at w = 0.25 it is 13%.  Do not go below 0.25
                                # without looking at the raw histogram first.
    frag_age_rule="inherit",    # AGE INHERITANCE, fragmentation only.  Which fragment keeps
                                # the parent's clock and which is born now, at t_phys.
                                #   'inherit'  both keep the parent's t_born (v2 behaviour).
                                #              The age class is then a FAMILY -- the whole
                                #              descendant tree of one injected body -- and it
                                #              spans the entire inertial range by construction.
                                #   'heavier'  the heavier fragment keeps the clock, the lighter
                                #              is reborn.  The tracer is the always-heavy chain:
                                #              a characteristic of the transport equation, i.e.
                                #              the front.  This is the isochrone one wants.
                                #   'lighter'  mirror of the above; kept as a sensitivity test,
                                #              not as physics.
                                #   'both_new' both reborn.  Null test: destroys the age axis
                                #              on purpose, so <m>(tau) must collapse to noise.
    # ---- stopping ----
    max_time=np.inf,
    max_events=np.inf,
    stop_max_mass=None,         # stop when m_max >= this (closed coagulation)
    stop_min_mass=None,         # stop when m_mean <= this (closed fragmentation)
    stop_sink_events=None,      # [v2] stop once this many PHYSICAL particles have been absorbed
                                # at the sink.  Unlike max_events, which is a statement about
                                # RESOURCES, this is a statement about the PHYSICS: the cascade
                                # has demonstrably delivered that many particles across the whole
                                # inertial range.  It therefore needs no retuning when N_ss or
                                # the mass range changes.  None disables it (the default).
                                # Open systems only; silently ignored when system='closed'.
    max_stall_tries=2_000_000,  # stop if this many majorant attempts yield no event
                                # (happens once everything sits below min_frag_mass)
    # ---- snapshots ----
    snapshot_mode="events",     # 'events' | 't_phys' | 'log_m0'   (see note [10])
                                #   'log_m0' is the one to use beyond ~2 decades
    snapshot_stride=1e4,        # events / time units / SNAPSHOTS PER DECADE of m0
    snapshot_first_at_start=True,
    # ---- isochrones ----
    iso_age_edges=None,         # 1D age-bin edges; None disables
    iso_t_start=0.0,            # start accumulating isochrones only after this ABSOLUTE time.
                                # Beware: the clock runs as dt = 2V/(wR) with R ~ N^2, so the
                                # time reached after E attempts is t = 2E/N^2.  A fixed value
                                # tuned at one N is 100x too large at 10N and the isochrone
                                # histogram then stays empty (b comes back as nan).  Prefer
                                # iso_start_sink, or scale this with 1/N_SS.
    iso_start_sink=_ISO_START_SINK_DEFAULT,
                                # start accumulating isochrones only after this many PHYSICAL
                                # particles have been absorbed at the sink.  This is the
                                # N-independent steady-state gate: mass has demonstrably
                                # crossed the whole inertial range before any age is binned.
                                # Both gates apply (AND).
                                # [v2] The default is 10 instead of 0, and it applies to
                                # COAGULATION and FRAGMENTATION alike -- the sink is the sink,
                                # whichever end of the cascade it sits at.  In a CLOSED system
                                # there is no sink, so the gate is silently disarmed instead of
                                # raising; only an explicitly passed value on a closed run is an
                                # error (see the validation block below).
    track_generations=True,     # [v4] see note [7].  Carry `gen` on every particle -- the
                                # number of splits between it and the body that ENTERED --
                                # and bin the live population by (gen, mass) at each
                                # snapshot, exactly as the isochrones bin it by (age, mass).
                                # This is what the tracer machinery of v2/v3 was for, done
                                # on the whole population rather than on a few thousand
                                # tagged paths: same quantity, four orders of magnitude more
                                # of it, 16 MB at N = 4e6.
    gen_max=64,                 # Highest generation resolved.  Everything above is counted
                                # in `gen_overflow` and NOT binned, so a clipped tail is a
                                # number rather than a spurious pile in the last bin.  A
                                # body needs ~log(m_inj/m_sink)/|<ln xi>| splits to reach
                                # the sink -- 28 over six decades at w = 1/2.
    age_rule="min",             # coagulation age inheritance: 'min'|'mass_weighted'|'heavier'
    track_waiting=True,         # [v5] see note [14].  Accumulate <dt>(m), the mean time
                                # between successive MASS-CHANGING events for a particle
                                # sitting in a given mass bin.  <dt> ~ m^(1/b), so this
                                # measures b DIRECTLY, without going through alpha and
                                # its db/dalpha = b^2 = 36 amplification.  Memoryless,
                                # therefore valid on a seeded population from t = 0.
                                # Costs one float64 per particle and two adds per event.
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
    if frag_age_rule not in ("inherit", "heavier", "lighter", "both_new"):
        raise ValueError("frag_age_rule must be 'inherit', 'heavier', "
                         "'lighter' or 'both_new'")
    if age_rule not in ("min", "max", "mass_weighted", "heavier"):
        raise ValueError("age_rule must be 'min', 'max', 'mass_weighted' or 'heavier'")
    if int(gen_max) < 1:
        raise ValueError("gen_max must be >= 1")
    if not (0.0 <= frag_split_width <= 0.5):
        raise ValueError("frag_split_width is a half-width: 0 <= w <= 0.5 "
                         "(0.5 = uniform split, 0 = exact halving)")
    # The sink gate is now armed BY DEFAULT (10 particles), so a bare
    # `raise` here would make every CLOSED run fail on an argument the caller
    # never passed.  Distinguish the two cases with the _Default marker: a value
    # that came from the signature is silently disarmed, a value the caller typed
    # is still an error, because asking for a sink gate in a system that has no
    # sink is a mistake worth hearing about.
    if system != "open":
        if not isinstance(iso_start_sink, _Default) and iso_start_sink > 0:
            raise ValueError("iso_start_sink needs an absorbing sink, i.e. system='open'")
        iso_start_sink = 0
        if stop_sink_events is not None:
            raise ValueError("stop_sink_events needs an absorbing sink, i.e. system='open'")

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

    w = float(weight)                               # physical particles per sim particle
    V = float(volume)

    # [v5] The rng is built BEFORE the initial condition is parsed, because a
    # seeded IC draws from it.  Nothing else draws in between, so a run with
    # ic={'m','N'} consumes the stream in exactly the order v4 did -- which is
    # what makes the bit-identity test meaningful.
    if rng is None:
        rng = np.random.default_rng()
    rnd = rng.random
    rint = rng.integers

    # ------------------------------------------------------------------
    # 0b. [v5] THE INITIAL CONDITION -- three forms, one of them warm
    # ------------------------------------------------------------------
    seeded = False              # was the population planted rather than grown?
    seed_info = None
    restart_from = None         # [v5] name of the state file this leg grew from
    t_origin = 0.0              # [v5] physical time accumulated in EARLIER legs
    _gen0 = _gt0 = _it0 = None  # [v5] inherited per-particle state, if any
    if "steady" in ic:
        # ---- WARM START, note [12] ----
        _s = dict(ic["steady"])
        alpha_seed = float(_s.get("alpha", -1.8333333333333333))
        m_lo_seed = float(_s["m_lo"])
        m_hi_seed = float(_s["m_hi"])
        if "N" in _s and _s["N"] is not None:
            N0 = int(_s["N"])
            _mbar = None
        elif "M" in _s and _s["M"] is not None:
            N0, _mbar = steady_N_for_mass(alpha_seed, m_lo_seed, m_hi_seed,
                                          float(_s["M"]) / w)
        else:
            raise ValueError("ic['steady'] needs either 'N' (bodies) or 'M' (mass)")
        if N0 < 2:
            raise ValueError("a seeded population needs at least 2 bodies")
        m_init_arr = sample_powerlaw(alpha_seed, m_lo_seed, m_hi_seed, N0, rng)
        seeded = True
        seed_info = {"alpha": alpha_seed, "m_lo": m_lo_seed, "m_hi": m_hi_seed,
                     "N": int(N0), "mean_mass": float(m_init_arr.mean())}
        # `m_init` is now only a REPORTING number and must not be used as a
        # default for the injection mass -- the seed has no single mass.  Which
        # end injection sits at is fixed by the process, not by the IC.
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            injection_mass = m_hi_seed if process == "fragmentation" else m_lo_seed
    elif "state" in ic:
        # ---- RESTART FROM A SAVED STATE, note [15] ----
        _st = ic["state"]
        m_init_arr = np.asarray(_st["mass"], dtype=float).ravel().copy()
        N0 = int(m_init_arr.size)
        if N0 < 2:
            raise ValueError("ic['state'] needs at least 2 particles")
        for _k in ("gen", "gen_time", "inj_time"):
            if _k not in _st:
                raise ValueError("ic['state'] needs 'mass', 'gen', 'gen_time' and "
                                 "'inj_time'; '%s' is missing.  A run saved with "
                                 "drop_particles=True does not carry them." % _k)
            if np.asarray(_st[_k]).ravel().size != N0:
                raise ValueError("ic['state']['%s'] has %d entries against %d masses"
                                 % (_k, np.asarray(_st[_k]).ravel().size, N0))
        _gen0 = np.asarray(_st["gen"], dtype=np.int64).ravel().copy()
        # THE TIME OFFSET, and it is the whole trick.  The clock of this leg
        # starts at zero, so every inherited timestamp is shifted back by the
        # previous leg's end time and lands NEGATIVE.  That is not a hack: those
        # particles genuinely entered before t = 0, and tau = t_phys - t_anc then
        # returns the FULL age across the seam instead of restarting it.  This is
        # exactly what was censored in v4, where tau(g) saturated at t_end and
        # gave b = 1.74 instead of 6.  Restarts compose: each leg subtracts its
        # own end time, so an age carried through five legs is still the true one.
        _t_prev = float(_st.get("t_end", 0.0))
        _gt0 = np.asarray(_st["gen_time"], dtype=float).ravel() - _t_prev
        _it0 = np.asarray(_st["inj_time"], dtype=float).ravel() - _t_prev
        restart_from = _st.get("source")
        t_origin = float(_st.get("t_origin", 0.0)) + _t_prev   # time before this leg
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            raise ValueError("ic={'state': ...} carries no single mass; "
                             "pass injection_mass explicitly")
    elif "mass" in ic:
        m_init_arr = np.asarray(ic["mass"], dtype=float).ravel().copy()
        N0 = int(m_init_arr.size)
        if N0 < 1:
            raise ValueError("ic['mass'] is empty")
        m_init = float(m_init_arr.mean())
        if injection_mass is None:
            raise ValueError("ic={'mass': ...} carries no single mass; "
                             "pass injection_mass explicitly")
    else:
        # ---- COLD START: v2/v3/v4 behaviour, unchanged ----
        m_init = float(ic["m"])
        N0 = int(ic["N"])
        m_init_arr = np.full(N0, m_init, dtype=float)
        if injection_mass is None:
            injection_mass = m_init

    if np.any(m_init_arr <= 0.0):
        raise ValueError("initial masses must be strictly positive")
    if _gen0 is not None:
        _gmax_in = int(_gen0.max()) if _gen0.size else 0
        if _gmax_in > int(gen_max):
            warnings.warn(
                "the restored state carries generations up to %d while gen_max = %d: "
                "everything above the cap goes straight into gen_overflow and is "
                "never binned.  Raise gen_max to at least %d."
                % (_gmax_in, int(gen_max), _gmax_in + 16),
                RuntimeWarning, stacklevel=2)
    # A seeded mass outside `edges` is silently CLAMPED into the end bin by
    # mass_to_bin, which turns a mis-specified seed into a spurious spike at
    # one end of the spectrum rather than into an error.  Say so instead.
    _n_out_of_grid = int(np.count_nonzero((m_init_arr < edges[0])
                                          | (m_init_arr >= edges[-1])))
    if _n_out_of_grid:
        warnings.warn(
            "%d of %d initial masses lie outside the bin grid [%.3g, %.3g) and "
            "will be clamped into the end bins, which shows up as a spike there. "
            "Widen `edges` or narrow the seed range."
            % (_n_out_of_grid, N0, edges[0], edges[-1]),
            RuntimeWarning, stacklevel=2)

    # ------------------------------------------------------------------
    # 1.  Particle arrays.  The live prefix [0:live] is the active set.
    # ------------------------------------------------------------------
    cap = max(N0, 1024)
    mass = np.empty(cap, dtype=np.float64)
    inj_time = np.empty(cap, dtype=np.float64)
    bin_of = np.empty(cap, dtype=np.int32)
    pos_in_bin = np.empty(cap, dtype=np.int32)

    gen = np.zeros(cap, dtype=np.int32)             # splits since the ancestor entered
    gen_time = np.zeros(cap, dtype=np.float64)      # when that ancestor entered
    # [v5] when this particle's MASS last changed.  Not the same as inj_time
    # (which is a family clock and survives fragmentation) and not the same as
    # gen_time (which points at the ancestor).  This one is strictly local: it
    # is reset every time the particle takes part in an event.       note [14]
    t_change = np.zeros(cap, dtype=np.float64)

    mass[:N0] = m_init_arr
    #  На перезапуске возрасты и счётчики НАСЛЕДУЮТСЯ, note [15]; иначе всё
    #  начинается с нуля, как в v2..v4.
    inj_time[:N0] = _it0 if _it0 is not None else 0.0
    #  t_change обнуляется всегда: это чисто локальные часы закона ожидания, и
    #  прошлый интервал был закрыт в прошлом прогоне.  Оценку exposure это не
    #  трогает вовсе -- занятость копится независимо, -- смещается только
    #  оценка interval, и только на первый интервал каждой частицы.
    t_change[:N0] = 0.0
    # [v5] A SEEDED body did not enter the system -- it was planted, and its
    # split count is unknowable.  gen = -1 marks that, and note [13] makes the
    # mark contagious.  A COLD monodisperse IC at m_inj is indistinguishable
    # from an injected body, so it stays honest at gen = 0, as in v4.
    if _gen0 is not None:
        gen[:N0] = np.clip(_gen0, -1, np.iinfo(np.int32).max).astype(np.int32)
        gen_time[:N0] = _gt0
    else:
        gen[:N0] = -1 if seeded else 0
        gen_time[:N0] = -np.inf if seeded else 0.0
    live = N0

    bins = [[] for _ in range(B)]                   # bin -> list of particle indices

    _PARR = ("mass", "inj_time", "gen", "gen_time", "t_change", "bin_of", "pos_in_bin")

    def ensure_capacity(need):
        """Grow all per-particle arrays together, keeping the live prefix."""
        nonlocal mass, inj_time, gen, gen_time, t_change, bin_of, pos_in_bin
        if need <= mass.size:
            return
        new_cap = max(need, int(mass.size * 1.6) + 1)
        for name in _PARR:
            old = locals_ref[name]
            new = np.empty(new_cap, dtype=old.dtype)
            new[:live] = old[:live]
            locals_ref[name] = new
        mass = locals_ref["mass"]
        inj_time = locals_ref["inj_time"]
        gen = locals_ref["gen"]
        gen_time = locals_ref["gen_time"]
        t_change = locals_ref["t_change"]
        bin_of = locals_ref["bin_of"]
        pos_in_bin = locals_ref["pos_in_bin"]

    locals_ref = {"mass": mass, "inj_time": inj_time, "gen": gen, "gen_time": gen_time,
                  "t_change": t_change, "bin_of": bin_of, "pos_in_bin": pos_in_bin}

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

    # [v5] Occupancy hook, note [14].  apply_delta is the ONE choke point every
    # change of N_bin passes through -- injection, coagulation, fragmentation,
    # the sink -- so hanging the Int N_b dt flush here is what makes that
    # integral exact rather than sampled.  A one-element list rather than a
    # forward reference to a function defined further down: no definition-order
    # hazard, and the hook is a no-op until the accumulators exist.
    _occ_hook = [None]

    def apply_delta(b, dN):
        """One bin count changed by dN: update T, rate_row and R in O(B), vectorised."""
        nonlocal R
        if dN == 0:
            return
        if _occ_hook[0] is not None:
            _occ_hook[0]()                       # flush BEFORE N_bin moves
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
            gen[j] = gen[last]
            gen_time[j] = gen_time[last]
            t_change[j] = t_change[last]
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

    def add_particle(m, t_born, g=0, t_anc=None):
        """Append one particle; returns its index.  `g` is its generation and `t_anc` the
        moment its ancestor entered -- both default to a newly ENTERED body."""
        nonlocal live
        ensure_capacity(live + 1)
        i = live
        mass[i] = m
        inj_time[i] = t_born
        gen[i] = g
        gen_time[i] = t_born if t_anc is None else t_anc
        t_change[i] = t_phys          # [v5] its mass is new as of NOW, note [14]
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
        if age_rule == "max":
            return max(t1, t2)                       # youngest ancestor wins
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
    # [v5] summed, not N0*m_init: with a seeded population m_init is only the
    # mean, and N0*mean differs from the true sum in the last few bits -- which
    # would show up as a non-zero mass_drift, the one number in this engine
    # that is supposed to be identically zero.
    M_sys = w * float(m_init_arr.sum())   # physical mass currently in the system
    M_in = 0.0
    M_out = 0.0
    m_max = float(m_init_arr.max())
    m_min = float(m_init_arr.min())
    sink_events = 0
    M_ref = M_sys                        # reference mass; drift must stay identically zero

    # [v2] "n_out" is the sink counter as a TIME SERIES.  Deliberately NOT named
    # "sink_events": the packing step builds `out` from `rows` first and then
    # assigns the final scalar out["sink_events"], which would silently clobber
    # the array.  Two names, no collision.
    rows = {k: [] for k in ("t", "events", "live", "N_phys", "m_mean", "m_max",
                            "m_min", "M_sys", "M_in", "M_out", "n_out", "weight",
                            "honest_num", "honest_mass", "dndm")}

    # [v4] Generation histogram, gated exactly like the isochrones -- the same
    # statement that the steady state is up.  gen_num and gen_tau_sum give <tau>(g)
    # for free, which is the clock and therefore b.
    use_gen = bool(track_generations)
    G_MAX = int(gen_max)
    gen_counts = np.zeros((G_MAX + 1, B), dtype=np.float64) if use_gen else None
    gen_num = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_tau_sum = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_mass_sum = np.zeros((G_MAX + 1, B), dtype=np.float64) if use_gen else None
    gen_overflow = 0.0
    gen_snaps = 0
    # [v4] TWO different times per generation, and they are not interchangeable.
    #   gen_tau_live  -- mean age of the particles CURRENTLY at g, taken at snapshots.
    #                    It saturates at the residence time of the box: after one
    #                    turnover everything alive is young whatever its generation.
    #   gen_tau_reach -- age AT THE MOMENT the particle reached g, accumulated at the
    #                    event.  This is the "time to reach generation g", the direct
    #                    analogue of the tracer's tau(k), and the one that carries b.
    gen_reach_sum = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None
    gen_reach_n = np.zeros(G_MAX + 1, dtype=np.float64) if use_gen else None

    def gen_reached(g, t_anc):
        """A particle has just arrived at generation g; record how long that took.

        [v5] g < 0 is the seeded-ancestor mark of note [13] and is silently
        skipped -- t_anc is -inf there, so the guard is load-bearing, not
        cosmetic."""
        if use_gen and 0 <= g <= G_MAX and w * sink_events >= iso_start_sink \
                and t_phys >= iso_t_start:
            gen_reach_sum[g] += w * (t_phys - t_anc)
            gen_reach_n[g] += w

    # [v5] THE WAITING LAW, note [14].  <dt>(m) per mass bin, accumulated at
    # the moment each interval ENDS, which is what makes it unbiased.  Two
    # copies: ungated (from t = 0, the one a warm start is for) and gated by
    # the same steady-state gate as the isochrones, so the size of the
    # transient contamination is a measurement rather than a hope.
    # TWO estimators of the same <dt>, and the difference between them is
    # itself the diagnostic:
    #
    #   INTERVAL   sum of completed intervals / their number.  Simple, but
    #              CENSORED: an interval still running when the run stops is
    #              never recorded, and long intervals are preferentially the
    #              ones still running.  Since <dt> grows with m, the loss is
    #              concentrated at large m and it FLATTENS the slope -- i.e.
    #              it inflates b.  Same disease as tau(g) in v4, milder.
    #
    #   EXPOSURE   total particle-time spent in the bin / number of events
    #              that ended a stay there.  This is the standard
    #              occupancy/exposure ratio, and it is unbiased WHATEVER the
    #              censoring, because a partly-elapsed interval contributes
    #              its elapsed part to the numerator and nothing to the
    #              denominator, which is exactly right.  Use this one.
    #
    # The occupancy integral Int N_b(t) dt is exact, not sampled: N_bin is
    # piecewise constant in time and every change to it goes through
    # apply_delta, so flushing there costs one length-B add per EVENT (not
    # per trial -- within one event dt is zero and the add is skipped).
    use_wait = bool(track_waiting)
    wait_sum = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_sq = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_n = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_sum_g = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_sq_g = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_n_g = np.zeros(B, dtype=np.float64) if use_wait else None
    occ = np.zeros(B, dtype=np.float64) if use_wait else None      # Int N_b dt * w
    occ_g = np.zeros(B, dtype=np.float64) if use_wait else None
    wait_ev = np.zeros(B, dtype=np.float64) if use_wait else None  # stays ended in b
    wait_ev_g = np.zeros(B, dtype=np.float64) if use_wait else None
    t_occ_last = 0.0
    t_occ_gate = np.nan          # when the gated occupancy started

    def occ_flush():
        """Add the particle-time accrued since the last flush.  Must run BEFORE
        any change to N_bin, which is why it lives at the top of apply_delta --
        the single choke point every count change passes through."""
        nonlocal t_occ_last, t_occ_gate
        if not use_wait:
            return
        dtl = t_phys - t_occ_last
        if dtl > 0.0:
            # `occ[:] +=`, not `occ +=`: the bare form would rebind the name and
            # make it a local of this closure.
            _inc = N_bin * (w * dtl)
            occ[:] += _inc
            if w * sink_events >= iso_start_sink and t_phys >= iso_t_start:
                if not np.isfinite(t_occ_gate):
                    t_occ_gate = t_occ_last
                occ_g[:] += _inc
        t_occ_last = t_phys

    def wait_close(i):
        """Particle i is about to have its mass changed: close its stay.

        The particle has sat at a CONSTANT mass since t_change[i], so the whole
        interval belongs to bin_of[i] with no ambiguity -- that is the reason
        this observable is local while tau(g) is not."""
        if not use_wait:
            return
        b_i = int(bin_of[i])
        gated = (w * sink_events >= iso_start_sink and t_phys >= iso_t_start)
        wait_ev[b_i] += w                      # exposure estimator: denominator
        if gated:
            wait_ev_g[b_i] += w
        dt_i = t_phys - t_change[i]
        if dt_i > 0.0:                         # interval estimator: cross-check
            wait_sum[b_i] += w * dt_i
            wait_sq[b_i] += w * dt_i * dt_i
            wait_n[b_i] += w
            if gated:
                wait_sum_g[b_i] += w * dt_i
                wait_sq_g[b_i] += w * dt_i * dt_i
                wait_n_g[b_i] += w
        t_change[i] = t_phys

    if use_wait:
        _occ_hook[0] = occ_flush          # arm it; see apply_delta

    use_iso = iso_age_edges is not None
    if use_iso:
        iso_age_edges = np.asarray(iso_age_edges, dtype=float)
        iso_counts = np.zeros((iso_age_edges.size - 1, B), dtype=np.float64)
        iso_snaps = 0
    else:
        iso_counts = None
        iso_snaps = 0
    iso_t_begin = np.nan       # physical time at which accumulation actually started

    def snapshot():
        nonlocal iso_snaps, iso_t_begin, gen_snaps, gen_overflow
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
        rows["n_out"].append(w * sink_events)        # [v2] physical particles absorbed so far
        rows["weight"].append(w)
        # [v5] What fraction of the population still has an honest history,
        # note [13].  On a cold run this is 1.0 at every snapshot by
        # construction; on a warm one it is the direct readout of how far the
        # seeded bodies have been flushed, i.e. of how much of the run is
        # actually usable for anything that depends on ancestry.
        if live > 0:
            _hm = gen[:live] >= 0
            rows["honest_num"].append(float(np.count_nonzero(_hm)) / live)
            _mm = mass[:live]
            _tot = float(_mm.sum())
            rows["honest_mass"].append(float(_mm[_hm].sum()) / _tot if _tot > 0 else np.nan)
        else:
            rows["honest_num"].append(np.nan)
            rows["honest_mass"].append(np.nan)
        # dN/dm in PHYSICAL units: counts * w / bin width / volume
        rows["dndm"].append(N_bin * (w / V) / widths)

        # Two gates, both permissive by default.  iso_start_sink is the physical one:
        # it waits until the cascade has actually delivered mass to the sink, which is
        # the only N-independent statement of "the steady state is up".
        if (use_gen and live > 0 and t_phys >= iso_t_start
                and w * sink_events >= iso_start_sink):
            _g = gen[:live]
            # [v5] THREE classes now, not two: honest and resolved, honest and
            # overflowing, and dishonest (gen < 0, note [13]).  The dishonest
            # ones must not land in gen_overflow either -- that counter is a
            # statement about gen_max being too small, and mixing the seeded
            # population into it would make it read as a truncation warning
            # for the whole first residence time.
            _hon = _g >= 0
            _ok = _hon & (_g <= G_MAX)
            gen_overflow += w * float(np.count_nonzero(_hon & (_g > G_MAX)))
            _gg = _g[_ok].astype(np.int64)
            _bb = bin_of[:live][_ok].astype(np.int64)
            np.add.at(gen_counts, (_gg, _bb), w)
            np.add.at(gen_mass_sum, (_gg, _bb), w * mass[:live][_ok])
            np.add.at(gen_num, _gg, w)
            np.add.at(gen_tau_sum, _gg, w * (t_phys - gen_time[:live][_ok]))
            gen_snaps += 1

        if (use_iso and live > 0 and t_phys >= iso_t_start
                and w * sink_events >= iso_start_sink):
            if iso_snaps == 0:
                iso_t_begin = t_phys
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
        # [v2] The PHYSICAL brake.  w * sink_events, not the bare counter, so the
        # criterion stays in physical particles and survives the move to BF.py,
        # where w != 1.  Same convention as the iso_start_sink gate below.
        if stop_sink_events is not None and w * sink_events >= stop_sink_events:
            stop_reason = "stop_sink_events"
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
                    i_inj = add_particle(float(injection_mass), t_phys, 0, t_phys)
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
            # [v5] Close BOTH intervals before either mass changes, note [14].
            # j is closed too even though it is about to be deleted: the time
            # it spent at m2 is data, and throwing it away would bias <dt>
            # against exactly the bins that merge most often.
            wait_close(i); wait_close(j)
            inj_time[i] = inherit_age(inj_time[i], inj_time[j], m1, m2)
            # [v4] one split deeper than the deeper parent; inherits the OLDER ancestor
            # [v5] unless either parent is seeded, in which case the product is
            # seeded too and the count is meaningless, note [13].
            if gen[i] < 0 or gen[j] < 0:
                gen[i] = -1
                gen_time[i] = -np.inf
            else:
                gen[i] = (gen[i] if gen[i] >= gen[j] else gen[j]) + 1
                if gen_time[j] < gen_time[i]:
                    gen_time[i] = gen_time[j]
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
            xi = 0.5 + frag_split_width * (2.0 * rnd() - 1.0)
            ma, mb = xi * mp, (1.0 - xi) * mp
            t_par = float(inj_time[ip])                  # the parent's clock
            # [v5] note [14].  ONLY the parent is closed: in this process the
            # impactor is untouched, its mass does not change, and its own
            # interval is still running.  Closing it here would count a
            # non-event and pull <dt> down in every bin that acts as an
            # impactor -- which, with a power law, is all of them.
            wait_close(ip)

            # Which piece carries the parent's clock and which starts a new one.
            # 'inherit' reproduces v2 exactly; everything else resets one of the two
            # to t_phys, which is ALREADY the current time here -- dt was added at the
            # top of the trial loop, the same t_phys the injector stamps on
            # new monomers.  Do not recompute it.
            if frag_age_rule == "inherit":
                t_a = t_b = t_par
            elif frag_age_rule == "both_new":
                t_a = t_b = t_phys
            else:
                keep_a = (ma >= mb) if frag_age_rule == "heavier" else (ma < mb)
                t_a, t_b = (t_par, t_phys) if keep_a else (t_phys, t_par)

            mass[ip] = ma
            inj_time[ip] = t_a          # <-- MUST be written now: the slot is REUSED and
                                        #     under v2 it was silently already correct.
            move_bin(ip, mass_to_bin(ma))
            # [v4] BOTH fragments go one generation deeper and keep the ancestor's
            # clock, whatever frag_age_rule did to inj_time -- gen_time is a separate
            # field precisely so the two cannot interfere.
            # [v5] -1 is contagious and never heals, note [13]: a fragment of a
            # seeded body is as historyless as the body was.
            _g0 = int(gen[ip])
            _g1 = -1 if _g0 < 0 else _g0 + 1
            _tanc = float(gen_time[ip])
            gen[ip] = _g1
            i_new = add_particle(mb, t_b, _g1, _tanc)
            gen_reached(_g1, _tanc)      # both fragments arrived at the same time
            gen_reached(_g1, _tanc)      # (no-op when _g1 < 0)
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

    if use_wait:
        occ_flush()   # [v5] close the occupancy integral at t_end, note [14]

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
    out["sink_events"] = float(sink_events)
    if use_iso:
        out["iso_age_edges"] = iso_age_edges
        out["iso_counts"] = iso_counts
        out["iso_dndm"] = iso_counts / widths[None, :]
        out["iso_snapshots"] = float(iso_snaps)
        out["iso_t_begin"] = float(iso_t_begin)
        # An empty isochrone histogram is the silent failure mode: every downstream
        # quantity comes back as nan and nothing says why.  Say why.
        if iso_snaps == 0:
            warnings.warn(
                "isochrones requested but NEVER accumulated: the run ended at "
                "t = %.3g with %d sink absorptions, while the gates ask for "
                "t >= %.3g and >= %g absorptions.  iso_counts is all zeros, so "
                "<m>(tau) and b will be nan.  The clock is t = 2*attempts/N^2, so "
                "raising N at fixed max_events SHORTENS the run."
                % (t_phys, sink_events, iso_t_start, iso_start_sink),
                RuntimeWarning, stacklevel=2)

    # ---- generations -----------------------------------------------------------
    if use_gen:
        out["gen_counts"] = gen_counts
        out["gen_dndm"] = gen_counts / widths[None, :] / max(gen_snaps, 1)
        out["gen_mass"] = gen_mass_sum / max(gen_snaps, 1)
        out["gen_num"] = gen_num
        with np.errstate(invalid="ignore", divide="ignore"):
            out["gen_tau_live"] = np.where(gen_num > 0, gen_tau_sum / gen_num, np.nan)
            out["gen_tau"] = np.where(gen_reach_n > 0, gen_reach_sum / gen_reach_n, np.nan)
        out["gen_reach_n"] = gen_reach_n
        out["gen_snapshots"] = float(gen_snaps)
        out["gen_overflow"] = float(gen_overflow)
        out["gen_index"] = np.arange(G_MAX + 1, dtype=float)
        out["final_gen"] = gen[:live].astype(float)
        out["final_gen_time"] = gen_time[:live].copy()
        # [v5] how much of the seeded population is left at the end, note [13]
        _fh = gen[:live] >= 0
        out["final_honest_num"] = (float(np.count_nonzero(_fh)) / live) if live else np.nan
        if seeded and live and np.count_nonzero(_fh) == 0:
            warnings.warn(
                "the run ended with NO honest bodies at all: every particle still "
                "descends from the seed, so gen_counts / gen_tau are empty and the "
                "only usable estimate of b is the waiting law.  The run is shorter "
                "than one residence time.", RuntimeWarning, stacklevel=2)
        if gen_snaps == 0:
            warnings.warn(
                "generations requested but NEVER accumulated: the run ended at t = %.3g "
                "with %d sink absorptions, while the gate asks for t >= %.3g and >= %g "
                "absorptions.  gen_counts is all zeros."
                % (t_phys, sink_events, iso_t_start, iso_start_sink),
                RuntimeWarning, stacklevel=2)
        elif gen_overflow > 0:
            warnings.warn(
                "%.3g particle-snapshots had gen > gen_max = %d and were NOT binned "
                "(%.2f%% of the total).  Raise gen_max, or read the histogram as "
                "truncated." % (gen_overflow, G_MAX,
                                100 * gen_overflow / max(gen_num.sum() + gen_overflow, 1.0)),
                RuntimeWarning, stacklevel=2)

    # ---- [v5] the waiting law, note [14] ---------------------------------------
    if use_wait:
        with np.errstate(invalid="ignore", divide="ignore"):
            # --- EXPOSURE estimator: particle-time per event.  The primary one.
            out["wait_mean"] = np.where(wait_ev > 0, occ / wait_ev, np.nan)
            out["wait_mean_gated"] = np.where(wait_ev_g > 0, occ_g / wait_ev_g, np.nan)
            # Poisson error on the COUNT propagates straight through, since the
            # exposure is a measured time and not a random variable here:
            # sd(1/nu)/(1/nu) = 1/sqrt(n_events).
            out["wait_sem"] = out["wait_mean"] / np.sqrt(np.maximum(wait_ev, 1.0))
            out["wait_sem_gated"] = (out["wait_mean_gated"]
                                     / np.sqrt(np.maximum(wait_ev_g, 1.0)))
            # --- INTERVAL estimator: the censored cross-check.
            _mi = np.where(wait_n > 0, wait_sum / wait_n, np.nan)
            _mig = np.where(wait_n_g > 0, wait_sum_g / wait_n_g, np.nan)
            _var = np.where(wait_n > 0, wait_sq / wait_n - _mi ** 2, np.nan)
            out["wait_mean_interval"] = _mi
            out["wait_mean_interval_gated"] = _mig
            out["wait_sem_interval"] = np.sqrt(np.maximum(_var, 0.0)
                                               / np.maximum(wait_n, 1.0))
        out["wait_events"] = wait_ev
        out["wait_events_gated"] = wait_ev_g
        out["wait_occupancy"] = occ
        out["wait_occupancy_gated"] = occ_g
        out["wait_n"] = wait_n
        out["wait_n_gated"] = wait_n_g
        out["wait_t_gate"] = float(t_occ_gate)
        # FREE BONUS, and a better spectrum than any single snapshot: the
        # occupancy integral IS the time-averaged number per bin.  Dividing by
        # the elapsed time and the bin width gives a dN/dm averaged over the
        # whole run rather than sampled at one instant, so it is smooth where
        # the snapshots are ragged.  Two versions, ungated and gated; on a warm
        # start the gated one is the physical spectrum.
        _T = float(t_phys)
        _Tg = (float(t_phys) - t_occ_gate) if np.isfinite(t_occ_gate) else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            out["dndm_time_avg"] = occ / max(_T, 1e-300) / widths / V
            out["dndm_time_avg_gated"] = occ_g / _Tg / widths / V if _Tg and _Tg > 0 \
                else np.full(B, np.nan)
        out["wait_t_span"] = _T
        out["wait_t_span_gated"] = float(_Tg)
        if float(wait_ev.sum()) <= 0.0:
            warnings.warn(
                "the waiting law was requested but no stay ever ended: "
                "wait_mean is all nan.  Either the run accepted no events, or "
                "track_waiting was toggled after the fact.",
                RuntimeWarning, stacklevel=2)

    out["meta"] = {
        "process": process,
        "system": system,
        "kernel": getattr(kernel, "__name__", "kernel"),
        "lambda": KERNEL_LAMBDA.get(getattr(kernel, "__name__", ""), None),
        "ic": {"m": m_init, "N": N0},
        # [v5] the warm start.  `seeded` is the one flag every downstream plot
        # should read before it believes anything about ancestry.
        "engine": "BF_warm_start_v5",
        "seeded": bool(seeded),
        "seed": seed_info,
        # [v5] происхождение прогона.  Перезапуски составляются, и t_origin --
        # физическое время, накопленное во ВСЕХ предыдущих прогоновах, так что
        # t_origin + final_t_phys есть полный возраст коробки.
        "restarted_from": restart_from,
        "t_origin": float(t_origin),
        "track_waiting": bool(use_wait),
        "injection_rate": float(injection_rate),
        "injection_mass": float(injection_mass),
        "sink_mass": None if sink_mass is None else float(sink_mass),
        "min_frag_mass": None if min_frag_mass is None else float(min_frag_mass),
        "frag_min_ratio": float(frag_min_ratio),
        "frag_split_width": float(frag_split_width),   
        "frag_age_rule": frag_age_rule,     
        "age_rule": age_rule,
        "track_generations": bool(use_gen),
        "gen_max": int(G_MAX),
        "weight": w,
        "volume": V,
        "clock": "dt = 2*V/(w*R), R over ordered pairs",
        "resampling": "none - baseline variant, w constant",
        "injection_scheme": "deterministic residual (no Poisson noise)",
        "stop_reason": stop_reason,
        "max_stall_tries": int(max_stall_tries),
        # [v2] both sink criteria recorded, so a saved run says what gated it
        "iso_start_sink": int(iso_start_sink),
        "stop_sink_events": None if stop_sink_events is None else float(stop_sink_events),
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


def state_of(run, source=None):
    """Build the ic={'state': ...} dict from a finished (or loaded) run.

    Everything needed to continue the run exactly where it stopped: the live
    masses, their generation counters, the entry time of each one's ancestor,
    and the family clock.  `t_end` and `t_origin` travel with it so the time
    offset of note [15] composes across arbitrarily many legs.

    Raises rather than guesses if the particle arrays are absent -- a run saved
    with drop_particles=True cannot be continued, and finding that out here is
    much better than finding it out from a silently wrong age axis.
    """
    need = ("final_mass", "final_gen", "final_gen_time", "final_inj_time")
    missing = [k for k in need if k not in run]
    if missing:
        raise KeyError(
            "this run cannot be continued: %s absent.  Save it with "
            "add_last_run(..., drop_particles=False), which is not the default "
            "because the arrays are one float per live particle." % ", ".join(missing))
    return {
        "mass": np.asarray(run["final_mass"], float),
        "gen": np.asarray(run["final_gen"]).astype(np.int64),
        "gen_time": np.asarray(run["final_gen_time"], float),
        "inj_time": np.asarray(run["final_inj_time"], float),
        "t_end": float(run["final_t_phys"]),
        "t_origin": float(run["meta"].get("t_origin", 0.0)) if "meta" in run else 0.0,
        "source": source if source is not None else (
            run["meta"].get("saved_as") if "meta" in run else None),
    }


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