# -*- coding: utf-8 -*-
"""
FQL-AGLEACH v3 simulator (revision).

Differences from FQL_AGLEACH_v2.py
----------------------------------
* Same protocol logic (grid, 27-rule Glorennec-Jouffe FQL, 5-term reward,
  P75 eligibility, utility U), compiled with numba for speed.
* Bug fix: an aggregated CH->BS packet is counted as delivered only if the CH
  actually has the energy to send it.  Throughput = number of sensor readings
  that reach the BS (a CH packet carries its own reading + one per member).
* Control-plane energy is charged to every protocol:
    - centralised protocols (Grid-LEACH, AGLEACH, k_opt-Grid, FQL-AGLEACH):
      16-bit residual-energy field piggy-backed on every data packet and a
      CH-assignment broadcast from the BS (64 + 16*k bits) received by all
      alive nodes;
    - distributed protocols (LEACH, Fuzzy-LEACH): CH advertisement (200 bit,
      radius = farthest alive node), join request (200 bit), TDMA schedule
      (200 bit, radius = farthest member).
* Additional metrics: LND, delivered readings, control-energy share,
  Jain fairness, CH-role fairness, member re-association rate.
* Generalised number of membership functions (nmf) for the sensitivity study.
"""
import math
import numpy as np
from numba import njit

SIDE = 100.0
BSX, BSY = 50.0, 50.0
E0 = 1.0
E_ELEC = 50e-9
E_FS = 100e-12
E_MP = 0.0013e-12
E_DA = 5e-9
D0 = math.sqrt(E_FS / E_MP)
PACKET = 4000.0
B_STAT = 16.0
B_CTRL = 200.0
P_LEACH = 0.1
D_MAX = math.hypot(50.0, 50.0)

# protocol codes
LEACH, FUZZY, GRID, AGL, KOPTGRID, FQL, FQL2, HEUR, FQL3 = 0, 1, 2, 3, 4, 5, 6, 7, 8
# grid modes
G_FIXED, G_LINEAR, G_KOPT, G_TILE = 0, 1, 2, 3

# default FQL parameter vector
P_NAMES = ["alpha", "gamma", "eps0", "epsmin", "epsfrac",
           "w1", "w2", "w3", "w4", "w5",
           "l1", "l2", "l3", "l4",
           "pct", "C", "nmf", "warm", "grid_mode", "ctrl",
           "einput", "eta_ref", "dens_r", "kappa", "bsx", "bsy", "tau"]
DEFAULT = dict(alpha=0.10, gamma=0.90, eps0=0.15, epsmin=0.01, epsfrac=1000.0,
               w1=0.30, w2=0.20, w3=0.20, w4=0.15, w5=1.00,
               l1=0.50, l2=0.45, l3=0.02, l4=0.03,
               pct=75.0, C=1.5, nmf=3.0, warm=1.0, grid_mode=2.0, ctrl=1.0,
               einput=0.0, eta_ref=2000.0, dens_r=0.5, kappa=1.0, bsx=50.0, bsy=50.0, tau=-1.0)


def pvec(**kw):
    d = dict(DEFAULT)
    d.update(kw)
    return np.array([d[k] for k in P_NAMES], dtype=np.float64)


def base_grid_size(n):
    if n <= 120:
        return 40.0
    gs = SIDE / math.sqrt(n / 12.0)
    return max(12.0, min(40.0, gs))


def make_nodes(n, seed, dep="uniform"):
    rng = np.random.default_rng(seed)
    if dep == "uniform":
        xs = rng.uniform(0, SIDE, n)
        ys = rng.uniform(0, SIDE, n)
        return xs, ys
    # clustered: 70% of nodes around 5 random hot-spots (sigma 8 m), 30% uniform
    k = 5
    cx = rng.uniform(10, SIDE - 10, k); cy = rng.uniform(10, SIDE - 10, k)
    nc = int(round(0.7 * n))
    lab = rng.integers(0, k, nc)
    xs = np.concatenate([cx[lab] + rng.normal(0, 8, nc), rng.uniform(0, SIDE, n - nc)])
    ys = np.concatenate([cy[lab] + rng.normal(0, 8, nc), rng.uniform(0, SIDE, n - nc)])
    return np.clip(xs, 0, SIDE - 1e-6), np.clip(ys, 0, SIDE - 1e-6)


# ---------------------------------------------------------------- radio
@njit(cache=True)
def e_tx(d, k):
    if d < D0:
        return k * E_ELEC + k * E_FS * d * d
    return k * E_ELEC + k * E_MP * d * d * d * d


@njit(cache=True)
def tri(x, a, b, c):
    if x <= a or x >= c:
        if (a == b and x <= a) or (b == c and x >= c):
            return 1.0
        return 0.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (c - x) / (c - b) if c > b else 1.0


@njit(cache=True)
def fuzzy_chance(e, d):
    el = tri(e, 0., 0., .5); em = tri(e, 0., .5, 1.); eh = tri(e, .5, 1., 1.)
    dn = tri(d, 0., 0., .5); dm = tri(d, 0., .5, 1.); df = tri(d, .5, 1., 1.)
    ws = np.array([eh*dn, eh*dm, eh*df, em*dn, em*dm, em*df, el*dn, el*dm, el*df])
    vs = np.array([.9, .8, .6, .6, .5, .35, .25, .15, .05])
    return (ws * vs).sum() / (ws.sum() + 1e-12)


@njit(cache=True)
def mf(x, nmf, out):
    # unity-partition triangular MFs with centres k/(nmf-1)
    s = nmf - 1
    for k in range(nmf):
        v = 1.0 - abs(x * s - k)
        out[k] = v if v > 0.0 else 0.0


@njit(cache=True)
def firing(e, d, r, nmf, w, me, md, mr):
    mf(min(max(e, 0.), 1.), nmf, me)
    mf(min(max(d, 0.), 1.), nmf, md)
    mf(min(max(r, 0.), 1.), nmf, mr)
    idx = 0
    s = 0.0
    for i in range(nmf):
        for j in range(nmf):
            for k in range(nmf):
                v = me[i] * md[j] * mr[k]
                w[idx] = v
                s += v
                idx += 1
    if s > 0:
        for q in range(idx):
            w[q] /= s


@njit(cache=True)
def pct_linear(v, p):
    v = np.sort(v)
    m = v.shape[0]
    pos = p / 100.0 * (m - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, m - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


# ---------------------------------------------------------------- core
@njit(cache=True)
def simulate(proto, xs, ys, rounds, seed, P, g0, qinit):
    np.random.seed(seed)
    n = xs.shape[0]
    E = np.full(n, E0)
    bsx = P[24]; bsy = P[25]; tau = P[26]
    dbs = np.sqrt((xs - bsx) ** 2 + (ys - bsy) ** 2)
    dmax = 0.0
    for cx_ in (0.0, SIDE):
        for cy_ in (0.0, SIDE):
            dd = math.hypot(cx_ - bsx, cy_ - bsy)
            if dd > dmax:
                dmax = dd

    alpha, gamma, eps0, epsmin, epsfrac = P[0], P[1], P[2], P[3], P[4]
    w1, w2, w3, w4, w5 = P[5], P[6], P[7], P[8], P[9]
    l1, l2, l3, l4 = P[10], P[11], P[12], P[13]
    pct, C = P[14], P[15]
    nmf = int(P[16])
    warm = P[17] > 0.5
    gmode = int(P[18])
    ctrl = P[19] > 0.5
    einput = int(P[20]); eta_ref = P[21]; dens_r = P[22]; kappa = P[23]
    if proto == GRID:
        gmode = G_FIXED
    elif proto == AGL:
        gmode = G_LINEAR
    elif proto == KOPTGRID and gmode != G_TILE:
        gmode = G_KOPT
    central = proto >= GRID

    # FQL state
    nr = nmf * nmf * nmf
    q = np.zeros((nr, 2))
    if warm:
        idx = 0
        for i in range(nmf):
            for j in range(nmf):
                for k in range(nmf):
                    ce = i / (nmf - 1.0)
                    cd = j / (nmf - 1.0)
                    q[idx, 1] = 0.30 * ce - 0.10 * cd
                    q[idx, 0] = 0.10
                    idx += 1
    if proto == FQL2:
        q[:, :] = 0.0
        if warm:
            idx = 0
            for i in range(nmf):
                for j in range(nmf):
                    for k in range(nmf):
                        q[idx, 1] = i / (nmf - 1.0)
                        idx += 1
    if proto == FQL3:
        q[:, :] = 0.0
        if warm:
            q[:, 1] = 0.1
    if qinit.shape[0] == nr:
        q[:, :] = qinit
    me = np.zeros(nmf); md = np.zeros(nmf); mr = np.zeros(nmf)
    dens = np.zeros(n)
    Ecell = np.zeros(n)
    gapN = np.zeros(n)
    W = np.zeros((n, nr))
    eps = eps0

    # outputs
    o_alive = np.zeros(rounds, np.int64)
    o_avgE = np.zeros(rounds)
    o_ch = np.zeros(rounds, np.int64)
    o_deliv = np.zeros(rounds)
    o_ctrlE = np.zeros(rounds)
    o_totE = np.zeros(rounds)
    o_reass = np.zeros(rounds)
    o_jain = np.zeros(rounds)
    o_G = np.zeros(rounds)
    o_rew = np.zeros(rounds)
    o_div = np.zeros(rounds)
    o_xd = np.zeros(rounds); o_xb = np.zeros(rounds); o_xe = np.zeros(rounds)
    ch_count = np.zeros(n)
    ch_count_hnd = np.zeros(n)
    hnd_seen = False

    last_ch = np.full(n, -1000000)
    prev_head = np.full(n, -1)
    head = np.full(n, -1)
    is_ch = np.zeros(n, np.bool_)
    act = np.zeros(n, np.int64)
    elig = np.zeros(n, np.bool_)
    q1v = np.zeros(n)
    Epre = np.zeros(n)
    rhoN = np.zeros(n)
    cellidx = np.full(n, -1)
    dec_node = np.zeros(n, np.int64)
    dec_a = np.zeros(n, np.int64)
    dec_adv = np.zeros(n)
    ndec = 0
    G = g0

    for r in range(rounds):
        alive = np.where(E > 0)[0]
        na = alive.shape[0]
        if na == 0:
            o_G[r] = G
            continue
        e_start = E.sum()
        egl = 0.0
        for i in alive:
            egl += E[i]
        egl /= (na * E0)
        Epre[:] = E
        is_ch[:] = False
        chs = np.empty(0, np.int64)
        ndiv = 0
        ncell_eval = 0

        if proto == LEACH or proto == FUZZY:
            sel = np.zeros(na, np.bool_)
            period = int(1.0 / P_LEACH)
            th = P_LEACH / (1.0 - P_LEACH * (r % period))
            for t in range(na):
                i = alive[t]
                if proto == LEACH:
                    if r - last_ch[i] >= period and np.random.random() < th:
                        sel[t] = True
                else:
                    sc = fuzzy_chance(E[i] / E0, dbs[i] / dmax)
                    if np.random.random() < P_LEACH * 2.0 * sc / 0.5:
                        sel[t] = True
            chs = alive[sel]
            if chs.shape[0] == 0:
                best = alive[0]
                for i in alive:
                    if E[i] > E[best]:
                        best = i
                chs = np.array([best])
            for c in chs:
                last_ch[c] = r
        else:
            # ---------------- grid size
            if gmode == G_FIXED:
                G = g0
            elif gmode == G_LINEAR:
                G = min(max(g0, g0 * n / (na + 1e-9)), SIDE)
            else:
                dm = 0.0
                for i in alive:
                    dm += dbs[i]
                dm /= na
                k = max(1.0, C * (SIDE / dm) * math.sqrt(na / (2 * math.pi)))
                if gmode == G_TILE:
                    mt = max(1, int(math.floor(math.sqrt(k) + 0.5)))
                    G = SIDE / mt
                else:
                    G = min(max(8.0, SIDE / math.sqrt(k)), SIDE)
            nc = int(SIDE // G) + 1 if gmode != G_TILE else int(math.floor(SIDE / G + 0.5))
            cell = np.empty(na, np.int64)
            cnt = np.zeros(nc * nc, np.int64)
            for t in range(na):
                i = alive[t]
                cx = min(int(xs[i] // G), nc - 1); cy = min(int(ys[i] // G), nc - 1)
                cell[t] = cx * nc + cy
                cnt[cell[t]] += 1
            rho_max = cnt.max()
            order = np.argsort(cell, kind="mergesort")
            chl = [0]
            chl.pop()
            ndec = 0
            s0 = 0
            while s0 < na:
                cid = cell[order[s0]]
                s1 = s0
                while s1 < na and cell[order[s1]] == cid:
                    s1 += 1
                members = alive[order[s0:s1]]
                m = members.shape[0]
                emax = 0.0
                gl = members[0]
                for i in members:
                    if E[i] > emax:
                        emax = E[i]; gl = i
                if proto == FQL3:
                    rho = m / rho_max
                    c0 = gl
                    firing(E[c0] / E0, min(dbs[c0] / dmax, 1.0), rho, nmf,
                           W[c0], me, md, mr)
                    rhoN[c0] = rho
                    qv0 = 0.0; qv1 = 0.0
                    for z in range(nr):
                        qv0 += W[c0, z] * q[z, 0]
                        qv1 += W[c0, z] * q[z, 1]
                    if np.random.random() < eps:
                        a = np.random.randint(0, 2)
                    else:
                        a = 1 if qv1 >= qv0 else 0
                    for i in members:
                        cellidx[i] = ndec
                    dec_node[ndec] = c0
                    dec_a[ndec] = a
                    dec_adv[ndec] = qv1 - qv0
                    ndec += 1
                    if a == 1:
                        chl.append(c0)
                    ncell_eval += 1
                    if a == 0:
                        ndiv += 1
                elif proto == FQL2 or proto == HEUR:
                    thr = pct_linear(E[members], pct)
                    rad2 = (dens_r * G) ** 2
                    em = 0.0
                    for i in members:
                        em += E[i]
                    em /= m
                    for i in members:
                        Ecell[i] = em
                        gapN[i] = (emax - E[i]) / (emax + 1e-12)
                        c2 = 0
                        for j in members:
                            if j != i and (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2 <= rad2:
                                c2 += 1
                        dens[i] = c2 / (m - 1.0) if m > 1 else 1.0
                    # eligible list
                    nel = 0
                    for i in members:
                        elig[i] = E[i] >= thr
                        if elig[i]:
                            nel += 1
                    best = -1; bu = -1e18
                    explore = proto == FQL2 and np.random.random() < eps
                    pick = np.random.randint(0, nel) if explore else -1
                    k2 = 0
                    for i in members:
                        if not elig[i]:
                            continue
                        dh = min(dbs[i] / dmax, 1.0)
                        if proto == HEUR:
                            if tau >= 0:
                                if egl > tau:
                                    u = E[i] / (emax + 1e-12)
                                else:
                                    u = (l1 * E[i] / (emax + 1e-12) + l2 * dens[i]
                                         + l3 * (1.0 - dh))
                            else:
                                u = (l1 * E[i] / (emax + 1e-12) + l2 * dens[i]
                                     + l3 * (1.0 - dh))
                        else:
                            ein = E[i] / E0 if einput == 0 else E[i] / (emax + 1e-12)
                            rhoN[i] = dens[i]
                            firing(ein, dh, dens[i], nmf, W[i], me, md, mr)
                            u = 0.0
                            for z in range(nr):
                                u += W[i, z] * q[z, 1]
                        if explore:
                            if k2 == pick:
                                best = i
                        elif u > bu:
                            bu = u; best = i
                        k2 += 1
                    chl.append(best)
                    ncell_eval += 1
                    if best != gl:
                        ndiv += 1
                    if nel > 0:
                        md_ = 0.0; mb_ = 0.0
                        for i in members:
                            if elig[i]:
                                md_ += dens[i]; mb_ += dbs[i]
                        md_ /= nel; mb_ /= nel
                        o_xd[r] += dens[best] - md_
                        o_xb[r] += (dbs[best] - mb_) / G
                        o_xe[r] += E[best] / (emax + 1e-12)
                elif proto != FQL:
                    chl.append(gl)
                else:
                    rho = m / rho_max
                    thr = pct_linear(E[members], pct)
                    for i in members:
                        rhoN[i] = rho
                        firing(E[i] / E0, min(dbs[i] / dmax, 1.0), rho, nmf,
                               W[i], me, md, mr)
                        qv0 = 0.0; qv1 = 0.0
                        for z in range(nr):
                            qv0 += W[i, z] * q[z, 0]
                            qv1 += W[i, z] * q[z, 1]
                        q1v[i] = qv1
                        elig[i] = E[i] >= thr
                        if elig[i]:
                            if np.random.random() < eps:
                                act[i] = np.random.randint(0, 2)
                            else:
                                act[i] = 1 if qv1 > qv0 else 0
                        else:
                            act[i] = 0
                    # candidate set
                    best = -1; bu = -1e18
                    for pas in range(3):
                        for i in members:
                            ok = False
                            if pas == 0:
                                ok = elig[i] and act[i] == 1
                            elif pas == 1:
                                ok = elig[i]
                            else:
                                ok = True
                            if ok:
                                u = (l1 * q1v[i] + l2 * E[i] / (emax + 1e-12)
                                     - l3 * min(dbs[i] / dmax, 1.0) - l4 * rho)
                                if u > bu:
                                    bu = u; best = i
                        if best >= 0:
                            break
                    chl.append(best)
                    ncell_eval += 1
                    if best != gl:
                        ndiv += 1
                s0 = s1
            if proto == FQL3 and len(chl) == 0:
                bb = 0
                for t in range(ndec):
                    if dec_adv[t] > dec_adv[bb]:
                        bb = t
                chl.append(dec_node[bb])
                dec_a[bb] = 1
            chs = np.array(chl, dtype=np.int64)
        for c in chs:
            is_ch[c] = True
            ch_count[c] += 1
        nch = chs.shape[0]

        # ---------------- association (nearest CH)
        head[:] = -1
        for i in alive:
            if is_ch[i]:
                head[i] = i
                continue
            bd = 1e18; bc = -1
            for c in chs:
                d2 = (xs[i] - xs[c]) ** 2 + (ys[i] - ys[c]) ** 2
                if d2 < bd:
                    bd = d2; bc = c
            head[i] = bc
        nmem = np.zeros(n)
        sumd = np.zeros(n)
        maxd = np.zeros(n)
        for i in alive:
            if not is_ch[i]:
                c = head[i]
                d = math.sqrt((xs[i] - xs[c]) ** 2 + (ys[i] - ys[c]) ** 2)
                nmem[c] += 1; sumd[c] += d
                if d > maxd[c]:
                    maxd[c] = d
        # re-association rate
        cnt_re = 0
        for i in alive:
            if prev_head[i] >= 0 and prev_head[i] != head[i]:
                cnt_re += 1
        o_reass[r] = cnt_re / na
        for i in alive:
            prev_head[i] = head[i]

        # ---------------- control plane
        ectl = 0.0
        if ctrl:
            if central:
                bits = 64.0 + 16.0 * nch
                for i in alive:
                    c = min(bits * E_ELEC, E[i]); E[i] -= c; ectl += c
            else:
                for c in chs:
                    far = 0.0
                    for i in alive:
                        d = math.sqrt((xs[i] - xs[c]) ** 2 + (ys[i] - ys[c]) ** 2)
                        if d > far:
                            far = d
                    cost = e_tx(far, B_CTRL) + nmem[c] * B_CTRL * E_ELEC \
                        + e_tx(maxd[c], B_CTRL)
                    cost = min(cost, E[c]); E[c] -= cost; ectl += cost
                for i in alive:
                    if not is_ch[i] and E[i] > 0:
                        c = head[i]
                        d = math.sqrt((xs[i] - xs[c]) ** 2 + (ys[i] - ys[c]) ** 2)
                        cost = nch * B_CTRL * E_ELEC + e_tx(d, B_CTRL) + B_CTRL * E_ELEC
                        cost = min(cost, E[i]); E[i] -= cost; ectl += cost

        # ---------------- data plane
        extra = B_STAT if (ctrl and central) else 0.0
        got = np.zeros(n)
        dlv = np.zeros(n, np.bool_)
        for i in alive:
            if is_ch[i] or E[i] <= 0:
                continue
            c = head[i]
            d = math.sqrt((xs[i] - xs[c]) ** 2 + (ys[i] - ys[c]) ** 2)
            cost = e_tx(d, PACKET + extra)
            if E[i] >= cost and E[c] > 0:
                E[i] -= cost
                got[c] += 1
                dlv[i] = True
                ectl += e_tx(d, PACKET + extra) - e_tx(d, PACKET)
            else:
                E[i] = 0.0 if E[i] < cost else E[i] - cost
        deliv = 0.0
        for c in chs:
            if E[c] <= 0:
                continue
            m = got[c]
            cost = m * (PACKET + extra) * E_ELEC + (m + 1) * PACKET * E_DA \
                + e_tx(dbs[c], PACKET + extra * (m + 1))
            if E[c] >= cost:
                E[c] -= cost
                deliv += m + 1
                ectl += m * extra * E_ELEC + e_tx(dbs[c], PACKET + extra * (m + 1)) - e_tx(dbs[c], PACKET)
            else:
                E[c] = 0.0

        # ---------------- learning (Glorennec-Jouffe per-rule TD update)
        if proto == FQL:
            alive_after = 0
            for i in range(n):
                if E[i] > 0:
                    alive_after += 1
            ar = alive_after / n
            nopt = max(1, int(math.floor(0.05 * na + 0.5)))
            pen = abs(nch - nopt) / nopt
            rnet = w2 * ar - w4 * pen
            wn = np.zeros(nr)
            rs = 0.0
            for pas in range(2):
                for i in alive:
                    if (pas == 0) != is_ch[i]:
                        continue
                    if is_ch[i]:
                        a = 1
                        eratio = E[i] / (Epre[i] + 1e-12)
                        dbar = (sumd[i] / nmem[i]) / dmax if nmem[i] > 0 else 0.0
                        mort = 1.0 if E[i] <= 0 else 0.0
                        rew = w1 * eratio + w2 * ar - w3 * dbar - w4 * pen - w5 * mort
                        rs += rew
                    else:
                        a = act[i]
                        rew = rnet
                    firing(max(E[i], 0.0) / E0, min(dbs[i] / dmax, 1.0), rhoN[i],
                           nmf, wn, me, md, mr)
                    qn0 = 0.0; qn1 = 0.0; qsa = 0.0
                    for z in range(nr):
                        qn0 += wn[z] * q[z, 0]
                        qn1 += wn[z] * q[z, 1]
                        qsa += W[i, z] * q[z, a]
                    qnext = qn0 if qn0 > qn1 else qn1
                    delta = rew + gamma * qnext - qsa
                    for z in range(nr):
                        q[z, a] += alpha * delta * W[i, z]
            o_rew[r] = rs / max(nch, 1)
            frac = min(r / (epsfrac if epsfrac > 1.0 else epsfrac * rounds), 1.0)
            eps = eps0 + (epsmin - eps0) * frac
            o_div[r] = ndiv / max(ncell_eval, 1)

        if proto == FQL3:
            gE = np.zeros(ndec); gR = np.zeros(ndec); gD = np.zeros(ndec)
            share = np.zeros(n)
            for c in chs:
                if E[c] > 0 or got[c] > 0:
                    share[c] = e_tx(dbs[c], PACKET + extra * (got[c] + 1)) / (got[c] + 1)
            for i in alive:
                g = cellidx[i]
                if g < 0:
                    continue
                c = head[i]
                if is_ch[i]:
                    gE[g] += PACKET * E_DA + share[i]
                    if E[i] > 0:
                        gR[g] += 1
                else:
                    gE[g] += Epre[i] - E[i]
                    if dlv[i]:
                        gE[g] += PACKET * (E_ELEC + E_DA) + share[c]
                        if E[c] > 0:
                            gR[g] += 1
                if E[i] <= 0:
                    gD[g] += 1
            wn = np.zeros(nr)
            rs = 0.0
            for t in range(ndec):
                c0 = dec_node[t]
                a = dec_a[t]
                eta = gR[t] / (gE[t] + 1e-12) / eta_ref
                rew = w1 * eta - w5 * gD[t]
                rs += rew
                firing(max(E[c0], 0.0) / E0, min(dbs[c0] / dmax, 1.0), rhoN[c0], nmf,
                       wn, me, md, mr)
                qn0 = 0.0; qn1 = 0.0; qsa = 0.0
                for z in range(nr):
                    qn0 += wn[z] * q[z, 0]
                    qn1 += wn[z] * q[z, 1]
                    qsa += W[c0, z] * q[z, a]
                qnext = qn0 if qn0 > qn1 else qn1
                delta = rew + gamma * qnext - qsa
                for z in range(nr):
                    q[z, a] += alpha * delta * W[c0, z]
            o_rew[r] = rs / max(ndec, 1)
            frac = min(r / (epsfrac if epsfrac > 1.0 else epsfrac * rounds), 1.0)
            eps = eps0 + (epsmin - eps0) * frac
            o_div[r] = ndiv / max(ncell_eval, 1)
            cellidx[:] = -1

        if proto == FQL2:
            # cluster-level energy and deaths
            cE = np.zeros(n); cD = np.zeros(n)
            for i in alive:
                c = head[i]
                cE[c] += Epre[i] - E[i]
                if E[i] <= 0:
                    cD[c] += 1
            wn = np.zeros(nr)
            rs = 0.0
            for c in chs:
                deliv_c = got[c] + 1 if E[c] > 0 else 0.0
                eta = deliv_c / (cE[c] + 1e-12) / eta_ref
                rew = w1 * eta - kappa * egl * egl * gapN[c] - w5 * cD[c]
                rs += rew
                ein = max(E[c], 0.0) / E0 if einput == 0 else 1.0
                firing(ein, min(dbs[c] / dmax, 1.0), rhoN[c], nmf, wn, me, md, mr)
                qn = 0.0; qsa = 0.0
                for z in range(nr):
                    qn += wn[z] * q[z, 1]
                    qsa += W[c, z] * q[z, 1]
                delta = rew + gamma * qn - qsa
                for z in range(nr):
                    q[z, 1] += alpha * delta * W[c, z]
            o_rew[r] = rs / max(nch, 1)
            frac = min(r / (epsfrac if epsfrac > 1.0 else epsfrac * rounds), 1.0)
            eps = eps0 + (epsmin - eps0) * frac
            o_div[r] = ndiv / max(ncell_eval, 1)

        if (proto == FQL2 or proto == HEUR) and ncell_eval > 0:
            o_xd[r] /= ncell_eval; o_xb[r] /= ncell_eval; o_xe[r] /= ncell_eval

        # ---------------- bookkeeping
        nal = 0
        se = 0.0; se2 = 0.0
        for i in range(n):
            if E[i] > 0:
                nal += 1; se += E[i]; se2 += E[i] * E[i]
        o_alive[r] = nal
        o_avgE[r] = E.sum() / n
        o_ch[r] = nch
        o_deliv[r] = deliv
        o_ctrlE[r] = ectl
        o_totE[r] = e_start - E.sum()
        o_jain[r] = (se * se) / (nal * se2) if nal > 0 and se2 > 0 else 0.0
        o_G[r] = G
        if (not hnd_seen) and nal <= n / 2:
            hnd_seen = True
            ch_count_hnd[:] = ch_count
    if not hnd_seen:
        ch_count_hnd[:] = ch_count
    return (o_alive, o_avgE, o_ch, o_deliv, o_ctrlE, o_totE, o_reass, o_jain,
            o_G, o_rew, o_div, ch_count_hnd, q, o_xd, o_xb, o_xe)
