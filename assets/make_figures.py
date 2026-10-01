"""
Generate the figures used in the README.

    python assets/make_figures.py            # everything (simulation results are cached)
    python assets/make_figures.py --refresh  # recompute cached simulations

Every figure comes from real simulator runs and the real refinement code.
Outputs land in assets/ with -light / -dark variants for GitHub's themes.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
CACHE = ASSETS / "_cache"
sys.path.insert(0, str(ROOT))

# Keep a handle on the original (field-returning) solver before src.bo_refiner patches it
from simulator import Heat2D  # noqa: E402

ORIGINAL_SOLVE = Heat2D.solve

from utils import make_training_dataset  # noqa: E402

LX, LY, NX, NY, DT = 2.0, 1.0, 100, 50, 0.004


# ==============================================================================
# Simulation data
# ==============================================================================
def make_sample(n_sources, n_sensors, seed, kappa=0.05, nt=1000, noise=0.1, bc="dirichlet"):
    ds = make_training_dataset(
        Lx=LX, Ly=LY, nx=NX, ny=NY, num_samples=1, dt=DT,
        num_sources_sensors={(n_sources, n_sensors): 1.0}, q_range=(0.5, 2.0),
        noise_std={noise: 1.0}, nt={nt: 1.0}, kappa={kappa: 1.0}, bc=bc, T0=None, seed=seed,
    )
    return ds["samples"][0]


def simulate_fields(sample, store_every=10):
    """Full temperature fields over time (original simulator, unpatched)."""
    meta = sample["sample_metadata"]
    solver = Heat2D(LX, LY, NX, NY, kappa=meta["kappa"], bc=meta["bc"])
    times, Us = ORIGINAL_SOLVE(solver, DT, meta["nt"], T0=meta["T0"],
                               sources=sample["true_sources"], store_every=store_every)
    return np.asarray(times), np.asarray(Us)


def hero_data():
    sample = make_sample(2, 5, seed=7)
    times, Us = simulate_fields(sample, store_every=10)
    return {"sample": sample, "times": times, "Us": Us}


def refinement_data(seeds=(26, 27, 15, 20, 29, 38)):
    """Run the real BO pipeline on 1-source samples, logging every objective call."""
    from src.bo_refiner import BORefiner
    from src.inference import InferencePipeline

    log, current = [], {"id": -1}
    orig_refine, orig_obj = BORefiner.refine, BORefiner._objective
    orig_skopt, orig_scipy = BORefiner.refine_skopt, BORefiner.refine_scipy
    phase = {"name": "bo"}

    def skopt(self, *a, **k):
        phase["name"] = "bo"
        return orig_skopt(self, *a, **k)

    def scipy_(self, *a, **k):
        phase["name"] = "lbfgs"
        return orig_scipy(self, *a, **k)

    def refine(self, initial, *a, **k):
        current["id"] += 1
        log.append({"start": np.array(initial, dtype=float), "path": []})
        return orig_refine(self, initial, *a, **k)

    def objective(self, params, *a, **k):
        val = orig_obj(self, params, *a, **k)
        log[current["id"]]["path"].append((*params, val))
        log[current["id"]].setdefault("phase", []).append(phase["name"])
        return val

    BORefiner.refine, BORefiner._objective = refine, objective
    BORefiner.refine_skopt, BORefiner.refine_scipy = skopt, scipy_
    pipe = InferencePipeline(str(ROOT / "checkpoints/best.pt"), device="cpu",
                             n_search_candidates=4, n_calls=30, method="bo", n_jobs=1)

    best = None
    for seed in seeds:
        sample = make_sample(1, 4, seed=seed, kappa=0.1, nt=800)
        log.clear(); current["id"] = -1
        nn_pred = pipe.predict_nn(sample)
        cands = pipe.bo_refiner.generate_candidates_adaptive(
            nn_pred, sample, n_candidates=3, total_budget=4 * 30, coarse_calls=2, seed=42, n_jobs=1)
        final = pipe.bo_refiner.filter_distinct(cands, tau=0.2, n_max=3)
        losses = [next(l for s, l in cands if s is f) for f in final]
        print(f"  seed {seed}: {len(final)} distinct candidates, losses {np.round(losses, 4)}")
        if len(final) == 3 and max(losses) < 0.25:
            best = {"sample": sample, "nn_pred": nn_pred, "runs": [dict(r) for r in log],
                    "final": [np.array(f) for f in final], "final_loss": losses}
            break
        if best is None:
            best = {"sample": sample, "nn_pred": nn_pred, "runs": [dict(r) for r in log],
                    "final": [np.array(f) for f in final], "final_loss": losses}
    BORefiner.refine, BORefiner._objective = orig_refine, orig_obj
    BORefiner.refine_skopt, BORefiner.refine_scipy = orig_skopt, orig_scipy

    # Profile misfit over (x, y): T is affine in q, so q* has a closed form per point
    from joblib import Parallel, delayed
    sample = best["sample"]
    meta, Y = sample["sample_metadata"], sample["Y_noisy"]
    xs, ys = np.linspace(0.05 * LX, 0.95 * LX, 96), np.linspace(0.05 * LY, 0.95 * LY, 48)

    def sensor_series(x, y, q):
        sys.path.insert(0, str(ROOT))
        from src.bo_refiner import Heat2D as PatchedHeat2D  # applies the Numba patch in workers
        solver = PatchedHeat2D(LX, LY, NX, NY, kappa=meta["kappa"], bc=meta["bc"])
        src = [] if q == 0 else [{"x": x, "y": y, "q": q, "on": (0.0, meta["nt"] * DT)}]
        return solver.solve(dt=DT, nt=meta["nt"], T0=meta["T0"], sources=src, sensor_xy=sample["sensors_xy"])

    base = sensor_series(0.0, 0.0, 0)
    unit = Parallel(n_jobs=-1)(delayed(sensor_series)(x, y, 1.0) for y in ys for x in xs)
    n = min(base.shape[0], Y.shape[0])
    rmse = np.zeros((len(ys), len(xs)))
    for k, U in enumerate(unit):
        g = U[:n] - base[:n]
        r = Y[:n] - base[:n]
        q = np.clip((g * r).sum() / (g * g).sum(), 0.5, 2.0)
        rmse[k // len(xs), k % len(xs)] = np.sqrt(np.mean((base[:n] + q * g - Y[:n]) ** 2))
    best.update({"xs": xs, "ys": ys, "rmse": rmse})

    # Predicted sensor curves for each final candidate
    best["fits"] = [sensor_series(*f[0])[:n] for f in best["final"]]
    return best


def cached(name, fn, refresh):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.pkl"
    if path.exists() and not refresh:
        return pickle.load(open(path, "rb"))
    print(f"Computing {name} ...")
    data = fn()
    pickle.dump(data, open(path, "wb"))
    return data


# ==============================================================================
# Styling: one token set per GitHub theme (light / dark)
# ==============================================================================
THEMES = {
    "light": dict(page="#ffffff", fg="#0b0b0b", fg2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
                  ramp=["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(page="#0d1117", fg="#ffffff", fg2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
                 ramp=["#0d1626", "#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"]),
}


def style_axes(ax, t, grid=True):
    import matplotlib.pyplot as plt  # noqa: F401
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelcolor=t["fg2"], labelsize=10)
    ax.xaxis.label.set_color(t["fg2"]); ax.yaxis.label.set_color(t["fg2"])
    ax.title.set_color(t["fg"])
    if grid:
        ax.grid(True, color=t["grid"], linewidth=0.8)
        ax.set_axisbelow(True)


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.titlepad": 10})
    return plt


# ==============================================================================
# 1. Banner
# ==============================================================================
def fig_banner(hero):
    plt = setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap
    bg = "#000004"
    U = hero["Us"][-1].T
    s = hero["sample"]
    fig = plt.figure(figsize=(16, 4.4), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.45, 0, 0.55, 1])
    ax.imshow(U, origin="lower", extent=[0, LX, 0, LY], cmap="inferno", aspect="auto",
              interpolation="bicubic", vmin=0, vmax=U.max() * 0.95)
    X, Y = np.meshgrid(np.linspace(0, LX, NX), np.linspace(0, LY, NY))
    ax.contour(X, Y, U, levels=10, colors="white", linewidths=0.6, alpha=0.18)
    sx, sy = s["sensors_xy"][:, 0], s["sensors_xy"][:, 1]
    ax.scatter(sx, sy, s=90, facecolors="none", edgecolors="white", linewidths=1.6, zorder=3)
    ax.scatter(sx, sy, s=10, color="white", zorder=3)
    # Fade the field into the background on its left edge
    fade = np.linspace(1, 0, 256) ** 1.6
    ax.imshow(np.tile(fade, (2, 1)), extent=[0, LX * 0.45, 0, LY], aspect="auto",
              cmap=LinearSegmentedColormap.from_list("f", [(0, 0, 0.016, 0), (0, 0, 0.016, 1)]),
              vmin=0, vmax=1, zorder=4)
    ax.set_xlim(0, LX); ax.set_ylim(0, LY); ax.axis("off")
    fig.text(0.04, 0.60, "Heat Signature Zero", color="white", fontsize=46, fontweight="bold")
    fig.text(0.042, 0.45, "Recovering hidden heat sources from a handful of sensors",
             color="#c3c2b7", fontsize=17)
    fig.text(0.042, 0.33, "Transformer prior  ·  simulator-in-the-loop Bayesian optimization  ·  diversity-aware search",
             color="#ec835a", fontsize=12.5)
    fig.text(0.042, 0.14, "ThinkOnward Challenge  ·  2025–26", color="#898781", fontsize=11.5)
    fig.savefig(ASSETS / "banner.png", facecolor=bg)
    plt.close(fig)


# ==============================================================================
# 2. Problem animation: the plate heats up while the sensors record
# ==============================================================================
def fig_problem_gif(hero, theme):
    plt = setup_mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    t = THEMES[theme]
    s, Us, times = hero["sample"], hero["Us"], hero["times"]
    Yn = s["Y_noisy"]
    ty = np.arange(Yn.shape[0]) * DT * (s["sample_metadata"]["nt"] / (Yn.shape[0] - 1))
    M = s["sensors_xy"].shape[0]
    vmax = Us.max() * 0.9

    fig = plt.figure(figsize=(10, 3.9), dpi=100, facecolor=t["page"])
    axf = fig.add_axes([0.035, 0.19, 0.45, 0.577])
    axs = fig.add_axes([0.575, 0.19, 0.39, 0.577])
    im = axf.imshow(Us[0].T, origin="lower", extent=[0, LX, 0, LY], cmap="inferno",
                    vmin=0, vmax=vmax, interpolation="bicubic", aspect="equal")
    for k, (x, y) in enumerate(s["sensors_xy"]):
        axf.scatter(x, y, s=70, color=t["series"][k], edgecolors="white", linewidths=1.6, zorder=3)
        axf.annotate(f"S{k+1}", (x, y), xytext=(7, 5), textcoords="offset points", color="white",
                     fontsize=10, fontweight="bold")
    for src in s["true_sources"]:
        axf.scatter(src["x"], src["y"], marker="x", s=55, color="white", linewidths=2, zorder=3)
    axf.set_xticks([]); axf.set_yticks([])
    for sp in axf.spines.values():
        sp.set_color(t["axis"])
    axf.set_title("Temperature field  (× = hidden sources)", color=t["fg"])
    clock = axf.text(0.02, 0.05, "", transform=axf.transAxes, color="white", fontsize=10)

    style_axes(axs, t)
    axs.set_xlim(0, ty[-1]); axs.set_ylim(min(0, Yn.min()) - 0.2, Yn.max() * 1.08)
    axs.set_xlabel("time"); axs.set_ylabel("temperature")
    axs.set_title("What the sensors record (noisy)", color=t["fg"])
    lines = [axs.plot([], [], color=t["series"][k], lw=1.6)[0] for k in range(M)]
    labels = [axs.text(0, 0, "", color=t["fg2"], fontsize=9.5, fontweight="bold") for _ in range(M)]

    n_frames = len(Us)
    idx = list(range(0, n_frames, 2)) + [n_frames - 1] * 12

    def update(i):
        im.set_data(Us[i].T)
        tt = times[i]
        clock.set_text(f"t = {tt:.2f}")
        m = ty <= tt + 1e-9
        for k in range(M):
            lines[k].set_data(ty[m], Yn[m, k])
        if m.sum() > 1:
            ends = np.array([Yn[m, k][-8:].mean() for k in range(M)])
            order, gap = np.argsort(ends), 0.075 * (axs.get_ylim()[1] - axs.get_ylim()[0])
            pos = ends.copy()
            for a, b in zip(order[:-1], order[1:]):  # push labels apart, bottom-up
                pos[b] = max(pos[b], pos[a] + gap)
            for k in range(M):
                labels[k].set_position((ty[m][-1] + ty[-1] * 0.012, pos[k]))
                labels[k].set_va("center")
                labels[k].set_text(f"S{k+1}")
        return [im, clock, *lines, *labels]

    anim = FuncAnimation(fig, update, frames=idx, blit=False)
    anim.save(ASSETS / f"problem-{theme}.gif", writer=PillowWriter(fps=14),
              savefig_kwargs={"facecolor": t["page"]})
    plt.close(fig)


# ==============================================================================
# 3. Refinement: misfit landscapes + real optimizer trajectories
# ==============================================================================
def chaikin(pts, iters=3):
    """Corner-cutting smoothing for display; keeps both endpoints fixed."""
    pts = np.asarray(pts, dtype=float)
    for _ in range(iters):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([pts[:1], np.column_stack([q, r]).reshape(-1, pts.shape[1]), pts[-1:]])
    return pts


def best_so_far(path):
    p = np.asarray(path)
    keep, best = [], np.inf
    for row in p:
        if row[-1] < best:
            best = row[-1]; keep.append(row)
    return np.asarray(keep)


def fig_refinement(d, theme):
    plt = setup_mpl()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    t = THEMES[theme]
    cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
    s = d["sample"]
    finals = [f[0] for f in d["final"]]
    true = s["true_sources"][0]

    def nearest_final(p):
        dists = [np.linalg.norm((p[:3] - f) / [LX, LY, 2.0]) for f in finals]
        return int(np.argmin(dists))

    fig = plt.figure(figsize=(12.5, 4.9), dpi=150)
    fig.patch.set_alpha(0)
    top, h = 0.86, 0.66
    ax1 = fig.add_axes([0.045, top - h, 0.47, h], anchor="NW")
    cax = fig.add_axes([0.045, 0.105, 0.2, 0.022])

    # (a) x-y map, q profiled out
    norm = LogNorm(vmin=0.1, vmax=2.0)
    from scipy.ndimage import zoom
    fine = np.exp(zoom(np.log(d["rmse"]), 4, order=3))  # smooth (cubic) upsampling in log space
    ext = [d["xs"][0], d["xs"][-1], d["ys"][0], d["ys"][-1]]
    cs = ax1.imshow(fine, origin="lower", extent=ext, cmap=cmap, norm=norm, aspect="auto", interpolation="bicubic")
    fx, fy = np.linspace(*ext[:2], fine.shape[1]), np.linspace(*ext[2:], fine.shape[0])
    ax1.contour(fx, fy, fine, levels=[0.12, 0.15, 0.2, 0.3, 0.5, 1.0], colors=t["page"], linewidths=0.6, alpha=0.55)
    style_axes(ax1, t, grid=False)
    ax1.set_xlim(d["xs"][0], d["xs"][-1]); ax1.set_ylim(d["ys"][0], d["ys"][-1]); ax1.set_aspect("equal")
    ax1.set_xlabel("x", labelpad=2); ax1.set_ylabel("y")
    ax1.set_title("(a) Misfit over the plate, optimizer paths", color=t["fg"])
    for k, (x, y) in enumerate(s["sensors_xy"]):
        ax1.scatter(x, y, marker="s", s=46, color=t["page"], edgecolors=t["fg"], linewidths=1.4, zorder=5)
        ax1.annotate(f"S{k+1}", (x, y), xytext=(6, -12), textcoords="offset points", color=t["fg"], fontsize=9)
    for run in d["runs"]:
        path, ph = np.asarray(run["path"]), np.asarray(run["phase"])
        c = t["series"][nearest_final(best_so_far(path)[-1])]
        st = run["start"][0]
        bo, lb = path[ph == "bo"], path[ph == "lbfgs"]
        hop = bo[np.argmin(bo[:, -1])] if len(bo) else st
        # BO: one global hop from the start to its best sample (dashed)
        if np.hypot(*(hop[:2] - st[:2])) > 0.02:
            ax1.annotate("", xy=hop[:2], xytext=st[:2], zorder=4,
                         arrowprops=dict(arrowstyle="-|>,head_width=0.18,head_length=0.45", color=c, lw=1.1,
                                         ls=(0, (3, 2.5)), alpha=0.85, shrinkA=4, shrinkB=1))
        # L-BFGS-B: local descent (solid)
        desc = best_so_far(np.vstack([hop[None], lb])) if len(lb) else hop[None]
        desc = chaikin(desc[:, :2])
        ax1.plot(desc[:, 0], desc[:, 1], color=c, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=5)
        ax1.scatter(st[0], st[1], s=55, facecolors=t["page"], edgecolors=c, linewidths=1.8, zorder=6)
    ax1.scatter(true["x"], true["y"], marker="*", s=230, color=t["fg"], edgecolors=t["page"], linewidths=1, zorder=8)
    for k, (f, L) in enumerate(zip(finals, d["final_loss"])):
        ax1.scatter(f[0], f[1], s=80, color=t["series"][k], edgecolors=t["page"], linewidths=1.6, zorder=7)
        right = f[0] >= true["x"]
        ax1.annotate(f"#{k+1}  q={f[2]:.2f}", (f[0], f[1]), xytext=(10 if right else -10, 0),
                     textcoords="offset points", ha="left" if right else "right", va="center",
                     color=t["fg"], fontsize=9.5, fontweight="bold", zorder=9,
                     bbox=dict(boxstyle="round,pad=0.2", fc=t["page"], ec="none", alpha=0.8))
    cb = fig.colorbar(cs, cax=cax, orientation="horizontal")
    cb.set_ticks([0.1, 0.2, 0.5, 1.0, 2.0]); cb.set_ticklabels(["0.1", "0.2", "0.5", "1", "2"])
    cb.outline.set_visible(False)
    cb.ax.minorticks_off()
    cb.ax.tick_params(colors=t["muted"], labelcolor=t["fg2"], labelsize=8.5)
    fig.text(0.255, 0.116, "RMSE with best q per point  (0.1 = noise floor)", color=t["fg2"], fontsize=9, va="center")
    fig.text(0.045, 0.012, "○ start   ⇢ BO global hop   — L-BFGS-B descent (smoothed)   ● final candidate   ★ true source   ■ sensor",
             color=t["fg2"], fontsize=9.5)

    # (b) small multiples: every candidate reproduces every sensor
    Y = s["Y_noisy"][: len(d["fits"][0])]
    tt = np.arange(len(Y)) * DT * s["sample_metadata"]["nt"] / (len(s["Y_noisy"]) - 1)
    M = Y.shape[1]
    cols, rows = 2, int(np.ceil(M / 2))
    x0, w, gx, gy = 0.60, 0.17, 0.055, 0.11
    gtop = top - 0.04
    hh = (gtop - (top - h) - gy * (rows - 1)) / rows
    fig.text(x0 - 0.04, top + 0.035, "(b) Three answers, one set of sensor readings",
             color=t["fg"], fontsize=12.5, fontweight="bold")
    for m in range(M):
        r, c = divmod(m, cols)
        ax = fig.add_axes([x0 + c * (w + gx), gtop - (r + 1) * hh - r * gy, w, hh])
        style_axes(ax, t)
        ax.plot(tt, Y[:, m], color=t["muted"], lw=0.7, alpha=0.55)
        for k, fit in enumerate(d["fits"]):
            ax.plot(tt, fit[:, m], color=t["series"][k], lw=1.8, ls=["-", (0, (5, 2)), (0, (1.5, 1.5))][k])
        ax.set_title(f"S{m+1}", fontsize=10.5, color=t["fg"], pad=4)
        ax.tick_params(labelsize=8.5)
        if r == rows - 1:
            ax.set_xlabel("time", fontsize=9.5, labelpad=2)
        else:
            ax.set_xticklabels([])
    lab = "   ".join(f"#{k+1}: L={L:.3f}" for k, L in enumerate(d["final_loss"]))
    fig.text(x0 - 0.04, 0.012, f"grey = observed (noisy)    {lab}", color=t["fg2"], fontsize=9.5)
    fig.savefig(ASSETS / f"refinement-{theme}.png", transparent=True)
    plt.close(fig)


# ==============================================================================
# 4. Score trade-off: when does one more candidate pay off?
# ==============================================================================
def fig_score(theme):
    plt = setup_mpl()
    t = THEMES[theme]
    L1, lam, nmax = 0.1, 0.3, 3
    a = 1 / (1 + L1)
    L = np.linspace(0, 1.0, 400)
    b = 1 / (1 + L)
    P = {1: np.full_like(L, a + lam / nmax),
         2: (a + b) / 2 + lam * 2 / nmax,
         3: (a + 2 * b) / 3 + lam * 3 / nmax}
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
    fig.patch.set_alpha(0)
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.15, top=0.86)
    style_axes(ax, t)
    names = {1: "best one only", 2: "+1 extra", 3: "+2 extras"}
    for n in (1, 2, 3):
        ax.plot(L, P[n], color=t["series"][n - 1], lw=2, label=names[n], zorder=3)
        ax.text(1.015, P[n][-1], names[n], color=t["fg2"], fontsize=10, va="center")
    # Break-even vs. submitting only the best one: P_n = P_1  <=>  1/(1+L) = a - n*lam/nmax
    for n, dx, dy, ha, msg in ((2, -8, -24, "right", "+1 extra pays off while L < {:.2f}"),
                               (3, 8, 16, "left", "+2 extras pay off while L < {:.2f}")):
        bstar = a - lam * n / nmax
        Ls = 1 / bstar - 1
        ax.axvline(Ls, color=t["muted"], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.scatter([Ls], [P[1][0]], s=55, color=t["series"][n - 1], edgecolors=t["page"], linewidths=2, zorder=5)
        ax.annotate(msg.format(Ls), (Ls, P[1][0]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    color=t["fg"], fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.0); ax.set_ylim(0.88, 1.3)
    ax.set_xlabel("RMSE of each extra candidate  (best candidate at L = 0.1)")
    ax.set_ylabel("sample score P")
    ax.set_title("When is it worth submitting more candidates?", color=t["fg"])
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    for txt in leg.get_texts():
        txt.set_color(t["fg2"])
    fig.savefig(ASSETS / f"score-tradeoff-{theme}.png", transparent=True)
    plt.close(fig)


# ==============================================================================
# 5. Diagrams (hand-laid SVG, one file per theme)
# ==============================================================================
SVG_THEMES = {
    "light": dict(fg="#1f2328", fg2="#59636e", card="#f6f8fa", border="#d1d9e0", line="#818b98",
                  accent="#eb6834", accent_bg="#fdf1eb", blue="#2a78d6", blue_bg="#eef5fd",
                  green="#1a7f37", green_bg="#ecf7ef"),
    "dark": dict(fg="#e6edf3", fg2="#9198a1", card="#151b23", border="#3d444d", line="#656c76",
                 accent="#f0834f", accent_bg="#2b1a12", blue="#4f94e8", blue_bg="#0f1d30",
                 green="#3fb950", green_bg="#0f2417"),
}


def _svg(w, h, body, t):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'font-family="-apple-system, BlinkMacSystemFont, \'Segoe UI\', Helvetica, Arial, sans-serif">\n'
            f'<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["line"]}"/></marker></defs>\n'
            f'{body}\n</svg>\n')


def _card(x, y, w, h, title, lines, t, kind="card", num=None):
    fill, stroke, tcol = {"card": (t["card"], t["border"], t["fg"]),
                          "nn": (t["blue_bg"], t["blue"], t["fg"]),
                          "sim": (t["accent_bg"], t["accent"], t["fg"]),
                          "out": (t["green_bg"], t["green"], t["fg"])}[kind]
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    tx = x + 16
    if num is not None:
        badge = t["line"] if kind == "card" else stroke
        out.append(f'<circle cx="{x + 24}" cy="{y + 26}" r="12" fill="{badge}"/>'
                   f'<text x="{x + 24}" y="{y + 30.5}" text-anchor="middle" font-size="13" font-weight="700" fill="{t["card"]}">{num}</text>')
        tx = x + 44
    out.append(f'<text x="{tx}" y="{y + 31}" font-size="15.5" font-weight="700" fill="{tcol}">{title}</text>')
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x + 16}" y="{y + 58 + i * 20}" font-size="13" fill="{t["fg2"]}">{ln}</text>')
    return "\n".join(out)


def _arrow(x1, y1, x2, y2, t, dashed=False, color=None):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or t["line"]}" stroke-width="1.8"{dash} '
            f'marker-end="url(#arr)"/>')


def fig_pipeline(theme):
    t = SVG_THEMES[theme]
    W, cw, ch, gap = 1080, 220, 118, 40
    xs = [20 + i * (cw + gap) for i in range(4)]
    y1, y2 = 20, 200
    b = []
    row1 = [("Sensor data", ["3–5 noisy time series", "+ positions, κ, boundary"], "card"),
            ("Transformer prior", ["one forward pass →", "initial (x, y, q) per source"], "nn"),
            ("Multi-start", ["grid coverage + physics", "heuristics + NN guess"], "card"),
            ("Coarse screening", ["every start scored on a", "cheap 50×25 grid"], "sim")]
    row2 = [("Diversity selection", ["greedy: good loss AND", "≥ τ apart from the rest"], "card"),
            ("Niching refinement", ["BO → L-BFGS-B; later runs", "repelled from found basins"], "sim"),
            ("Score-aware filter", ["keep a candidate only if", "it raises expected score"], "card"),
            ("≤ 3 candidates", ["accurate and mutually", "distinct source sets"], "out")]
    for i, (ti, ln, k) in enumerate(row1):
        b.append(_card(xs[i], y1, cw, ch, ti, ln, t, k, num=i + 1))
        if i < 3:
            b.append(_arrow(xs[i] + cw + 4, y1 + ch / 2, xs[i + 1] - 4, y1 + ch / 2, t))
    for i, (ti, ln, k) in enumerate(row2):
        b.append(_card(xs[i], y2, cw, ch, ti, ln, t, k, num=i + 5 if i < 3 else None))
        if i < 3:
            b.append(_arrow(xs[i] + cw + 4, y2 + ch / 2, xs[i + 1] - 4, y2 + ch / 2, t))
    # snake connector from end of row 1 to start of row 2
    mid = (y1 + ch + y2) / 2
    b.append(f'<path d="M{xs[3] + cw / 2},{y1 + ch + 4} V{mid} H{xs[0] + cw / 2} V{y2 - 5}" fill="none" '
             f'stroke="{t["line"]}" stroke-width="1.8" marker-end="url(#arr)"/>')
    # simulator bar (feeds stage 4 via the right margin, stage 6 directly)
    sy = 375
    rx_bar = xs[3] + cw + 30 - 20 + 12
    b.append(f'<rect x="20" y="{sy}" width="{rx_bar}" height="50" rx="10" fill="{t["accent_bg"]}" stroke="{t["accent"]}" stroke-width="1.5"/>')
    b.append(f'<text x="40" y="{sy + 31}" font-size="15" font-weight="700" fill="{t["fg"]}">Heat2D simulator</text>')
    b.append(f'<text x="182" y="{sy + 31}" font-size="14" fill="{t["fg2"]}">·  the official ADI solver, Numba-JIT accelerated  ·  '
             f'every candidate is judged by its forward RMSE</text>')
    b.append(_arrow(xs[1] + cw / 2, sy - 4, xs[1] + cw / 2, y2 + ch + 6, t, dashed=True, color=t["accent"]))
    rx = xs[3] + cw + 30
    b.append(f'<path d="M{rx},{sy} V{y1 + ch / 2} H{xs[3] + cw + 6}" fill="none" stroke="{t["accent"]}" '
             f'stroke-width="1.8" stroke-dasharray="5 4" marker-end="url(#arr)"/>')
    return _svg(W, 440, "\n".join(b), t)


def fig_architecture(theme):
    t = SVG_THEMES[theme]
    b = []
    # Inputs column
    inputs = [("Sensor time series", "1D CNN  (avg + max pool)"),
              ("Sensor position (x, y)", "MLP"),
              ("κ, boundary condition", "embedding + MLP"),
              ("Global stats", "max · mean · std · rise rate")]
    for i, (ti, sub) in enumerate(inputs):
        y = 20 + i * 78
        b.append(_card(20, y, 230, 62, ti, [], t, "card"))
        b.append(f'<text x="36" y="{y + 50}" font-size="12.5" fill="{t["fg2"]}">{sub}</text>')
        b.append(_arrow(254, y + 31, 296, 166, t))
    b.append(_card(300, 126, 150, 80, "Fuse", ["per-sensor token", "d = 256"], t, "card"))
    b.append(_arrow(454, 166, 496, 166, t))
    b.append(_card(500, 96, 190, 140, "Sensor encoder", ["self-attention", "across sensors", "× 3 layers", "(masked, any M)"], t, "nn"))
    b.append(_arrow(694, 166, 736, 166, t))
    b.append(_card(740, 96, 190, 140, "Source decoder", ["K = 3 learned queries", "cross-attend to", "sensor tokens", "× 2 layers"], t, "nn"))
    b.append(_arrow(835, 240, 835, 276, t))
    b.append(_card(740, 280, 190, 62, "Sigmoid heads", [], t, "out"))
    b.append(f'<text x="756" y="330" font-size="12.5" fill="{t["fg2"]}">bounded (x, y, q) × K</text>')
    b.append(f'<text x="500" y="290" font-size="13" fill="{t["fg2"]}">Loss: MSE + 0.5·L1 after</text>')
    b.append(f'<text x="500" y="310" font-size="13" fill="{t["fg2"]}">Hungarian matching</text>')
    b.append(f'<text x="500" y="330" font-size="13" fill="{t["fg2"]}">(source order doesn\'t matter)</text>')
    return _svg(950, 345, "\n".join(b), t)


def make_all(hero, refine):
    fig_banner(hero)
    for theme in ("light", "dark"):
        fig_problem_gif(hero, theme)
        fig_refinement(refine, theme)
        fig_score(theme)
        (ASSETS / f"pipeline-{theme}.svg").write_text(fig_pipeline(theme))
        (ASSETS / f"architecture-{theme}.svg").write_text(fig_architecture(theme))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    hero = cached("hero", hero_data, args.refresh)
    refine = cached("refinement", refinement_data, args.refresh)
    print("hero field:", hero["Us"].shape, "| refinement runs:", len(refine["runs"]),
          "| final losses:", np.round(refine["final_loss"], 4))
    make_all(hero, refine)
    print("Figures written to", ASSETS)
