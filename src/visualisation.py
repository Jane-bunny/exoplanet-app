from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ======================================================
#  Generic Visualisation Helpers (Layer 1)
# ======================================================

def plot_hist(
    df: pd.DataFrame,
    col: str,
    *,
    title: str | None = None,
    fname: str | None = None,
    bins: int = 50,
    logx: bool = False,
    color: str = "steelblue",
    alpha: float = 0.8,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a simple histogram for one numeric column.

    Parameters
    ----------
    df : DataFrame
    col : str
        Column to plot.
    logx : bool
        Apply log10(x) to the data before plotting (for wide distributions).
    save_dir : str | Path
        Optional directory to save figure (if fname is given).
    show : bool
        Display figure interactively (default True).

    Returns
    -------
    matplotlib.figure.Figure
    """
    s = df[col].dropna()
    if logx:
        s = np.log10(s[s > 0])

    fig, ax = plt.subplots()
    ax.hist(s, bins=bins, color=color, alpha=alpha, edgecolor="black", linewidth=0.3)
    ax.set_xlabel(("log10 " if logx else "") + col)
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if fname and save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / fname, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    c: str | None = None,
    title: str | None = None,
    fname: str | None = None,
    logx: bool = False,
    logy: bool = False,
    alpha: float = 0.6,
    s: float = 15,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Generic scatter plot with optional grouping by color column.
    """
    sub = df[[x, y] + ([c] if c else [])].dropna()
    fig, ax = plt.subplots(figsize=(6, 5))

    if c:
        for label, grp in sub.groupby(c):
            ax.scatter(grp[x], grp[y], s=s, alpha=alpha, label=str(label))
        ax.legend(title=c)
    else:
        ax.scatter(sub[x], sub[y], s=s, alpha=alpha, color="steelblue")

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    fig.tight_layout()

    if fname and save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / fname, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def plot_2d_density(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    bins: int | tuple[int, int] = 60,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    logx: bool = True,
    logy: bool = True,
    cmap: str = "viridis",
    title: str | None = None,
    fname: str | None = None,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if logx: sub = sub[sub[x] > 0]
    if logy: sub = sub[sub[y] > 0]

    # resolve bin counts
    if isinstance(bins, int):
        nxb, nyb = bins, bins
    else:
        nxb, nyb = bins

    # infer ranges if not provided
    xmin = xlim[0] if xlim else sub[x].min()
    xmax = xlim[1] if xlim else sub[x].max()
    ymin = ylim[0] if ylim else sub[y].min()
    ymax = ylim[1] if ylim else sub[y].max()

    # build edges consistent with axis scale
    if logx:
        xedges = np.logspace(np.log10(xmin), np.log10(xmax), nxb + 1)
    else:
        xedges = np.linspace(xmin, xmax, nxb + 1)
    if logy:
        yedges = np.logspace(np.log10(ymin), np.log10(ymax), nyb + 1)
    else:
        yedges = np.linspace(ymin, ymax, nyb + 1)

    # histogram in those edges
    H, _, _ = np.histogram2d(sub[x].to_numpy(), sub[y].to_numpy(),
                             bins=[xedges, yedges])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=1), cmap=cmap, shading="auto")

    if logx: ax.set_xscale("log")
    if logy: ax.set_yscale("log")
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])

    ax.set_xlabel(x); ax.set_ylabel(y)
    if title: ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Counts (log scale)")
    fig.tight_layout()

    if fname and save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / fname, dpi=150)

    if show: plt.show()
    else: plt.close(fig)
    return fig
# ======================================================
#  Domain-Specific Presets (Layer 2)
# ======================================================
def hist2d_radius_period(
    df: pd.DataFrame,
    *,
    xcol: str = "orbital_period",
    ycol: str = "planet_radius",
    xlim: tuple[float, float] = (0.2, 1000),
    ylim: tuple[float, float] = (0.5, 30),
    bins: int | tuple[int, int] = (60, 60),
    title: str | None = "Planet radius vs orbital period (density)",
    **kwargs,
):
    """Standard log–log 2D density plot of planet radius vs orbital period."""
    return plot_2d_density(
        df, xcol, ycol, bins=bins,
        xlim=xlim, ylim=ylim,
        logx=True, logy=True,
        title=title, **kwargs
    )
def hist2d_radius_flux(
    df: pd.DataFrame,
    *,
    xcol: str = "insolation_flux",
    ycol: str = "planet_radius",
    xlim: tuple[float, float] = (1e-3, 1e4),
    ylim: tuple[float, float] = (0.5, 30),
    bins: int | tuple[int, int] = (60, 60),
    title: str | None = "Planet radius vs stellar flux (density)",
    **kwargs,
):
    """Log–log 2D density plot of planet radius vs stellar insolation flux."""
    return plot_2d_density(
        df, xcol, ycol, bins=bins,
        xlim=xlim, ylim=ylim,
        logx=True, logy=True,
        title=title, **kwargs
    )
def facet_rp_hist2d(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    *,
    labels: tuple[str, str] = ("Kepler", "TESS"),
    xcol: str = "orbital_period",
    ycol: str = "planet_radius",
    xlim: tuple[float, float] = (0.2, 1000),
    ylim: tuple[float, float] = (0.5, 30),
    bins: int = 60,
    cmap: str = "viridis",
    title: str | None = "Radius–Period Distributions",
    savepath: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Side-by-side comparison of two surveys (shared scale, outer colorbar)."""
    xedges = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bins + 1)
    yedges = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), bins + 1)

    def hist2d(df):
        sub = df[[xcol, ycol]].dropna()
        return np.histogram2d(sub[xcol], sub[ycol], bins=[xedges, yedges])[0]

    H1, H2 = hist2d(dfA), hist2d(dfB)
    vmax = max(H1.max(), H2.max())

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, H, label in zip(axs, (H1, H2), labels):
        im = ax.pcolormesh(xedges, yedges, H.T, cmap=cmap, norm=LogNorm(vmin=1, vmax=vmax))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("Orbital period (days)")
        ax.set_ylabel("Planet radius (R⊕)")
        ax.set_title(label)

    # shared colorbar outside both
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label="Counts (log scale)")

    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

def fractional_difference_map_plot(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    *,
    xcol: str = "orbital_period",
    ycol: str = "planet_radius",
    xlim: tuple[float, float] = (0.2, 1000),
    ylim: tuple[float, float] = (0.5, 30),
    bins: int = 60,
    cmap: str = "coolwarm",
    label_A: str = "Kepler",
    label_B: str = "TESS",
    min_total: int = 3,          # mask very sparse bins
    title: str | None = None,
    savepath: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot fractional difference F = (A - B) / (A + B), with A=dfA, B=dfB.
    Colorbar and title use the provided survey labels.
    """
    xedges = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), bins + 1)
    yedges = np.logspace(np.log10(ylim[0]), np.log10(ylim[1]), bins + 1)

    def xy(df):
        sub = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
        sub = sub[(sub[xcol] > 0) & (sub[ycol] > 0)]
        return sub[xcol].to_numpy(), sub[ycol].to_numpy()

    xA, yA = xy(dfA); xB, yB = xy(dfB)
    H_A, _, _ = np.histogram2d(xA, yA, bins=[xedges, yedges])
    H_B, _, _ = np.histogram2d(xB, yB, bins=[xedges, yedges])
    T = H_A + H_B
    F = (H_A - H_B) / (T + 1e-9)

    # mask low-total bins so extreme colors don’t show on noise
    F = np.where(T >= min_total, F, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    # set NaNs to gray
    from matplotlib import cm
    cmap_obj = cm.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="#d9d9d9")
    im = ax.pcolormesh(xedges, yedges, F.T, cmap=cmap_obj, vmin=-1, vmax=1, shading="auto")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("Orbital period (days, log)")
    ax.set_ylabel("Planet radius (R⊕, log)")

    # clearer colorbar label with survey names
    cb_label = f"({label_A} − {label_B}) / ({label_A} + {label_B})"
    cbar = fig.colorbar(im, ax=ax, label=cb_label)
    cbar.ax.set_ylabel(cb_label)

    if title is None:
        title = f"Fractional difference: {cb_label}"
    ax.set_title(title)

    fig.tight_layout()
    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
def gmm_logP_logR_plot(
    df,
    n_components=3,
    *,
    xcol="orbital_period",
    ycol="planet_radius",
    title_prefix=None,       # e.g. "Kepler" or "TESS"
    title=None,
    cmap="tab10",
    savepath=None,
    show=True,
):
    """Fit GMM in log10(P)–log10(R) and plot points + 1σ ellipses."""
    from sklearn.mixture import GaussianMixture
    from matplotlib.patches import Ellipse

    sub = df[[xcol, ycol]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub[xcol] > 0) & (sub[ycol] > 0)]
    X = np.log10(sub[[xcol, ycol]].to_numpy())

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=0).fit(X)
    labels = gmm.predict(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(10**X[:, 0], 10**X[:, 1], s=6, alpha=0.3, c=labels, cmap=cmap)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Orbital period (days)")
    ax.set_ylabel("Planet radius (R⊕)")

    # 1σ ellipses
    for k in range(n_components):
        m = gmm.means_[k]
        C = gmm.covariances_[k]
        vals, vecs = np.linalg.eigh(C)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * np.sqrt(vals)
        e = Ellipse(
            xy=(10**m[0], 10**m[1]),
            width=10**(m[0] + width/2) / 10**(m[0] - width/2),
            height=10**(m[1] + height/2) / 10**(m[1] - height/2),
            angle=angle, fill=False, lw=2, ec="k"
        )
        ax.add_patch(e)

    # Title logic
    if title_prefix and not title:
        title = f"{title_prefix} — GMM clusters in logP–logR"
    elif not title:
        title = "GMM clusters in logP–logR"
    ax.set_title(title)

    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()
        return None
    else:
        plt.close(fig)
        return fig, gmm, labels