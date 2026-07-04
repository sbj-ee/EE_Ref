import marimo

__generated_with = "0.19.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(
        """
        # Chapter 18: Optics — Example Visualizations

        Interactive graph for dielectric mirror reflectivity analysis.
        """
    )
    return


# --- 18.2.4 Dielectric Mirror Reflectivity ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 18.2.4 Dielectric HR Mirror — Reflectivity vs Number of Layer Pairs

        A quarter-wave stack of TiO₂/SiO₂ layers achieves extremely high
        reflectivity. With n_H/n_L = 2.30/1.46 = 1.575, only 8 pairs
        give 99.72% and 15 pairs give 99.9996%.
        """
    )
    return


@app.cell
def _(np, plt):
    n_H = 2.30  # TiO2
    n_L = 1.46  # SiO2
    ratio = n_H / n_L  # 1.575

    N_pairs = np.arange(1, 21)
    R = ((ratio**(2 * N_pairs) - 1) / (ratio**(2 * N_pairs) + 1))**2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(N_pairs, R * 100, "b-o", linewidth=2, markersize=6)

    # Mark example points
    R8 = ((ratio**16 - 1) / (ratio**16 + 1))**2
    R15 = ((ratio**30 - 1) / (ratio**30 + 1))**2
    ax.plot(8, R8 * 100, "ro", markersize=12, zorder=5)
    ax.annotate(f"N = 8: R = {R8*100:.2f}%",
                xy=(8, R8 * 100), xytext=(10, R8 * 100 - 3),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")
    ax.plot(15, R15 * 100, "go", markersize=12, zorder=5)
    ax.annotate(f"N = 15: R = {R15*100:.4f}%",
                xy=(15, R15 * 100), xytext=(16, 97),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")

    ax.axhline(y=99, color="gray", linestyle=":", alpha=0.5, label="99%")
    ax.axhline(y=99.9, color="gray", linestyle="--", alpha=0.5, label="99.9%")

    ax.set_xlabel("Number of Layer Pairs (N)")
    ax.set_ylabel("Reflectivity (%)")
    ax.set_title(f"Dielectric HR Mirror Reflectivity (TiO₂/SiO₂, n_H/n_L = {ratio:.3f})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 101)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
