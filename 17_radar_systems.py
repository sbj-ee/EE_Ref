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
        # Chapter 17: Radar Systems — Example Visualizations

        Interactive graph for FMCW radar beat frequency analysis.
        """
    )
    return


# --- 17.2.2 FMCW Radar Beat Frequency ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 17.2.2 FMCW Radar — Beat Frequency vs Range

        An automotive FMCW radar at 77 GHz with 4 GHz bandwidth and 40 μs
        sweep produces a beat frequency proportional to target range.
        The range resolution is c/(2B) = 3.75 cm.
        """
    )
    return


@app.cell
def _(np, plt):
    c = 3e8
    B_fmcw = 4e9  # Hz bandwidth
    T_sweep = 40e-6  # sweep time (s)

    R_range = np.linspace(0, 100, 500)  # range in meters
    f_beat = 2 * R_range * B_fmcw / (c * T_sweep)

    # ADC Nyquist limit at 40 MHz sampling
    f_nyq = 20e6  # Hz
    R_max_40M = f_nyq * c * T_sweep / (2 * B_fmcw)

    # Mark example: 50 m target
    R_ex = 50
    fb_ex = 2 * R_ex * B_fmcw / (c * T_sweep)

    # Range resolution
    dR = c / (2 * B_fmcw)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(R_range, f_beat / 1e6, "b-", linewidth=2, label="Beat frequency")
    ax.axhline(y=f_nyq / 1e6, color="red", linestyle="--", alpha=0.6,
               label=f"ADC Nyquist limit (40 MHz sampling) = {f_nyq/1e6:.0f} MHz")
    ax.axvline(x=R_max_40M, color="red", linestyle=":", alpha=0.4)

    ax.plot(R_ex, fb_ex / 1e6, "go", markersize=10, zorder=5)
    ax.annotate(f"R = {R_ex} m\nf_b = {fb_ex/1e6:.1f} MHz\n(exceeds ADC limit!)",
                xy=(R_ex, fb_ex / 1e6), xytext=(R_ex + 8, fb_ex / 1e6 - 5),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")

    ax.annotate(f"R_max = {R_max_40M:.0f} m\n(at 40 MHz ADC)",
                xy=(R_max_40M, f_nyq / 1e6), xytext=(R_max_40M + 8, f_nyq / 1e6 + 5),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")

    ax.annotate(f"ΔR = {dR*100:.2f} cm",
                xy=(5, 2), fontsize=11, color="purple",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Beat Frequency (MHz)")
    ax.set_title("FMCW Radar: Beat Frequency vs Range (77 GHz, B = 4 GHz, T = 40 μs)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
