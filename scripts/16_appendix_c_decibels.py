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
        # Appendix C: Decibels — Visualizations

        Interactive graphs showing the logarithmic dB scale, a fiber optic
        link budget waterfall, and an op-amp Bode plot with gain-bandwidth product.
        """
    )
    return


# --- C.1.3 Common dB Values: Scale visualization ---

@app.cell
def _(mo):
    mo.md(
        """
        ## C.1.3 Common Decibel Values — dB Scale

        The decibel compresses enormous linear ratios into a compact logarithmic
        scale. This plot shows both the power ratio and voltage ratio as a function
        of dB, illustrating key reference points: 3 dB (2× power), 6 dB (2× voltage),
        10 dB (10× power), and 20 dB (10× voltage).
        """
    )
    return


@app.cell
def _(np, plt):
    dB_range = np.linspace(-20, 40, 500)
    power_ratio = 10 ** (dB_range / 10)
    voltage_ratio = 10 ** (dB_range / 20)

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

    # Power ratio
    ax1a.semilogy(dB_range, power_ratio, "b-", linewidth=2)
    key_dB_p = [(-20, 0.01), (-10, 0.1), (-3, 0.5), (0, 1), (3, 2), (10, 10), (20, 100), (30, 1000)]
    for db, ratio in key_dB_p:
        ax1a.plot(db, ratio, "ro", markersize=6, zorder=5)
        ax1a.annotate(f"{db} dB → {ratio}×", xy=(db, ratio),
                      xytext=(db + 1.5, ratio * 1.5), fontsize=8, color="red",
                      arrowprops=dict(arrowstyle="->", color="red", lw=0.5))
    ax1a.set_xlabel("Decibels (dB)")
    ax1a.set_ylabel("Power Ratio (linear)")
    ax1a.set_title("Power Ratio: 10^(dB/10)")
    ax1a.grid(True, alpha=0.3, which="both")
    ax1a.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax1a.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    # Voltage ratio
    ax1b.semilogy(dB_range, voltage_ratio, "r-", linewidth=2)
    key_dB_v = [(-20, 0.1), (-6, 0.5), (-3, 0.707), (0, 1), (3, 1.414), (6, 2), (20, 10), (40, 100)]
    for db, ratio in key_dB_v:
        ax1b.plot(db, ratio, "bo", markersize=6, zorder=5)
        ax1b.annotate(f"{db} dB → {ratio}×", xy=(db, ratio),
                      xytext=(db + 1.5, ratio * 1.3), fontsize=8, color="blue",
                      arrowprops=dict(arrowstyle="->", color="blue", lw=0.5))
    ax1b.set_xlabel("Decibels (dB)")
    ax1b.set_ylabel("Voltage Ratio (linear)")
    ax1b.set_title("Voltage Ratio: 10^(dB/20)")
    ax1b.grid(True, alpha=0.3, which="both")
    ax1b.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
    ax1b.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    fig1.tight_layout()
    fig1
    return


# --- C.3.1 Link Budget: Fiber optic waterfall ---

@app.cell
def _(mo):
    mo.md(
        """
        ## C.3.1 Cascaded Gains and Losses — Fiber Optic Link Budget

        A fiber optic link starts at +3 dBm and passes through connectors (−0.5 dB each)
        and 20 km of fiber (−6 dB). The received power of −4 dBm has a 24 dB margin
        above the receiver sensitivity of −28 dBm.
        """
    )
    return


@app.cell
def _(np, plt):
    stages = ["Laser\nSource", "Connector\n1", "Fiber\n(20 km)", "Connector\n2", "Received\nPower"]
    gains = [3, -0.5, -6, -0.5, 0]  # last is a placeholder
    cumulative = [3]
    for g in gains[1:-1]:
        cumulative.append(cumulative[-1] + g)
    cumulative.append(cumulative[-1])  # received power stays same

    sensitivity = -28  # dBm

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Bar chart showing cumulative power at each stage
    colors_lb = ["green", "orange", "red", "orange", "blue"]
    bars = ax2.bar(range(len(stages)), cumulative, color=colors_lb, alpha=0.7, edgecolor="black", linewidth=1.2)

    # Annotate values
    for i, (stage, val) in enumerate(zip(stages, cumulative)):
        offset = 1 if val >= 0 else -1.5
        ax2.text(i, val + offset, f"{val:.1f} dBm", ha="center", fontsize=11, fontweight="bold")

    # Annotate individual gains/losses
    for i in range(1, len(cumulative)):
        g = gains[i] if i < len(gains) else 0
        if g != 0:
            mid_y = (cumulative[i - 1] + cumulative[i]) / 2
            ax2.annotate(f"{g:+.1f} dB", xy=(i - 0.5, mid_y), fontsize=9, color="gray", ha="center")

    # Receiver sensitivity line
    ax2.axhline(y=sensitivity, color="red", linestyle="--", linewidth=2,
                label=f"Receiver sensitivity: {sensitivity} dBm")

    # Margin annotation
    ax2.annotate("", xy=(4.4, cumulative[-1]), xytext=(4.4, sensitivity),
                 arrowprops=dict(arrowstyle="<->", color="green", lw=2))
    margin = cumulative[-1] - sensitivity
    ax2.text(4.55, (cumulative[-1] + sensitivity) / 2, f"Margin\n{margin:.0f} dB",
             fontsize=11, color="green", fontweight="bold")

    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels(stages)
    ax2.set_ylabel("Power Level (dBm)")
    ax2.set_title("Fiber Optic Link Budget")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(-35, 10)

    fig2.tight_layout()
    fig2
    return


# --- C.4.1 Amplifier Gain: Op-amp Bode plot ---

@app.cell
def _(mo):
    mo.md(
        """
        ## C.4.1 Amplifier Gain — Op-Amp Open-Loop Bode Plot

        An op-amp with 100 dB open-loop gain and GBW = 10 MHz rolls off at
        −20 dB/decade after a dominant pole. When configured for a closed-loop
        gain of 40 dB (A_CL = 100), the bandwidth is limited to
        BW = GBW / A_CL = 100 kHz.
        """
    )
    return


@app.cell
def _(np, plt):
    A_OL_dB = 100  # dB
    GBW = 10e6  # Hz
    A_OL = 10 ** (A_OL_dB / 20)  # 100,000
    f_pole = GBW / A_OL  # 100 Hz (dominant pole)

    A_CL_dB = 40  # dB
    A_CL = 10 ** (A_CL_dB / 20)  # 100
    BW_CL = GBW / A_CL  # 100 kHz

    f_bode = np.logspace(0, 8, 1000)  # 1 Hz to 100 MHz

    # Open-loop gain (single-pole model)
    H_OL = A_OL / (1 + 1j * f_bode / f_pole)
    gain_OL_dB = 20 * np.log10(np.abs(H_OL))

    # Closed-loop gain (flat up to BW, then roll off with open-loop)
    gain_CL_dB = np.minimum(A_CL_dB, gain_OL_dB)

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    ax3.semilogx(f_bode, gain_OL_dB, "b-", linewidth=2, label="Open-loop gain (A_OL)")
    ax3.semilogx(f_bode, gain_CL_dB, "r-", linewidth=2, label=f"Closed-loop gain (A_CL = {A_CL_dB} dB)")
    ax3.semilogx(f_bode, np.zeros_like(f_bode), "k:", linewidth=0.5, alpha=0.3)

    # Mark key points
    ax3.plot(f_pole, A_OL_dB - 3, "go", markersize=8, label=f"Dominant pole: {f_pole:.0f} Hz")
    ax3.plot(GBW, 0, "mo", markersize=8, label=f"Unity gain: {GBW/1e6:.0f} MHz")
    ax3.plot(BW_CL, A_CL_dB - 3, "rs", markersize=8, label=f"CL bandwidth: {BW_CL/1e3:.0f} kHz")

    # GBW annotation
    ax3.annotate("GBW = 10 MHz\n(constant)", xy=(3e5, 50), fontsize=10, color="gray",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # -20 dB/decade slope annotation
    ax3.annotate("−20 dB/decade", xy=(1e4, 60), fontsize=10, color="blue", rotation=-20)

    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Gain (dB)")
    ax3.set_title("Op-Amp Bode Plot: Open-Loop and Closed-Loop Gain")
    ax3.set_xlim(1, 1e8)
    ax3.set_ylim(-10, 110)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3, which="both")

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
