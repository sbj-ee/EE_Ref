import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Chapter 1: Power Engineering — Example Visualizations

    Interactive graphs for selected topics from Chapter 1,
    covering U.S. electricity generation mix, power factor correction,
    and harmonic distortion analysis.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1.1 U.S. Electricity Generation by Source (2022)

    Horizontal bar chart of Y2022 U.S. net electricity generation
    by source in million kWh. Sources are color-coded by category:
    fossil fuels (red shades), nuclear (orange), and renewables
    (green shades).
    """)
    return


@app.cell
def _(np, plt):
    sources = [
        "Natural Gas",
        "Coal",
        "Nuclear",
        "Wind",
        "Conv. Hydro",
        "Solar",
        "Biomass — Wood",
        "Petroleum",
        "Biomass — Waste",
        "Geothermal",
        "Other Gases",
        "Pumped Storage",
    ]
    values = np.array([
        1_579_190,
        897_999,
        779_645,
        380_300,
        251_585,
        163_550,
        36_463,
        19_173,
        17_790,
        15_975,
        11_397,
        -5_112,
    ])

    # Color coding: fossil=red shades, nuclear=orange, renewable=green shades
    colors = [
        "#CC3333",   # Natural Gas — fossil
        "#992222",   # Coal — fossil
        "#E68A00",   # Nuclear — orange
        "#33AA33",   # Wind — renewable
        "#2288CC",   # Conv. Hydro — renewable (blue-green)
        "#FFB833",   # Solar — renewable (golden)
        "#558833",   # Biomass Wood — renewable
        "#DD5555",   # Petroleum — fossil
        "#668844",   # Biomass Waste — renewable
        "#44AA88",   # Geothermal — renewable
        "#EE7777",   # Other Gases — fossil
        "#5599BB",   # Pumped Storage — hydro
    ]

    # Sort by value for the bar chart (ascending so largest is at top)
    order = np.argsort(values)
    sorted_sources = [sources[i] for i in order]
    sorted_values = values[order]
    sorted_colors = [colors[i] for i in order]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    _bars = ax1.barh(range(len(sorted_sources)), sorted_values / 1000,
                    color=sorted_colors, edgecolor="white", linewidth=0.5)

    ax1.set_yticks(range(len(sorted_sources)))
    ax1.set_yticklabels(sorted_sources, fontsize=9)
    ax1.set_xlabel("Net Generation (Thousand GWh)")
    ax1.set_title("Y2022 U.S. Electricity Net Generation by Source")
    ax1.grid(True, axis="x", alpha=0.3)

    # Annotate values on bars
    for i, (val, _bar) in enumerate(zip(sorted_values, _bars)):
        if val >= 0:
            ax1.text(val / 1000 + 15, i, f"{val:,.0f}", va="center", fontsize=8)
        else:
            ax1.text(val / 1000 - 15, i, f"{val:,.0f}", va="center", fontsize=8,
                     ha="right")

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#CC3333", label="Fossil Fuels"),
        Patch(facecolor="#E68A00", label="Nuclear"),
        Patch(facecolor="#33AA33", label="Renewable"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax1.set_xlim(-100, 1750)
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1.3.6 Power Factor Correction

    A 500 kW load at various power factors requires different amounts of
    reactive power. Left subplot: the power triangle at PF = 0.85, showing
    real power (P), reactive power (Q), and apparent power (S). Right subplot:
    capacitive kVAR required to correct from the original power factor to
    unity, plotted for PFs from 0.70 to 1.0.
    """)
    return


@app.cell
def _(np, plt):
    P_load = 500  # kW

    # --- Subplot (a): Power triangle at PF = 0.85 ---
    pf_example = 0.85
    theta_ex = np.arccos(pf_example)
    S_ex = P_load / pf_example  # kVA
    Q_ex = P_load * np.tan(theta_ex)  # kVAR

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5))

    # Draw power triangle
    # P along x-axis, Q along negative y-axis (lagging), S as hypotenuse
    ax2a.annotate("", xy=(P_load, 0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="blue", linewidth=2))
    ax2a.annotate("", xy=(P_load, -Q_ex), xytext=(P_load, 0),
                  arrowprops=dict(arrowstyle="->", color="red", linewidth=2))
    ax2a.annotate("", xy=(P_load, -Q_ex), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="green", linewidth=2))

    # Labels
    ax2a.text(P_load / 2, 15, f"P = {P_load} kW", ha="center", fontsize=11,
              color="blue", fontweight="bold")
    ax2a.text(P_load + 15, -Q_ex / 2, f"Q = {Q_ex:.0f}\nkVAR", ha="left",
              fontsize=10, color="red", fontweight="bold")
    ax2a.text(P_load / 2 - 40, -Q_ex / 2 - 20,
              f"S = {S_ex:.0f} kVA", ha="center", fontsize=10,
              color="green", fontweight="bold")

    # Angle arc
    arc_angles = np.linspace(0, -theta_ex, 30)
    arc_r = 80
    ax2a.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), "k-",
              linewidth=1)
    ax2a.text(90, -25, f"θ = {np.degrees(theta_ex):.1f}°", fontsize=9)

    ax2a.set_xlim(-50, P_load + 100)
    ax2a.set_ylim(-Q_ex - 50, 60)
    ax2a.set_aspect("equal")
    ax2a.set_title(f"Power Triangle at PF = {pf_example}")
    ax2a.set_xlabel("Real Power (kW)")
    ax2a.set_ylabel("Reactive Power (kVAR)")
    ax2a.grid(True, alpha=0.3)
    ax2a.axhline(y=0, color="black", linewidth=0.5)
    ax2a.axvline(x=0, color="black", linewidth=0.5)

    # --- Subplot (b): Required correction kVAR vs original PF ---
    pf_range = np.linspace(0.70, 0.999, 200)
    theta_range = np.arccos(pf_range)
    # Q at each PF for the 500 kW load
    Q_original = P_load * np.tan(theta_range)
    # To correct to unity PF (Q=0), capacitor must supply all Q
    Q_correction = Q_original  # kVAR of capacitor needed

    ax2b.plot(pf_range, Q_correction, "b-", linewidth=2)
    ax2b.fill_between(pf_range, Q_correction, alpha=0.1, color="blue")

    # Mark specific points
    for pf_mark in [0.70, 0.80, 0.85, 0.90, 0.95]:
        q_mark = P_load * np.tan(np.arccos(pf_mark))
        ax2b.plot(pf_mark, q_mark, "ro", markersize=7, zorder=5)
        ax2b.annotate(f"{q_mark:.0f} kVAR",
                      xy=(pf_mark, q_mark),
                      xytext=(pf_mark - 0.03, q_mark + 20),
                      fontsize=8, ha="center")

    ax2b.set_xlabel("Original Power Factor")
    ax2b.set_ylabel("Capacitive kVAR for Unity PF Correction")
    ax2b.set_title(f"Correction kVAR vs Power Factor ({P_load} kW Load)")
    ax2b.grid(True, alpha=0.3)
    ax2b.set_xlim(0.68, 1.02)

    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1.5.1 Harmonics and Total Harmonic Distortion (THD)

    A 60 Hz fundamental waveform is overlaid with a distorted waveform
    containing 3rd harmonic (20%), 5th harmonic (15%), and 7th harmonic (10%).
    The second subplot shows the harmonic spectrum as a bar chart with the
    magnitude of each component and the calculated THD value.
    """)
    return


@app.cell
def _(np, plt):
    f0 = 60  # fundamental frequency (Hz)
    t = np.linspace(0, 2 / f0, 1000)  # two full cycles

    # Fundamental and harmonic components (amplitudes as fraction of fundamental)
    fundamental = np.sin(2 * np.pi * f0 * t)
    h3 = 0.20 * np.sin(2 * np.pi * 3 * f0 * t)
    h5 = 0.15 * np.sin(2 * np.pi * 5 * f0 * t)
    h7 = 0.10 * np.sin(2 * np.pi * 7 * f0 * t)

    distorted = fundamental + h3 + h5 + h7

    # THD calculation
    thd = np.sqrt(0.20**2 + 0.15**2 + 0.10**2) * 100  # percent

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot (a): Waveforms ---
    ax3a.plot(t * 1000, fundamental, "b-", linewidth=1.5, alpha=0.6,
              label="Fundamental (60 Hz)")
    ax3a.plot(t * 1000, distorted, "r-", linewidth=2,
              label="Distorted Waveform")
    ax3a.set_xlabel("Time (ms)")
    ax3a.set_ylabel("Amplitude (per-unit)")
    ax3a.set_title("Fundamental vs Distorted Waveform")
    ax3a.legend(fontsize=9, loc="upper right")
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim(0, 2 / f0 * 1000)

    # --- Subplot (b): Harmonic spectrum ---
    harmonics = [1, 3, 5, 7]
    magnitudes = [100.0, 20.0, 15.0, 10.0]  # percent of fundamental
    bar_colors = ["#2266CC", "#CC4444", "#CC4444", "#CC4444"]
    labels = ["1st\n(60 Hz)", "3rd\n(180 Hz)", "5th\n(300 Hz)", "7th\n(420 Hz)"]

    _bars = ax3b.bar(labels, magnitudes, color=bar_colors, edgecolor="white",
                    width=0.6)

    # Annotate bar values
    for _bar, mag in zip(_bars, magnitudes):
        ax3b.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 1.5,
                  f"{mag:.0f}%", ha="center", fontsize=10, fontweight="bold")

    # THD annotation
    ax3b.text(0.95, 0.92, f"THD = {thd:.1f}%",
              transform=ax3b.transAxes, fontsize=12, fontweight="bold",
              ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        edgecolor="gray", alpha=0.9))

    ax3b.set_ylabel("Magnitude (% of Fundamental)")
    ax3b.set_title("Harmonic Spectrum")
    ax3b.grid(True, axis="y", alpha=0.3)
    ax3b.set_ylim(0, 115)

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
