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
    ## 1.1 U.S. Electricity Generation by Source (2025)

    Horizontal bar chart of 2025 U.S. net electricity generation by source,
    from the EIA *Electric Power Annual 2025* (released October 21, 2026),
    Tables 3.1.A and 3.1.B, utility-scale facilities only. Solar is
    utility-scale PV plus solar thermal; the estimated 93,041 million kWh of
    small-scale (rooftop) PV is excluded, matching the Chapter 1 table.
    The EIA publishes these figures in million kWh; the chart plots them in
    terawatt-hours (1 TWh = 1,000 million kWh) so the axis, the bar lengths,
    and the printed values all carry the same unit. Sources are color-coded:
    fossil fuels in reds, nuclear in orange, wind and biomass in greens,
    hydro and pumped storage in blues, solar in gold, geothermal in teal.
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
        "Solar (utility-scale)",
        "Biomass — Wood",
        "Petroleum",
        "Biomass — Waste",
        "Geothermal",
        "Other Gases",
        "Pumped Storage",
    ]
    values = np.array([
        1_811_004,
        734_928,
        785_304,
        461_577,
        241_303,
        300_303,
        31_179,
        19_245,
        14_113,
        15_628,
        9_528,
        -5_362,
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
    ax1.set_xlabel("Net Generation (TWh)")
    ax1.set_title("2025 U.S. Electricity Net Generation by Source "
                  "(utility-scale; EIA Electric Power Annual)")
    ax1.grid(True, axis="x", alpha=0.3)

    # Annotate values on bars
    for i, (val, _bar) in enumerate(zip(sorted_values, _bars)):
        if val >= 0:
            ax1.text(val / 1000 + 15, i, f"{val/1000:,.0f}", va="center", fontsize=8)
        else:
            ax1.text(val / 1000 - 15, i, f"{val/1000:,.0f}", va="center", fontsize=8,
                     ha="right")

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#CC3333", label="Fossil fuels (reds)"),
        Patch(facecolor="#E68A00", label="Nuclear (orange)"),
        Patch(facecolor="#33AA33", label="Wind and biomass (greens)"),
        Patch(facecolor="#2288CC", label="Hydro and pumped storage (blues)"),
        Patch(facecolor="#FFB833", label="Solar (gold)"),
        Patch(facecolor="#44AA88", label="Geothermal (teal)"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax1.set_xlim(-100, 2000)
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1.3.6 Power Factor Correction

    Example 1.3.6: a 500 kW load is corrected from PF 0.72 to PF 0.95 lagging.
    Left subplot: the power triangles before and after correction, with
    lagging reactive power drawn upward (positive), as in the text:
    Q₁ = 481.9 kVAR, Q₂ = 164.3 kVAR, so the capacitor bank supplies
    Q<sub>C</sub> = 317.6 kVAR. Right subplot: capacitive kVAR required to
    correct the same 500 kW load from its original power factor to unity,
    for PFs from 0.70 to 1.0.
    """)
    return


@app.cell
def _(np, plt):
    P_load = 500  # kW
    pf1, pf2 = 0.72, 0.95
    th1, th2 = np.arccos(pf1), np.arccos(pf2)
    Q1 = P_load * np.tan(th1)   # 481.9 kVAR
    Q2 = P_load * np.tan(th2)   # 164.3 kVAR
    Qc = Q1 - Q2                # 317.6 kVAR
    S1, S2 = P_load / pf1, P_load / pf2

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(12, 5.4))

    # --- Subplot (a): power triangles, lagging Q drawn upward ---
    ax2a.annotate("", xy=(P_load, 0), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="blue", linewidth=2))
    ax2a.annotate("", xy=(P_load, Q1), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="green", linewidth=2))
    ax2a.annotate("", xy=(P_load, Q2), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="#2E8B57",
                                  linewidth=2, linestyle="--"))
    ax2a.annotate("", xy=(P_load, Q2), xytext=(P_load, Q1),
                  arrowprops=dict(arrowstyle="->", color="red", linewidth=2))
    ax2a.plot([P_load, P_load], [0, Q2], color="gray", linewidth=1.2)

    ax2a.text(P_load / 2, -28, f"P = {P_load} kW", ha="center", fontsize=11,
              color="blue", fontweight="bold")
    ax2a.text(P_load + 12, (Q1 + Q2) / 2, f"Q$_C$ = {Qc:.1f}\nkVAR\n(capacitor)",
              ha="left", va="center", fontsize=10, color="red",
              fontweight="bold")
    ax2a.text(P_load + 12, Q1, f"Q₁ = {Q1:.1f} kVAR", ha="left", va="center",
              fontsize=9, color="green")
    ax2a.text(P_load + 12, Q2 / 2, f"Q₂ = {Q2:.1f} kVAR", ha="left",
              va="center", fontsize=9, color="#2E8B57")
    ax2a.text(P_load * 0.42 - 40, Q1 * 0.42 + 30,
              f"S₁ = {S1:.1f} kVA\n(PF {pf1})", ha="center", fontsize=10,
              color="green", fontweight="bold")
    ax2a.text(330, 150, f"S₂ = {S2:.1f} kVA\n(PF {pf2})", ha="center",
              fontsize=10, color="#2E8B57", fontweight="bold")
    for _thx, _r, _xy, _lab in ((th1, 80, (62, 82), f"θ₁ = {np.degrees(th1):.2f}°"),
                                (th2, 160, (168, 18), f"θ₂ = {np.degrees(th2):.2f}°")):
        _arc = np.linspace(0, _thx, 30)
        ax2a.plot(_r * np.cos(_arc), _r * np.sin(_arc), "k-", linewidth=1)
        ax2a.text(_xy[0], _xy[1], _lab, fontsize=9)

    ax2a.set_xlim(-50, P_load + 210)
    ax2a.set_ylim(-60, Q1 + 60)
    ax2a.set_aspect("equal")
    ax2a.set_title(f"Power Triangle, PF {pf1} → {pf2} (Example 1.3.6)")
    ax2a.set_xlabel("Real Power (kW)")
    ax2a.set_ylabel("Reactive Power (kVAR, lagging positive)")
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

    The 6-pulse VFD current spectrum of Example 1.5.1: 5th 20%, 7th 14%,
    11th 9%, 13th 7.7%, 17th 5.9% and 19th 5.3% of the fundamental. Only the
    characteristic orders h = 6k ± 1 appear; there are no triplens. The first
    subplot overlays the resulting current on the 60 Hz fundamental, and the
    second shows the spectrum with THD = √(Σ I<sub>h</sub>²) = 28.3%.
    """)
    return


@app.cell
def _(np, plt):
    f0 = 60  # fundamental frequency (Hz)
    t = np.linspace(0, 2 / f0, 2000)  # two full cycles

    harmonics = [1, 5, 7, 11, 13, 17, 19]
    magnitudes = [100.0, 20.0, 14.0, 9.0, 7.7, 5.9, 5.3]  # % of fundamental
    # Typical 6-pulse sign pattern (-, -, +, +, -, -) for 5, 7, 11, 13, 17, 19
    signs = [1, -1, -1, 1, 1, -1, -1]
    distorted = sum(_sg * _m / 100 * np.sin(2 * np.pi * _h * f0 * t)
                    for _h, _m, _sg in zip(harmonics, magnitudes, signs))
    fundamental = np.sin(2 * np.pi * f0 * t)

    # THD calculation
    thd = np.sqrt(sum(_m**2 for _m in magnitudes[1:]))  # 28.3 percent

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot (a): Waveforms ---
    ax3a.plot(t * 1000, fundamental, "b-", linewidth=1.5, alpha=0.6,
              label="Fundamental (60 Hz)")
    ax3a.plot(t * 1000, distorted, "r-", linewidth=2,
              label="6-pulse VFD current")
    ax3a.set_xlabel("Time (ms)")
    ax3a.set_ylabel("Amplitude (per-unit)")
    ax3a.set_title("Fundamental vs Distorted Current (Example 1.5.1)")
    ax3a.legend(fontsize=9, loc="upper right")
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim(0, 2 / f0 * 1000)
    ax3a.set_ylim(-1.6, 1.8)

    # --- Subplot (b): Harmonic spectrum ---
    bar_colors = ["#2266CC"] + ["#CC4444"] * 6
    labels = [f"{_h}{'st' if _h == 1 else 'th'}\n({_h * f0} Hz)"
              for _h in harmonics]

    _bars = ax3b.bar(labels, magnitudes, color=bar_colors, edgecolor="white",
                    width=0.6)

    # Annotate bar values
    for _bar, mag in zip(_bars, magnitudes):
        ax3b.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 1.5,
                  f"{mag:g}%", ha="center", fontsize=10, fontweight="bold")

    # THD annotation
    ax3b.text(0.95, 0.92, f"THD = {thd:.1f}%\n(6k ± 1 only; no triplens)",
              transform=ax3b.transAxes, fontsize=12, fontweight="bold",
              ha="right", va="top",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        edgecolor="gray", alpha=0.9))

    ax3b.set_ylabel("Magnitude (% of Fundamental)")
    ax3b.set_title("Harmonic Spectrum: 6-Pulse VFD")
    ax3b.grid(True, axis="y", alpha=0.3)
    ax3b.tick_params(axis="x", labelsize=8)
    ax3b.set_ylim(0, 115)

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
