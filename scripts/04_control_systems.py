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
        # Chapter 4: Control Systems — Example Visualizations

        Interactive graphs for selected example problems from Chapter 4,
        covering Bode plots, step response, and root locus analysis.
        """
    )
    return


# --- 4.5 Bode Plot: Second-Order System ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 4.5 Bode Plot — Second-Order System Frequency Response

        Bode magnitude and phase plots for H(s) = ωn² / (s² + 2ζωns + ωn²)
        with ωn = 10 rad/s. Underdamped systems (ζ < 0.707) exhibit a resonance
        peak near ωn, while heavily damped systems roll off smoothly. The phase
        transitions from 0° to −180° through the natural frequency, with sharper
        transitions at lower damping ratios.
        """
    )
    return


@app.cell
def _(np, plt):
    wn = 10.0  # natural frequency (rad/s)
    zeta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    w = np.logspace(-1, 2.5, 1000)  # frequency range: 0.1 to ~316 rad/s

    fig_bode, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for _zeta, _color in zip(zeta_values, colors):
        # H(jw) = wn^2 / ((jw)^2 + 2*zeta*wn*(jw) + wn^2)
        # H(jw) = wn^2 / (wn^2 - w^2 + j*2*zeta*wn*w)
        H = wn**2 / (wn**2 - w**2 + 1j * 2 * _zeta * wn * w)

        mag_db = 20 * np.log10(np.abs(H))
        phase_deg = np.degrees(np.angle(H))

        ax_mag.plot(w, mag_db, color=_color, linewidth=2, label=f"ζ = {_zeta}")
        ax_phase.plot(w, phase_deg, color=_color, linewidth=2, label=f"ζ = {_zeta}")

    # Mark resonance peaks for underdamped cases
    for _zeta, _color in zip([0.1, 0.3, 0.5], colors[:3]):
        if _zeta < 1 / np.sqrt(2):
            w_peak = wn * np.sqrt(1 - 2 * _zeta**2)
            H_peak = wn**2 / (wn**2 - w_peak**2 + 1j * 2 * _zeta * wn * w_peak)
            peak_db = 20 * np.log10(np.abs(H_peak))
            ax_mag.plot(w_peak, peak_db, "o", color=_color, markersize=6, zorder=5)

    # Magnitude plot formatting
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title(f"Bode Plot: Second-Order System (ωn = {wn:.0f} rad/s)")
    ax_mag.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax_mag.axvline(x=wn, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax_mag.text(wn * 1.05, -35, f"ωn = {wn:.0f}", fontsize=9, color="gray", va="top")
    ax_mag.legend(fontsize=9, loc="lower left")
    ax_mag.grid(True, alpha=0.3, which="both")
    ax_mag.set_ylim(-40, 25)

    # Phase plot formatting
    ax_phase.set_xlabel("Frequency (rad/s)")
    ax_phase.set_ylabel("Phase (degrees)")
    ax_phase.axhline(y=-90, color="gray", linewidth=0.5, linestyle="--")
    ax_phase.axhline(y=-180, color="gray", linewidth=0.5, linestyle="--")
    ax_phase.axvline(x=wn, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax_phase.set_yticks([0, -45, -90, -135, -180])
    ax_phase.legend(fontsize=9, loc="lower left")
    ax_phase.grid(True, alpha=0.3, which="both")

    ax_mag.set_xscale("log")
    ax_phase.set_xscale("log")

    fig_bode.tight_layout()
    fig_bode
    return


# --- 4.6 Step Response: Second-Order System ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 4.6 Step Response — Second-Order System

        Unit step response of H(s) = ωn² / (s² + 2ζωns + ωn²) with ωn = 10 rad/s
        for damping ratios from underdamped (ζ = 0.1) to overdamped (ζ = 2.0).
        Lower damping produces faster rise time but higher overshoot and longer
        settling time. The ±2% settling band and overshoot annotation are shown
        for the ζ = 0.3 case.
        """
    )
    return


@app.cell
def _(np, plt):
    wn_step = 10.0
    zeta_step = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    colors_step = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]

    t = np.linspace(0, 3, 2000)

    fig_step, ax_step = plt.subplots(figsize=(10, 6))

    for _zeta, _color in zip(zeta_step, colors_step):
        if _zeta < 1.0:
            # Underdamped: c(t) = 1 - (e^(-zeta*wn*t) / sqrt(1-zeta^2)) * sin(wd*t + phi)
            wd = wn_step * np.sqrt(1 - _zeta**2)
            phi = np.arctan2(np.sqrt(1 - _zeta**2), _zeta)
            y = 1 - (np.exp(-_zeta * wn_step * t) / np.sqrt(1 - _zeta**2)) * np.sin(wd * t + phi)
        elif _zeta == 1.0:
            # Critically damped: c(t) = 1 - (1 + wn*t) * e^(-wn*t)
            y = 1 - (1 + wn_step * t) * np.exp(-wn_step * t)
        else:
            # Overdamped: two real poles
            s1 = -_zeta * wn_step + wn_step * np.sqrt(_zeta**2 - 1)
            s2 = -_zeta * wn_step - wn_step * np.sqrt(_zeta**2 - 1)
            y = 1 + (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s2 - s1)

        ax_step.plot(t, y, color=_color, linewidth=2, label=f"ζ = {_zeta}")

    # Settling band (±2%)
    ax_step.axhline(y=1.02, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax_step.axhline(y=0.98, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax_step.fill_between(t, 0.98, 1.02, color="gray", alpha=0.08)
    ax_step.text(2.85, 1.035, "±2% band", fontsize=9, color="gray", ha="right")

    # Steady-state reference
    ax_step.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":")

    # Annotate overshoot for zeta = 0.3
    zeta_ann = 0.3
    wd_ann = wn_step * np.sqrt(1 - zeta_ann**2)
    t_peak = np.pi / wd_ann
    Mp = np.exp(-zeta_ann * np.pi / np.sqrt(1 - zeta_ann**2))
    y_peak = 1 + Mp

    ax_step.plot(t_peak, y_peak, "o", color="#ff7f0e", markersize=8, zorder=5)
    ax_step.annotate(f"Overshoot = {Mp*100:.1f}%\nt_p = {t_peak:.3f} s",
                     xy=(t_peak, y_peak), xytext=(t_peak + 0.25, y_peak + 0.05),
                     fontsize=9, color="#ff7f0e",
                     arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Annotate settling time for zeta = 0.3
    t_settle = 4 / (zeta_ann * wn_step)
    ax_step.axvline(x=t_settle, color="#ff7f0e", linewidth=1, linestyle=":", alpha=0.6)
    ax_step.annotate(f"t_s = {t_settle:.2f} s (ζ=0.3)",
                     xy=(t_settle, 0.5), xytext=(t_settle + 0.15, 0.45),
                     fontsize=9, color="#ff7f0e",
                     arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax_step.set_xlabel("Time (s)")
    ax_step.set_ylabel("Response c(t)")
    ax_step.set_title(f"Unit Step Response: Second-Order System (ωn = {wn_step:.0f} rad/s)")
    ax_step.legend(fontsize=9, loc="upper right")
    ax_step.grid(True, alpha=0.3)
    ax_step.set_xlim(0, 3)
    ax_step.set_ylim(0, 1.85)
    fig_step.tight_layout()
    fig_step
    return


# --- 4.7 Root Locus: G(s) = K / [s(s+2)(s+5)] ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 4.7 Root Locus — G(s) = K / [s(s+2)(s+5)]

        The root locus traces the closed-loop poles as gain K varies from 0 to 200.
        The characteristic equation is s³ + 7s² + 10s + K = 0. Three branches
        originate at the open-loop poles (0, −2, −5) and move toward infinity along
        asymptotes. The locus crosses the imaginary axis at K = 70 (the stability
        limit from Routh-Hurwitz), where the closed-loop poles are at ±j√10. The
        left-half plane is shaded to indicate the stable region.
        """
    )
    return


@app.cell
def _(np, plt):
    # Characteristic equation: s^3 + 7s^2 + 10s + K = 0
    K_values = np.linspace(0, 200, 5000)

    # Compute roots for each K
    roots_all = np.array([np.roots([1, 7, 10, K]) for K in K_values])

    fig_rl, ax_rl = plt.subplots(figsize=(10, 8))

    # Shade left-half plane as stable
    ax_rl.axvspan(-12, 0, color="green", alpha=0.04)
    ax_rl.text(-5.5, 7.5, "Stable Region\n(LHP)", fontsize=11, color="green",
               ha="center", alpha=0.6, fontstyle="italic")

    # Shade right-half plane as unstable
    ax_rl.text(2.5, 7.5, "Unstable\n(RHP)", fontsize=11, color="red",
               ha="center", alpha=0.5, fontstyle="italic")

    # Plot imaginary axis
    ax_rl.axvline(x=0, color="black", linewidth=1.0)
    ax_rl.axhline(y=0, color="black", linewidth=0.5)

    # Plot root locus branches
    for i in range(3):
        real_parts = roots_all[:, i].real
        imag_parts = roots_all[:, i].imag
        ax_rl.plot(real_parts, imag_parts, "b.", markersize=0.5)

    # Mark open-loop poles (K=0): s = 0, -2, -5
    poles = [0, -2, -5]
    for p in poles:
        ax_rl.plot(p, 0, "kx", markersize=12, markeredgewidth=2.5, zorder=10)
    ax_rl.text(0.3, -0.5, "0", fontsize=10, color="black")
    ax_rl.text(-1.7, -0.7, "−2", fontsize=10, color="black")
    ax_rl.text(-4.7, -0.7, "−5", fontsize=10, color="black")

    # Imaginary axis crossing: Routh-Hurwitz gives K_crit = 70
    # At K=70: s^3 + 7s^2 + 10s + 70 = 0
    # Roots: s = -7, s = ±j*sqrt(10)
    K_crit = 70
    roots_crit = np.roots([1, 7, 10, K_crit])
    # Find the purely imaginary roots
    for r in roots_crit:
        if abs(r.real) < 0.1:  # approximately on jw axis
            ax_rl.plot(r.real, r.imag, "r*", markersize=15, zorder=10)

    w_cross = np.sqrt(10)
    ax_rl.annotate(f"jω crossing at K = {K_crit}\nω = ±√10 ≈ ±{w_cross:.2f}",
                   xy=(0, w_cross), xytext=(2, w_cross + 1.5),
                   fontsize=9, color="red",
                   arrowprops=dict(arrowstyle="->", color="red"),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_rl.annotate("",
                   xy=(0, -w_cross), xytext=(2, -w_cross - 1.5),
                   arrowprops=dict(arrowstyle="->", color="red"))

    # Mark breakaway point (on real axis between 0 and -2)
    # dK/ds = 0 where K = -(s^3 + 7s^2 + 10s)
    # dK/ds = -(3s^2 + 14s + 10) = 0 => s = (-14 ± sqrt(196-120))/6
    s_break = (-14 + np.sqrt(196 - 120)) / 6  # = -0.874
    K_break = -(s_break**3 + 7 * s_break**2 + 10 * s_break)
    ax_rl.plot(s_break, 0, "gD", markersize=8, zorder=10)
    ax_rl.annotate(f"Breakaway\nσ = {s_break:.2f}, K = {K_break:.1f}",
                   xy=(s_break, 0), xytext=(s_break - 1.5, 2.5),
                   fontsize=9, color="green",
                   arrowprops=dict(arrowstyle="->", color="green"),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Legend entries
    ax_rl.plot([], [], "kx", markersize=10, markeredgewidth=2.5, label="Open-loop poles")
    ax_rl.plot([], [], "r*", markersize=12, label=f"jω axis crossing (K = {K_crit})")
    ax_rl.plot([], [], "gD", markersize=8, label="Breakaway point")
    ax_rl.plot([], [], "b-", linewidth=2, label="Root locus")

    ax_rl.set_xlabel("Real Axis (σ)")
    ax_rl.set_ylabel("Imaginary Axis (jω)")
    ax_rl.set_title("Root Locus: G(s) = K / [s(s+2)(s+5)]")
    ax_rl.legend(fontsize=9, loc="upper left")
    ax_rl.grid(True, alpha=0.3)
    ax_rl.set_xlim(-12, 6)
    ax_rl.set_ylim(-9, 9)
    ax_rl.set_aspect("equal")
    fig_rl.tight_layout()
    fig_rl
    return


if __name__ == "__main__":
    app.run()
