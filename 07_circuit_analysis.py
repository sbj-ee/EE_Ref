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
        # Chapter 7: Circuit Analysis — Example Visualizations

        Interactive graphs for selected example problems from Chapter 7,
        covering AC impedance, resonance, and transient analysis of RC, RL, and RLC circuits.
        """
    )
    return


# --- 7.5.1 Impedance: Series RL circuit ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 7.5.1 Impedance — Series RL Circuit

        A series circuit has R = 100 Ω and L = 50 mH driven by a 120 Vrms source.
        The impedance magnitude increases with frequency as the inductive reactance
        X_L = ωL grows, while the phase angle approaches 90° at high frequencies.
        """
    )
    return


@app.cell
def _(np, plt):
    R_rl = 100  # Ω
    L_rl = 50e-3  # H
    f_rl = np.linspace(1, 1000, 1000)
    omega_rl = 2 * np.pi * f_rl
    X_L = omega_rl * L_rl
    Z_mag_rl = np.sqrt(R_rl**2 + X_L**2)
    Z_phase_rl = np.degrees(np.arctan2(X_L, R_rl))

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1a.plot(f_rl, Z_mag_rl, "b-", linewidth=2)
    ax1a.axhline(y=R_rl, color="gray", linestyle="--", alpha=0.6, label="R = 100 Ω")
    # Mark the example point at 60 Hz
    f_ex = 60
    omega_ex = 2 * np.pi * f_ex
    Z_ex = np.sqrt(R_rl**2 + (omega_ex * L_rl) ** 2)
    ax1a.plot(f_ex, Z_ex, "ro", markersize=8, label=f"60 Hz: |Z| = {Z_ex:.1f} Ω")
    ax1a.set_ylabel("|Z| (Ω)")
    ax1a.set_title("Series RL Impedance vs Frequency (R = 100 Ω, L = 50 mH)")
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    ax1b.plot(f_rl, Z_phase_rl, "r-", linewidth=2)
    theta_ex = np.degrees(np.arctan2(omega_ex * L_rl, R_rl))
    ax1b.plot(f_ex, theta_ex, "ro", markersize=8, label=f"60 Hz: θ = {theta_ex:.1f}°")
    ax1b.set_xlabel("Frequency (Hz)")
    ax1b.set_ylabel("Phase Angle (°)")
    ax1b.set_ylim(0, 90)
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1
    return


# --- 7.5.2 Resonance: Series RLC circuit ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 7.5.2 Resonance — Series RLC Circuit

        A series RLC circuit (R = 10 Ω, L = 1 mH, C = 10 nF) resonates at
        f₀ ≈ 50.3 kHz where impedance is minimum (Z = R) and the phase crosses zero.
        The quality factor Q = 31.62 determines the sharpness of the resonance peak.
        """
    )
    return


@app.cell
def _(np, plt):
    R_rlc_res = 10  # Ω
    L_rlc_res = 1e-3  # H
    C_rlc_res = 10e-9  # F
    f0_res = 1 / (2 * np.pi * np.sqrt(L_rlc_res * C_rlc_res))
    Q_res = (1 / R_rlc_res) * np.sqrt(L_rlc_res / C_rlc_res)
    BW_res = f0_res / Q_res

    f_res = np.linspace(10e3, 100e3, 5000)
    omega_res = 2 * np.pi * f_res
    X_res = omega_res * L_rlc_res - 1 / (omega_res * C_rlc_res)
    Z_mag_res = np.sqrt(R_rlc_res**2 + X_res**2)
    Z_phase_res = np.degrees(np.arctan2(X_res, R_rlc_res))

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax2a.semilogy(f_res / 1e3, Z_mag_res, "b-", linewidth=2)
    ax2a.axvline(x=f0_res / 1e3, color="red", linestyle="--", alpha=0.6, label=f"f₀ = {f0_res/1e3:.1f} kHz")
    ax2a.axhline(y=R_rlc_res, color="gray", linestyle=":", alpha=0.6, label=f"R = {R_rlc_res} Ω (minimum)")
    # Mark bandwidth
    f_low = (f0_res - BW_res / 2) / 1e3
    f_high = (f0_res + BW_res / 2) / 1e3
    ax2a.axvspan(f_low, f_high, alpha=0.15, color="green", label=f"BW = {BW_res:.0f} Hz, Q = {Q_res:.1f}")
    ax2a.set_ylabel("|Z| (Ω)")
    ax2a.set_title(f"Series RLC Resonance (R={R_rlc_res} Ω, L={L_rlc_res*1e3} mH, C={C_rlc_res*1e9} nF)")
    ax2a.legend(fontsize=9)
    ax2a.grid(True, alpha=0.3, which="both")

    ax2b.plot(f_res / 1e3, Z_phase_res, "r-", linewidth=2)
    ax2b.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
    ax2b.axvline(x=f0_res / 1e3, color="red", linestyle="--", alpha=0.6)
    ax2b.set_xlabel("Frequency (kHz)")
    ax2b.set_ylabel("Phase Angle (°)")
    ax2b.set_ylim(-90, 90)
    ax2b.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2
    return


# --- 7.6.1 RC Circuits: Charging curve ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 7.6.1 RC Circuit — Capacitor Charging

        A 10 μF capacitor charges through a 47 kΩ resistor from a 9 V battery.
        The time constant τ = RC = 0.47 s. At t = 0.5 s the voltage reaches 5.89 V
        (about 65.5% of the source voltage, slightly past one time constant).
        """
    )
    return


@app.cell
def _(np, plt):
    R_rc = 47e3  # Ω
    C_rc = 10e-6  # F
    Vs_rc = 9  # V
    tau_rc = R_rc * C_rc  # 0.47 s

    t_rc = np.linspace(0, 5 * tau_rc, 500)
    Vc_rc = Vs_rc * (1 - np.exp(-t_rc / tau_rc))

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(t_rc, Vc_rc, "b-", linewidth=2, label="Vc(t) = 9(1 − e⁻ᵗ/⁰·⁴⁷)")
    ax3.axhline(y=Vs_rc, color="gray", linestyle="--", alpha=0.5, label="Vs = 9 V")

    # Mark τ
    Vc_at_tau = Vs_rc * (1 - np.exp(-1))
    ax3.plot(tau_rc, Vc_at_tau, "go", markersize=10, zorder=5)
    ax3.annotate(f"τ = {tau_rc} s\nVc = {Vc_at_tau:.2f} V (63.2%)",
                 xy=(tau_rc, Vc_at_tau), xytext=(tau_rc + 0.3, Vc_at_tau - 1.5),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="green"),
                 color="green")

    # Mark example point at t = 0.5 s
    Vc_05 = Vs_rc * (1 - np.exp(-0.5 / tau_rc))
    ax3.plot(0.5, Vc_05, "ro", markersize=10, zorder=5)
    ax3.annotate(f"t = 0.5 s\nVc = {Vc_05:.2f} V",
                 xy=(0.5, Vc_05), xytext=(0.8, Vc_05 - 2),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="red"),
                 color="red")

    # Mark 5τ
    ax3.axvline(x=5 * tau_rc, color="orange", linestyle=":", alpha=0.6, label=f"5τ = {5*tau_rc:.2f} s (≈99.3%)")

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Capacitor Voltage (V)")
    ax3.set_title("RC Charging Curve (R = 47 kΩ, C = 10 μF, Vs = 9 V)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 5 * tau_rc)
    ax3.set_ylim(0, 10)

    fig3.tight_layout()
    fig3
    return


# --- 7.6.2 RL Circuits: Current rise and voltage decay ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 7.6.2 RL Circuit — Current Rise and Inductor Voltage Decay

        A 200 mH inductor with 50 Ω series resistance is connected to 24 V DC.
        The time constant τ = L/R = 4 ms. The current rises exponentially while
        the inductor voltage decays, both governed by the same time constant.
        """
    )
    return


@app.cell
def _(np, plt):
    R_rl2 = 50  # Ω
    L_rl2 = 200e-3  # H
    Vs_rl2 = 24  # V
    tau_rl2 = L_rl2 / R_rl2  # 4 ms
    Iss_rl2 = Vs_rl2 / R_rl2  # 480 mA

    t_rl2 = np.linspace(0, 5 * tau_rl2, 500)
    I_rl2 = Iss_rl2 * (1 - np.exp(-t_rl2 / tau_rl2))
    VL_rl2 = Vs_rl2 * np.exp(-t_rl2 / tau_rl2)

    fig4, ax4a = plt.subplots(figsize=(10, 5))
    ax4b = ax4a.twinx()

    line1, = ax4a.plot(t_rl2 * 1e3, I_rl2 * 1e3, "b-", linewidth=2, label="I(t)")
    ax4a.set_xlabel("Time (ms)")
    ax4a.set_ylabel("Current (mA)", color="blue")
    ax4a.tick_params(axis="y", labelcolor="blue")

    line2, = ax4b.plot(t_rl2 * 1e3, VL_rl2, "r-", linewidth=2, label="V_L(t)")
    ax4b.set_ylabel("Inductor Voltage (V)", color="red")
    ax4b.tick_params(axis="y", labelcolor="red")

    # Mark example point at t = 2 ms
    t_ex2 = 2e-3
    I_ex2 = Iss_rl2 * (1 - np.exp(-t_ex2 / tau_rl2))
    VL_ex2 = Vs_rl2 * np.exp(-t_ex2 / tau_rl2)
    ax4a.plot(2, I_ex2 * 1e3, "bo", markersize=8, zorder=5)
    ax4b.plot(2, VL_ex2, "ro", markersize=8, zorder=5)
    ax4a.annotate(f"t=2 ms: I={I_ex2*1e3:.1f} mA",
                  xy=(2, I_ex2 * 1e3), xytext=(6, I_ex2 * 1e3),
                  fontsize=9, arrowprops=dict(arrowstyle="->", color="blue"), color="blue")
    ax4b.annotate(f"t=2 ms: V_L={VL_ex2:.2f} V",
                  xy=(2, VL_ex2), xytext=(6, VL_ex2 + 2),
                  fontsize=9, arrowprops=dict(arrowstyle="->", color="red"), color="red")

    ax4a.set_title("RL Circuit Transient (R = 50 Ω, L = 200 mH, Vs = 24 V, τ = 4 ms)")
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax4a.legend(lines, labels, loc="center right")
    ax4a.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4
    return


# --- 7.6.3 RLC Circuits: Underdamped response ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 7.6.3 RLC Circuit — Underdamped Transient Response

        A series RLC circuit (R = 100 Ω, L = 10 mH, C = 1 μF) has ζ = 0.5 (underdamped).
        The natural frequency is ω₀ = 10,000 rad/s and the damped frequency is
        ω_d = 8,660 rad/s (≈1,378 Hz). The response oscillates with an exponentially
        decaying envelope.
        """
    )
    return


@app.cell
def _(np, plt):
    R_rlc = 100  # Ω
    L_rlc = 10e-3  # H
    C_rlc = 1e-6  # F
    omega0_rlc = 1 / np.sqrt(L_rlc * C_rlc)  # 10,000 rad/s
    zeta_rlc = R_rlc / (2 * np.sqrt(L_rlc / C_rlc))  # 0.5
    omega_d_rlc = omega0_rlc * np.sqrt(1 - zeta_rlc**2)  # 8,660 rad/s
    sigma_rlc = zeta_rlc * omega0_rlc  # 5,000 (decay rate)

    # Step response of underdamped series RLC
    t_rlc = np.linspace(0, 2e-3, 1000)
    # Capacitor voltage step response for series RLC with step input V
    V_step = 1.0  # normalized
    env_rlc = np.exp(-sigma_rlc * t_rlc)
    phi = np.arccos(zeta_rlc)
    Vc_rlc = V_step * (1 - (1 / np.sqrt(1 - zeta_rlc**2)) * env_rlc * np.sin(omega_d_rlc * t_rlc + phi))

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(t_rlc * 1e3, Vc_rlc, "b-", linewidth=2, label="Vc(t) — underdamped (ζ = 0.5)")
    # Envelope
    env_upper = V_step * (1 + (1 / np.sqrt(1 - zeta_rlc**2)) * env_rlc)
    env_lower = V_step * (1 - (1 / np.sqrt(1 - zeta_rlc**2)) * env_rlc)
    ax5.plot(t_rlc * 1e3, env_upper, "r--", alpha=0.5, linewidth=1, label="Envelope")
    ax5.plot(t_rlc * 1e3, env_lower, "r--", alpha=0.5, linewidth=1)
    ax5.axhline(y=V_step, color="gray", linestyle=":", alpha=0.6, label="Steady state")

    # Mark damped period
    T_d = 2 * np.pi / omega_d_rlc
    ax5.annotate(f"T_d = {T_d*1e3:.3f} ms\n(f_d = {omega_d_rlc/(2*np.pi):.0f} Hz)",
                 xy=(T_d * 1e3, Vc_rlc[int(T_d / t_rlc[-1] * len(t_rlc))]),
                 xytext=(1.2, 1.35),
                 fontsize=9, arrowprops=dict(arrowstyle="->", color="green"), color="green")

    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Normalized Capacitor Voltage")
    ax5.set_title(f"Underdamped RLC Step Response (ζ = {zeta_rlc}, ω₀ = {omega0_rlc:.0f} rad/s, ω_d = {omega_d_rlc:.0f} rad/s)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5
    return


if __name__ == "__main__":
    app.run()
