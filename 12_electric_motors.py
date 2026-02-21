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
        # Chapter 12: Electric Motors — Example Visualizations

        Interactive graphs for selected example problems from Chapter 12,
        covering stepper motor torque curves, FOC vector diagrams,
        and motor thermal modeling.
        """
    )
    return


# --- 12.3.4 Stepper Motor Torque Curves ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 12.3.4 Stepper Motor — Pull-In and Pull-Out Torque Curves

        A NEMA 23 stepper motor with 1.9 N·m holding torque exhibits
        speed-dependent torque rolloff. The pull-out torque is always higher
        than pull-in torque. Mid-band resonance near 175 steps/s causes a
        torque dip that should be avoided or ramped through quickly.
        """
    )
    return


@app.cell
def _(np, plt):
    # Stepper motor torque vs speed curves (typical NEMA 23)
    T_hold = 1.9  # N·m holding torque
    f_res = 175  # resonant step rate (Hz)

    steps_s = np.linspace(0, 2000, 1000)

    # Pull-out torque: starts at holding torque, decays with speed
    T_pullout = T_hold * np.exp(-steps_s / 800)
    # Add resonance dip near 175 steps/s
    resonance_dip = 0.3 * np.exp(-((steps_s - f_res) / 30)**2)
    T_pullout = T_pullout - resonance_dip
    T_pullout = np.maximum(T_pullout, 0)

    # Pull-in torque: lower, decays faster
    T_pullin = 0.7 * T_hold * np.exp(-steps_s / 500)
    T_pullin = T_pullin - resonance_dip * 0.8
    T_pullin = np.maximum(T_pullin, 0)

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.plot(steps_s, T_pullout, "b-", linewidth=2, label="Pull-out torque")
    _ax.plot(steps_s, T_pullin, "r--", linewidth=2, label="Pull-in torque")
    _ax.fill_between(steps_s, T_pullin, T_pullout, alpha=0.1, color="blue",
                     label="Slew range (run only)")
    _ax.fill_between(steps_s, 0, T_pullin, alpha=0.1, color="green",
                     label="Start/stop range")

    # Mark resonance
    _ax.axvline(x=f_res, color="orange", linestyle=":", alpha=0.7)
    _ax.annotate(f"Mid-band\nresonance\n({f_res} steps/s)",
                xy=(f_res, T_pullout[np.argmin(np.abs(steps_s - f_res))]),
                xytext=(f_res + 150, 1.4), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="orange"), color="orange")

    # Mark the example point at 1000 steps/s
    idx_1000 = np.argmin(np.abs(steps_s - 1000))
    _ax.plot(1000, T_pullout[idx_1000], "go", markersize=8, zorder=5)
    _ax.annotate(f"1000 steps/s\nT_pullout ≈ {T_pullout[idx_1000]:.2f} N·m",
                xy=(1000, T_pullout[idx_1000]),
                xytext=(1100, T_pullout[idx_1000] + 0.3), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="green"), color="green")

    _ax.set_xlabel("Step Rate (steps/s)")
    _ax.set_ylabel("Torque (N·m)")
    _ax.set_title("Stepper Motor Torque vs Speed (NEMA 23, 1.9 N·m holding)")
    _ax.legend(fontsize=9)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 2000)
    _ax.set_ylim(0, 2.2)
    _fig.tight_layout()
    _fig
    return


# --- 12.4.5 FOC Vector Diagram ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 12.4.5 Field-Oriented Control — dq-Frame Vector Diagram

        A PMSM under FOC with I_d = 0 control has the stator current
        entirely on the q-axis (torque-producing). The stator voltage vector
        includes the back-EMF (ω_e × λ_m on q-axis) and the inductive
        voltage drops, resulting in a voltage vector that leads the current.
        """
    )
    return


@app.cell
def _(np, plt):
    # FOC parameters from Example 12.4.5
    Id = 0  # A (surface-mount PM, Id=0 control)
    Iq = 8.0  # A
    Rs = 0.5  # Ω
    Ld = 8e-3  # H
    Lq = 12e-3  # H
    lam_m = 0.25  # Wb
    omega_e = 314.2  # rad/s (electrical)

    # Voltage components
    Vd = Rs * Id - omega_e * Lq * Iq   # -30.2 V
    Vq = Rs * Iq + omega_e * Ld * Id + omega_e * lam_m  # 82.6 V
    Vs = np.sqrt(Vd**2 + Vq**2)
    Is = np.sqrt(Id**2 + Iq**2)

    _fig, _ax = plt.subplots(figsize=(8, 8))

    # Draw axes
    _ax.axhline(y=0, color="gray", linewidth=0.5)
    _ax.axvline(x=0, color="gray", linewidth=0.5)
    _ax.set_xlabel("d-axis")
    _ax.set_ylabel("q-axis")

    # Scale for visibility
    scale_I = 5  # scale current vectors for visibility
    scale_V = 0.5  # scale voltage vectors

    # Current vector (all on q-axis)
    _ax.annotate("", xy=(Id * scale_I, Iq * scale_I), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
    _ax.text(Id * scale_I + 1, Iq * scale_I, f"$I_s$ = {Is:.1f} A\n($I_q$ = {Iq:.1f} A)",
            fontsize=11, color="blue")

    # Voltage vector
    _ax.annotate("", xy=(Vd * scale_V, Vq * scale_V), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    _ax.text(Vd * scale_V - 8, Vq * scale_V + 1,
            f"$V_s$ = {Vs:.1f} V\n($V_d$ = {Vd:.1f}, $V_q$ = {Vq:.1f})",
            fontsize=11, color="red")

    # Back-EMF vector (on q-axis only)
    emf_q = omega_e * lam_m
    _ax.annotate("", xy=(0, emf_q * scale_V), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="green", lw=2, linestyle="--"))
    _ax.text(1, emf_q * scale_V - 2, f"Back-EMF\nω_e λ_m = {emf_q:.1f} V",
            fontsize=10, color="green")

    # Flux linkage (on d-axis)
    _ax.annotate("", xy=(lam_m * 150, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="purple", lw=2, linestyle="--"))
    _ax.text(lam_m * 150 + 1, -3, f"λ_m = {lam_m} Wb", fontsize=10, color="purple")

    _ax.set_xlim(-25, 50)
    _ax.set_ylim(-10, 55)
    _ax.set_aspect("equal")
    _ax.set_title("FOC Vector Diagram: PMSM with $I_d$ = 0 Control at 1500 RPM")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return


# --- 12.5.6 Motor Thermal / Insulation Life ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 12.5.6 Motor Insulation Life vs Temperature

        The Arrhenius/10-degree rule: insulation life halves for every 10°C
        above the rated temperature. A Class F motor rated for 20,000 hours
        at 155°C sees dramatic life reduction with even modest overtemperature.
        """
    )
    return


@app.cell
def _(np, plt):
    T_rated = 155  # °C (Class F)
    L_rated = 20000  # hours

    T_range = np.linspace(130, 200, 200)
    L_life = L_rated * 2**((T_rated - T_range) / 10)

    # Example point: 172.1°C → 6120 hours
    T_ex = 172.1
    L_ex = L_rated * 2**((T_rated - T_ex) / 10)

    _fig, _ax = plt.subplots(figsize=(10, 5))
    _ax.semilogy(T_range, L_life, "b-", linewidth=2, label="Insulation life (Arrhenius)")

    # Mark rated point
    _ax.plot(T_rated, L_rated, "go", markersize=10, zorder=5)
    _ax.annotate(f"Rated: {T_rated}°C, {L_rated:,} hrs",
                xy=(T_rated, L_rated), xytext=(T_rated - 15, L_rated * 2),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")

    # Mark example point (overloaded motor)
    _ax.plot(T_ex, L_ex, "ro", markersize=10, zorder=5)
    _ax.annotate(f"Example: {T_ex}°C\n{L_ex:,.0f} hrs (69% reduction)",
                xy=(T_ex, L_ex), xytext=(T_ex + 3, L_ex * 3),
                fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")

    # Mark insulation classes
    for temp, cls in [(130, "Class B"), (155, "Class F"), (180, "Class H")]:
        _ax.axvline(x=temp, color="gray", linestyle=":", alpha=0.5)
        _ax.text(temp + 0.5, _ax.get_ylim()[0] * 2, cls, fontsize=8, color="gray", rotation=90)

    _ax.set_xlabel("Hot-Spot Temperature (°C)")
    _ax.set_ylabel("Insulation Life (hours)")
    _ax.set_title("Motor Insulation Life vs Temperature (Arrhenius / 10°C Rule)")
    _ax.legend()
    _ax.grid(True, alpha=0.3, which="both")
    _ax.set_xlim(130, 200)
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
