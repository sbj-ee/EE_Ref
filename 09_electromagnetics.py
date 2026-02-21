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
        # Chapter 9: Electromagnetics — Example Visualizations

        Interactive graphs for selected example problems from Chapter 9,
        covering magnetic hysteresis, skin effect, transmission line transients,
        and electromagnetic shielding.
        """
    )
    return


# --- 9.2.5 B-H Hysteresis Curve ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 9.2.5 Magnetic Materials and Hysteresis — B-H Curve

        Ferromagnetic materials exhibit a nonlinear B-H relationship with hysteresis.
        The loop shows saturation (B_sat), remanence (B_r — flux remaining at H = 0),
        and coercivity (H_c — field needed to demagnetize). Soft magnetic materials
        (silicon steel) have narrow loops; hard materials (NdFeB) have wide loops.
        """
    )
    return


@app.cell
def _(np, plt):
    # Model a hysteresis loop using the arctangent approximation
    # B = B_sat * (2/pi) * arctan(k * (H ± H_c))
    B_sat = 1.8  # T (silicon steel saturation)
    H_c = 50     # A/m (coercivity for soft steel)
    B_r = 1.2    # T (remanence)

    # Derive k from remanence: B_r = B_sat * (2/pi) * arctan(k * H_c)
    k = np.tan(B_r / B_sat * np.pi / 2) / H_c

    H_max = 500  # A/m
    H_up = np.linspace(-H_max, H_max, 1000)
    H_down = np.linspace(H_max, -H_max, 1000)

    B_up = B_sat * (2 / np.pi) * np.arctan(k * (H_up + H_c))
    B_down = B_sat * (2 / np.pi) * np.arctan(k * (H_down - H_c))

    fig1, ax1 = plt.subplots(figsize=(9, 7))

    ax1.plot(H_up, B_up, "b-", linewidth=2, label="Magnetizing (H increasing)")
    ax1.plot(H_down, B_down, "r-", linewidth=2, label="Demagnetizing (H decreasing)")

    # Mark key points
    # Saturation
    ax1.plot(H_max, B_up[-1], "ko", markersize=8)
    ax1.annotate(f"B_sat ≈ {B_up[-1]:.2f} T", xy=(H_max, B_up[-1]),
                 xytext=(H_max - 150, B_up[-1] + 0.15), fontsize=10, color="black")

    # Remanence (H=0, upper branch)
    idx_rem = np.argmin(np.abs(H_down))
    ax1.plot(0, B_down[idx_rem], "go", markersize=10, zorder=5)
    ax1.annotate(f"B_r = {B_down[idx_rem]:.2f} T\n(Remanence)",
                 xy=(0, B_down[idx_rem]), xytext=(80, B_down[idx_rem] + 0.1),
                 fontsize=10, color="green", arrowprops=dict(arrowstyle="->", color="green"))

    # Coercivity (B=0, lower branch)
    idx_coer = np.argmin(np.abs(B_down))
    ax1.plot(H_down[idx_coer], 0, "mo", markersize=10, zorder=5)
    ax1.annotate(f"H_c = {abs(H_down[idx_coer]):.0f} A/m\n(Coercivity)",
                 xy=(H_down[idx_coer], 0), xytext=(H_down[idx_coer] - 180, -0.5),
                 fontsize=10, color="purple", arrowprops=dict(arrowstyle="->", color="purple"))

    ax1.axhline(y=0, color="gray", linewidth=0.5)
    ax1.axvline(x=0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Magnetic Field Intensity H (A/m)")
    ax1.set_ylabel("Flux Density B (T)")
    ax1.set_title("B-H Hysteresis Loop — Soft Magnetic Material (Silicon Steel)")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-H_max, H_max)
    ax1.set_ylim(-2.0, 2.0)

    fig1.tight_layout()
    fig1
    return


# --- 9.4.4 Skin Effect: Skin Depth vs Frequency ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 9.4.4 Skin Effect — Skin Depth vs Frequency

        The skin depth δ = 1/√(πfμ₀σ) decreases with frequency, confining AC current
        to a thin surface layer. At 60 Hz copper has δ ≈ 8.5 mm; at 1 MHz, δ ≈ 66 μm;
        at 10 GHz, δ ≈ 0.66 μm. The R_AC/R_DC ratio increases as the current is
        squeezed into a thinner annulus.
        """
    )
    return


@app.cell
def _(np, plt):
    mu0 = 4 * np.pi * 1e-7
    sigma_cu = 5.8e7  # S/m (copper)
    sigma_al = 3.5e7  # S/m (aluminum)
    sigma_steel = 1.0e7  # S/m (mild steel, μ_r ≈ 200)
    mu_r_steel = 200

    f_skin = np.logspace(0, 11, 1000)  # 1 Hz to 100 GHz

    delta_cu = 1 / np.sqrt(np.pi * f_skin * mu0 * sigma_cu)
    delta_al = 1 / np.sqrt(np.pi * f_skin * mu0 * sigma_al)
    delta_steel = 1 / np.sqrt(np.pi * f_skin * mu0 * mu_r_steel * sigma_steel)

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    ax2.loglog(f_skin, delta_cu * 1e3, "b-", linewidth=2, label="Copper (σ = 5.8×10⁷ S/m)")
    ax2.loglog(f_skin, delta_al * 1e3, "r-", linewidth=2, label="Aluminum (σ = 3.5×10⁷ S/m)")
    ax2.loglog(f_skin, delta_steel * 1e3, "g-", linewidth=2, label="Mild Steel (σ = 10⁷, μᵣ = 200)")

    # Mark key points from the example
    for f_pt, label in [(60, "60 Hz"), (1e6, "1 MHz"), (10e9, "10 GHz")]:
        d_pt = 1 / np.sqrt(np.pi * f_pt * mu0 * sigma_cu)
        ax2.plot(f_pt, d_pt * 1e3, "ko", markersize=7, zorder=5)
        if d_pt > 1e-3:
            txt = f"{d_pt*1e3:.1f} mm"
        else:
            txt = f"{d_pt*1e6:.1f} μm"
        ax2.annotate(f"{label}\nδ = {txt}", xy=(f_pt, d_pt * 1e3),
                     xytext=(f_pt * 3, d_pt * 1e3 * 2), fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="black"))

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Skin Depth (mm)")
    ax2.set_title("Skin Depth vs Frequency for Common Conductors")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(1, 1e11)
    ax2.set_ylim(1e-4, 100)

    fig2.tight_layout()
    fig2
    return


# --- 9.5.6 Transmission Line Transients and TDR ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 9.5.6 TDR — Transmission Line Lattice Diagram

        A 1 V step from a 50 Ω source drives a 50 Ω cable (3 m, v = 2×10⁸ m/s)
        terminated in 150 Ω. The source is matched (Γ_S = 0) so only one reflection
        occurs at the load (Γ_L = 0.5). The load voltage steps from 0 → 0.75 V at
        t = 15 ns and stays there.
        """
    )
    return


@app.cell
def _(np, plt):
    # Parameters from Example 9.5.6
    Vs = 1.0     # V source
    Z0 = 50      # Ω line impedance
    Zs = 50      # Ω source impedance
    ZL = 150     # Ω load impedance
    v_prop = 2e8  # m/s
    length = 3    # m
    td = length / v_prop  # 15 ns one-way delay

    V_inc = Vs * Z0 / (Zs + Z0)  # 0.5 V
    Gamma_L = (ZL - Z0) / (ZL + Z0)  # 0.5
    Gamma_S = (Zs - Z0) / (Zs + Z0)  # 0.0

    V_refl = Gamma_L * V_inc  # 0.25 V

    t_max = 80e-9
    t_arr = np.linspace(0, t_max, 1000)

    # Source-end voltage
    V_source = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        v = 0
        if t >= 0:
            v += V_inc                    # initial wave launched
        if t >= 2 * td:
            v += V_refl                   # reflected wave arrives back at source
        V_source[i] = v

    # Load-end voltage
    V_load = np.zeros_like(t_arr)
    for i, t in enumerate(t_arr):
        v = 0
        if t >= td:
            v += V_inc + V_refl           # incident + reflected at load
        V_load[i] = v

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 8))

    # Lattice diagram (top)
    ax3a.set_xlim(0, 3)
    ax3a.set_ylim(0, 70)
    ax3a.invert_yaxis()

    # Draw the source and load vertical lines
    ax3a.axvline(x=0, color="black", linewidth=2)
    ax3a.axvline(x=3, color="black", linewidth=2)
    ax3a.text(0, -3, "Source\n(50 Ω)", ha="center", fontsize=10, fontweight="bold")
    ax3a.text(3, -3, "Load\n(150 Ω)", ha="center", fontsize=10, fontweight="bold")

    # Incident wave: source → load, t=0 to t=15ns
    ax3a.annotate("", xy=(3, 15), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
    ax3a.text(1.5, 4, f"V⁺ = {V_inc:.2f} V", fontsize=11, color="blue",
              ha="center", rotation=-15, fontweight="bold")

    # Reflected wave: load → source, t=15ns to t=30ns
    ax3a.annotate("", xy=(0, 30), xytext=(3, 15),
                  arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    ax3a.text(1.5, 19, f"V⁻ = Γ_L × V⁺ = {V_refl:.2f} V", fontsize=11, color="red",
              ha="center", rotation=15, fontweight="bold")

    # Absorbed at source (Γ_S = 0)
    ax3a.text(0.1, 33, "Γ_S = 0 (absorbed)", fontsize=9, color="gray", style="italic")

    # Time labels
    for t_ns in [0, 15, 30]:
        ax3a.axhline(y=t_ns, color="gray", linestyle=":", alpha=0.3, xmin=0, xmax=1)
        ax3a.text(-0.2, t_ns, f"{t_ns} ns", fontsize=9, ha="right", va="center")

    ax3a.set_xlabel("Position along line (m)")
    ax3a.set_ylabel("Time (ns)")
    ax3a.set_title("Lattice (Bounce) Diagram: 50 Ω Source → 50 Ω Line → 150 Ω Load")
    ax3a.grid(False)

    # Voltage vs time at source and load (bottom)
    ax3b.step(t_arr * 1e9, V_source, "b-", linewidth=2, where="post", label="V at source end")
    ax3b.step(t_arr * 1e9, V_load, "r-", linewidth=2, where="post", label="V at load end")
    ax3b.axhline(y=0.75, color="green", linestyle="--", alpha=0.6,
                 label=f"Steady state = {Vs*ZL/(Zs+ZL):.2f} V")

    # Annotate key transitions
    ax3b.annotate(f"V⁺ = {V_inc:.2f} V", xy=(1, V_inc), xytext=(8, V_inc + 0.1),
                  fontsize=10, color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
    ax3b.annotate(f"V_L = {V_inc + V_refl:.2f} V", xy=(td * 1e9 + 1, V_inc + V_refl),
                  xytext=(td * 1e9 + 10, V_inc + V_refl + 0.1),
                  fontsize=10, color="red", arrowprops=dict(arrowstyle="->", color="red"))

    ax3b.set_xlabel("Time (ns)")
    ax3b.set_ylabel("Voltage (V)")
    ax3b.set_title("Voltage vs Time at Source and Load Ends")
    ax3b.set_xlim(0, t_max * 1e9)
    ax3b.set_ylim(-0.1, 1.0)
    ax3b.legend(fontsize=9)
    ax3b.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3
    return


# --- 9.7.1 Shielding Effectiveness ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 9.7.1 Shielding Effectiveness vs Frequency

        Shielding effectiveness of a conductive enclosure has two components:
        absorption loss (increases with frequency as skin depth shrinks) and
        reflection loss (decreases with frequency). For aluminum at 1.5 mm thickness,
        the total SE is dominated by absorption above ~1 MHz.
        """
    )
    return


@app.cell
def _(np, plt):
    sigma_al_se = 3.5e7   # S/m aluminum
    sigma_cu_ref = 5.8e7   # S/m copper reference
    sigma_r = sigma_al_se / sigma_cu_ref  # relative to copper
    mu_r_al = 1
    t_shield = 1.5e-3  # m (1.5 mm)
    mu0_se = 4 * np.pi * 1e-7

    f_se = np.logspace(3, 10, 1000)  # 1 kHz to 10 GHz

    # Skin depth for aluminum
    delta_al_se = 1 / np.sqrt(np.pi * f_se * mu0_se * sigma_al_se)

    # Absorption loss: A = 8.686 * (t / δ) dB
    A_loss = 8.686 * (t_shield / delta_al_se)

    # Reflection loss for plane wave: R ≈ 168 - 10*log10(f*μ_r/σ_r) dB
    R_loss = 168 - 10 * np.log10(f_se * mu_r_al / sigma_r)

    # Total SE (ignoring multi-reflection correction since A >> 15 dB above ~10 kHz)
    SE_total = A_loss + R_loss

    fig4, ax4 = plt.subplots(figsize=(10, 6))

    ax4.semilogx(f_se, A_loss, "b--", linewidth=1.5, label="Absorption loss A")
    ax4.semilogx(f_se, R_loss, "r--", linewidth=1.5, label="Reflection loss R")
    ax4.semilogx(f_se, SE_total, "k-", linewidth=2.5, label="Total SE = A + R")

    # Mark the 100 MHz point from the example
    f_mark = 1e8
    d_mark = 1 / np.sqrt(np.pi * f_mark * mu0_se * sigma_al_se)
    A_mark = 8.686 * (t_shield / d_mark)
    R_mark = 168 - 10 * np.log10(f_mark * mu_r_al / sigma_r)
    ax4.plot(f_mark, A_mark + R_mark, "go", markersize=10, zorder=5)
    ax4.annotate(f"100 MHz: SE = {A_mark+R_mark:.0f} dB\n(A = {A_mark:.0f}, R = {R_mark:.0f})",
                 xy=(f_mark, A_mark + R_mark),
                 xytext=(f_mark / 10, A_mark + R_mark + 100),
                 fontsize=10, color="green",
                 arrowprops=dict(arrowstyle="->", color="green"))

    # 60 dB target line
    ax4.axhline(y=60, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Target: 60 dB")

    ax4.set_xlabel("Frequency (Hz)")
    ax4.set_ylabel("Shielding Effectiveness (dB)")
    ax4.set_title("Shielding Effectiveness — 1.5 mm Aluminum Enclosure")
    ax4.legend(fontsize=9, loc="upper left")
    ax4.grid(True, alpha=0.3, which="both")
    ax4.set_xlim(1e3, 1e10)
    ax4.set_ylim(0, 2000)

    fig4.tight_layout()
    fig4
    return


# --- 9.5.5 Microstrip Characteristic Impedance vs W/h ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 9.5.5 Microstrip — Characteristic Impedance vs Trace Width

        Microstrip Z₀ depends on the ratio W/h (trace width / substrate height)
        and the substrate dielectric constant εᵣ.  For W/h < 1 the formula uses
        a narrow-strip approximation; for W/h ≥ 1 a wide-strip form applies.
        The effective dielectric constant εₑff < εᵣ because part of the field
        travels in the air above the trace.

        The example point marks the 50 Ω design on FR-4 (εᵣ = 4.3, h = 0.254 mm)
        from §9.5.5.
        """
    )
    return


@app.cell
def _(np, plt):
    def _ms_Z0(wh_ratio, er):
        """Hammerstad approximate microstrip Z0 (Ω) and effective εᵣ."""
        wh = wh_ratio
        eta0 = 377.0
        if wh < 1:
            _F = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / wh) ** 0.7528)
            _Z = (eta0 / (2 * np.pi)) * np.log(_F / wh + np.sqrt(1 + (2 / wh) ** 2))
            _e = (er + 1) / 2 + (er - 1) / 2 * (1 / np.sqrt(1 + 12 / wh) + 0.04 * (1 - wh) ** 2)
        else:
            _e = (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 / wh)
            _Z = (eta0 / (2 * np.pi * np.sqrt(_e))) / (wh + 1.393 + 0.667 * np.log(wh + 1.444))
        return _Z, _e

    _wh_arr = np.linspace(0.05, 6.0, 800)

    _substrates = [
        ("FR-4 (εᵣ = 4.3)",           4.3,  "blue"),
        ("Rogers RO4003 (εᵣ = 3.55)", 3.55, "red"),
        ("PTFE (εᵣ = 2.2)",           2.2,  "green"),
    ]

    fig5, (_ax5a, _ax5b) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    for _lbl5, _er5, _c5 in _substrates:
        _Z0v  = np.array([_ms_Z0(_w, _er5)[0] for _w in _wh_arr])
        _eefv = np.array([_ms_Z0(_w, _er5)[1] for _w in _wh_arr])
        _ax5a.plot(_wh_arr, _Z0v,  color=_c5, linewidth=2, label=_lbl5)
        _ax5b.plot(_wh_arr, _eefv, color=_c5, linewidth=2, label=_lbl5)

    _ax5a.axhline(y=50, color="gray", linestyle="--", alpha=0.7, linewidth=1.5,
                  label="Z₀ = 50 Ω target")
    _ax5a.axhline(y=75, color="gray", linestyle=":",  alpha=0.5, linewidth=1.5,
                  label="Z₀ = 75 Ω target")

    _wh_ex = 1.81
    _Z0_ex, _eeff_ex = _ms_Z0(_wh_ex, 4.3)
    _ax5a.plot(_wh_ex, _Z0_ex,   "ko", markersize=9, zorder=6)
    _ax5a.annotate(
        f"FR-4: W/h = {_wh_ex}, Z₀ = {_Z0_ex:.1f} Ω",
        xy=(_wh_ex, _Z0_ex), xytext=(_wh_ex + 0.8, _Z0_ex + 20),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
    )
    _ax5b.plot(_wh_ex, _eeff_ex, "ko", markersize=9, zorder=6)
    _ax5b.annotate(
        f"εeff = {_eeff_ex:.2f}",
        xy=(_wh_ex, _eeff_ex), xytext=(_wh_ex + 0.8, _eeff_ex - 0.25),
        fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
    )

    _ax5a.set_ylabel("Characteristic Impedance Z₀ (Ω)", fontsize=10)
    _ax5a.set_title("Microstrip Characteristic Impedance vs Trace Width Ratio (W/h)", fontsize=11)
    _ax5a.legend(fontsize=9)
    _ax5a.grid(True, alpha=0.3)
    _ax5a.set_ylim(0, 200)

    _ax5b.set_xlabel("W/h  (Trace Width / Substrate Height)", fontsize=10)
    _ax5b.set_ylabel("Effective Dielectric Constant εeff", fontsize=10)
    _ax5b.set_title("Effective Dielectric Constant vs W/h", fontsize=11)
    _ax5b.legend(fontsize=9)
    _ax5b.grid(True, alpha=0.3)
    _ax5b.set_xlim(0.05, 6)

    fig5.tight_layout()
    fig5
    return fig5


if __name__ == "__main__":
    app.run()
