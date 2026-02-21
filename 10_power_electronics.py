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
        # Chapter 10: Power Electronics — Example Visualizations

        Interactive graphs for selected example problems from Chapter 10,
        covering rectifiers, DC-DC converters, inverters, and power losses.
        """
    )
    return


# --- 10.2.1 Single-Phase Rectifiers ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.2.1 Single-Phase Rectifier — Full-Wave with Capacitor Filter

        A full-wave bridge rectifier converts a 120 Vrms, 60 Hz AC supply to DC.
        The peak rectified voltage is 120√2 − 2 × 0.7 = 168.3 V (accounting for
        two diode drops). A capacitor filter reduces the ripple to maintain a
        relatively smooth DC output.
        """
    )
    return


@app.cell
def _(np, plt):
    f_ac = 60  # Hz
    Vrms = 120
    Vpeak = Vrms * np.sqrt(2)
    Vd = 0.7  # diode drop
    Vpeak_rect = Vpeak - 2 * Vd  # two diodes in bridge

    t_rect = np.linspace(0, 3 / f_ac, 2000)  # 3 cycles
    _v_ac = Vpeak * np.sin(2 * np.pi * f_ac * t_rect)
    v_rectified = np.abs(Vpeak * np.sin(2 * np.pi * f_ac * t_rect)) - 2 * Vd
    v_rectified = np.maximum(v_rectified, 0)

    # Simulate capacitor filter (approximate exponential discharge)
    C_filter = 1000e-6  # 1000 μF
    R_load = 100  # Ω
    tau_filter = R_load * C_filter
    v_filtered = np.zeros_like(t_rect)
    v_filtered[0] = Vpeak_rect
    _dt = t_rect[1] - t_rect[0]
    for _i in range(1, len(t_rect)):
        if v_rectified[_i] > v_filtered[_i - 1]:
            v_filtered[_i] = v_rectified[_i]
        else:
            v_filtered[_i] = v_filtered[_i - 1] * np.exp(-_dt / tau_filter)

    ripple = np.max(v_filtered[len(t_rect) // 3:]) - np.min(v_filtered[len(t_rect) // 3:])

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1a.plot(t_rect * 1e3, _v_ac, "b-", linewidth=1, alpha=0.5, label="AC Input")
    ax1a.plot(t_rect * 1e3, v_rectified, "r-", linewidth=1.5, label="Full-Wave Rectified")
    ax1a.set_ylabel("Voltage (V)")
    ax1a.set_title("Full-Wave Bridge Rectifier")
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    ax1b.plot(t_rect * 1e3, v_rectified, "r-", linewidth=1, alpha=0.3, label="Rectified (no filter)")
    ax1b.plot(t_rect * 1e3, v_filtered, "b-", linewidth=2, label=f"Filtered (C = 1000 μF)")
    ax1b.axhline(y=np.mean(v_filtered[len(t_rect) // 3:]), color="green", linestyle="--", alpha=0.6,
                 label=f"Avg DC ≈ {np.mean(v_filtered[len(t_rect)//3:]):.1f} V")
    ax1b.annotate(f"Ripple ≈ {ripple:.1f} Vpp", xy=(30, np.min(v_filtered[len(t_rect) // 3:]) + 1),
                  fontsize=10, color="red")
    ax1b.set_xlabel("Time (ms)")
    ax1b.set_ylabel("Voltage (V)")
    ax1b.set_title("Capacitor-Filtered Output (R_load = 100 Ω)")
    ax1b.legend(fontsize=9)
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1
    return


# --- 10.3.1 Buck Converter ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.3.1 Buck Converter — Inductor Current Waveform

        A buck converter steps 48 V down to 12 V at 2.5 A with L = 100 μH
        at 100 kHz. The duty cycle D = 12/48 = 0.25. The inductor current has a
        triangular ripple of ΔI = 0.9 A (30% ripple) in CCM.
        """
    )
    return


@app.cell
def _(np, plt):
    Vin_buck = 48  # V
    Vout_buck = 12  # V
    Iout_buck = 2.5  # A
    L_buck = 100e-6  # H
    fsw_buck = 100e3  # Hz
    D_buck = Vout_buck / Vin_buck  # 0.25
    T_buck = 1 / fsw_buck  # 10 μs
    dI_buck = (Vin_buck - Vout_buck) * D_buck * T_buck / L_buck  # ripple

    # Generate 3 switching cycles
    n_cycles = 3
    t_buck = np.linspace(0, n_cycles * T_buck, 3000)
    i_buck = np.zeros_like(t_buck)
    pwm_buck = np.zeros_like(t_buck)

    for _i, _t_val in enumerate(t_buck):
        _t_in_cycle = _t_val % T_buck
        if _t_in_cycle < D_buck * T_buck:
            # Switch ON: current ramps up
            i_buck[_i] = Iout_buck - dI_buck / 2 + (Vin_buck - Vout_buck) / L_buck * _t_in_cycle
            pwm_buck[_i] = 1
        else:
            # Switch OFF: current ramps down
            _t_off = _t_in_cycle - D_buck * T_buck
            i_buck[_i] = Iout_buck + dI_buck / 2 - Vout_buck / L_buck * _t_off
            pwm_buck[_i] = 0

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 3]})

    ax2a.fill_between(t_buck * 1e6, pwm_buck * Vin_buck, step="post", alpha=0.3, color="orange")
    ax2a.step(t_buck * 1e6, pwm_buck * Vin_buck, "orange", linewidth=1.5, where="post", label="Switch Node")
    ax2a.set_ylabel("V_sw (V)")
    ax2a.set_title(f"Buck Converter (Vin={Vin_buck}V → Vout={Vout_buck}V, D={D_buck:.2f}, f_sw={fsw_buck/1e3:.0f} kHz)")
    ax2a.legend()
    ax2a.grid(True, alpha=0.3)

    ax2b.plot(t_buck * 1e6, i_buck, "b-", linewidth=2, label=f"I_L(t), ΔI = {dI_buck:.2f} A ({dI_buck/Iout_buck*100:.0f}% ripple)")
    ax2b.axhline(y=Iout_buck, color="red", linestyle="--", alpha=0.6, label=f"I_avg = {Iout_buck} A")
    ax2b.axhline(y=Iout_buck + dI_buck / 2, color="gray", linestyle=":", alpha=0.5, label=f"I_peak = {Iout_buck + dI_buck/2:.2f} A")
    ax2b.axhline(y=Iout_buck - dI_buck / 2, color="gray", linestyle=":", alpha=0.5, label=f"I_valley = {Iout_buck - dI_buck/2:.2f} A")
    ax2b.set_xlabel("Time (μs)")
    ax2b.set_ylabel("Inductor Current (A)")
    ax2b.legend(fontsize=9)
    ax2b.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2
    return


# --- 10.3.2 Boost Converter ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.3.2 Boost Converter — Inductor Current Waveform

        A boost converter steps 12 V up to 48 V at 0.5 A output with L = 220 μH
        at 150 kHz. The duty cycle D = 1 − 12/48 = 0.75. The inductor current
        ripple and average input current are visualized.
        """
    )
    return


@app.cell
def _(np, plt):
    Vin_boost = 12  # V
    Vout_boost = 48  # V
    Iout_boost = 0.5  # A
    L_boost = 220e-6  # H
    fsw_boost = 150e3  # Hz
    D_boost = 1 - Vin_boost / Vout_boost  # 0.75
    T_boost = 1 / fsw_boost
    Iin_boost = Iout_boost / (1 - D_boost)  # 2 A average inductor current
    dI_boost = Vin_boost * D_boost * T_boost / L_boost

    n_cycles_b = 3
    t_boost = np.linspace(0, n_cycles_b * T_boost, 3000)
    i_boost = np.zeros_like(t_boost)

    for _i, _t_val in enumerate(t_boost):
        _t_in_cycle = _t_val % T_boost
        if _t_in_cycle < D_boost * T_boost:
            # Switch ON: inductor charges
            i_boost[_i] = Iin_boost - dI_boost / 2 + Vin_boost / L_boost * _t_in_cycle
        else:
            # Switch OFF: inductor delivers to output
            _t_off = _t_in_cycle - D_boost * T_boost
            i_boost[_i] = Iin_boost + dI_boost / 2 - (Vout_boost - Vin_boost) / L_boost * _t_off

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(t_boost * 1e6, i_boost, "b-", linewidth=2,
             label=f"I_L(t), ΔI = {dI_boost:.2f} A")
    ax3.axhline(y=Iin_boost, color="red", linestyle="--", alpha=0.6,
                label=f"I_avg = {Iin_boost:.1f} A")

    # Shade on/off regions
    for c in range(n_cycles_b):
        t_start = c * T_boost * 1e6
        t_on_end = t_start + D_boost * T_boost * 1e6
        t_cycle_end = t_start + T_boost * 1e6
        ax3.axvspan(t_start, t_on_end, alpha=0.05, color="orange")
        ax3.axvspan(t_on_end, t_cycle_end, alpha=0.05, color="green")

    ax3.set_xlabel("Time (μs)")
    ax3.set_ylabel("Inductor Current (A)")
    ax3.set_title(f"Boost Converter (Vin={Vin_boost}V → Vout={Vout_boost}V, D={D_boost:.2f}, f_sw={fsw_boost/1e3:.0f} kHz)")
    ax3.text(1, 2.4, "ON (charging)", fontsize=9, color="orange", alpha=0.7)
    ax3.text(D_boost * T_boost * 1e6 + 0.2, 2.4, "OFF (delivering)", fontsize=9, color="green", alpha=0.7)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3
    return


# --- 10.4.1 SPWM Inverter ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.4.1 Single-Phase Inverter — Sinusoidal PWM

        A single-phase inverter uses SPWM with a 60 Hz sinusoidal reference and
        a 5 kHz triangular carrier. The modulation index m_a = 0.8 sets the output
        fundamental amplitude. When the reference exceeds the carrier, the switch
        is ON; otherwise OFF.
        """
    )
    return


@app.cell
def _(np, plt):
    f_ref = 60  # Hz (fundamental)
    f_carrier = 5000  # Hz (switching)
    ma = 0.8  # modulation index
    _Vdc = 400  # DC bus voltage

    t_spwm = np.linspace(0, 1 / f_ref, 10000)  # one fundamental cycle

    # Reference and carrier
    v_ref = ma * np.sin(2 * np.pi * f_ref * t_spwm)
    # Triangular carrier: period = 1/f_carrier
    v_carrier = 2 * np.abs(2 * (t_spwm * f_carrier - np.floor(t_spwm * f_carrier + 0.5))) - 1

    # PWM output
    v_pwm = np.where(v_ref > v_carrier, _Vdc / 2, -_Vdc / 2)

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax4a.plot(t_spwm * 1e3, v_ref, "b-", linewidth=2, label=f"Reference (60 Hz, m_a={ma})")
    ax4a.plot(t_spwm * 1e3, v_carrier, "gray", linewidth=0.5, alpha=0.7, label="Carrier (5 kHz)")
    ax4a.set_ylabel("Normalized Amplitude")
    ax4a.set_title("SPWM: Sinusoidal Reference vs Triangular Carrier")
    ax4a.legend()
    ax4a.grid(True, alpha=0.3)

    ax4b.fill_between(t_spwm * 1e3, v_pwm, step="mid", alpha=0.3, color="orange")
    ax4b.step(t_spwm * 1e3, v_pwm, "orange", linewidth=0.5, where="mid")
    # Overlay fundamental
    _V1_peak = ma * _Vdc / 2
    ax4b.plot(t_spwm * 1e3, _V1_peak * np.sin(2 * np.pi * f_ref * t_spwm), "b--", linewidth=2,
              label=f"Fundamental: V₁ = {_V1_peak:.0f} V peak")
    ax4b.set_xlabel("Time (ms)")
    ax4b.set_ylabel("Output Voltage (V)")
    ax4b.set_title("Inverter Output (PWM Pulses and Fundamental)")
    ax4b.legend()
    ax4b.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4
    return


# --- 10.6.1 Power Losses vs Switching Frequency ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.6.1 Power Losses — Conduction and Switching Losses

        MOSFET power losses consist of conduction losses (I²R_DS(on), constant with
        frequency) and switching losses (proportional to frequency). As switching
        frequency increases, total losses increase — illustrating the efficiency vs.
        size trade-off in converter design.
        """
    )
    return


@app.cell
def _(np, plt):
    Rds_on = 0.05  # Ω
    Id_loss = 10  # A
    P_cond = Id_loss**2 * Rds_on  # 5 W (constant)

    Vds_loss = 48  # V
    t_rise = 30e-9  # 30 ns
    t_fall = 50e-9  # 50 ns
    fsw_range = np.linspace(10e3, 500e3, 500)
    P_sw = 0.5 * Vds_loss * Id_loss * (t_rise + t_fall) * fsw_range
    P_total_loss = P_cond + P_sw

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(fsw_range / 1e3, np.full_like(fsw_range, P_cond), "b--", linewidth=2,
             label=f"Conduction: P_cond = {P_cond:.1f} W")
    ax5.plot(fsw_range / 1e3, P_sw, "r--", linewidth=2, label="Switching: P_sw ∝ f_sw")
    ax5.plot(fsw_range / 1e3, P_total_loss, "k-", linewidth=2.5, label="Total Losses")

    # Mark crossover point
    crossover_idx = np.argmin(np.abs(P_sw - P_cond))
    ax5.plot(fsw_range[crossover_idx] / 1e3, P_total_loss[crossover_idx], "go", markersize=10,
             label=f"Equal at {fsw_range[crossover_idx]/1e3:.0f} kHz")

    ax5.set_xlabel("Switching Frequency (kHz)")
    ax5.set_ylabel("Power Loss (W)")
    ax5.set_title(f"MOSFET Losses (R_DS(on)={Rds_on*1e3:.0f} mΩ, I_D={Id_loss} A, V_DS={Vds_loss} V)")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5
    return


# --- 10.2.3 Rectifier Harmonics: 6-pulse vs 12-pulse ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.2.3 Rectifier Harmonics — 6-Pulse vs 12-Pulse

        A six-pulse rectifier produces characteristic harmonics at h = 6k ± 1
        (5th, 7th, 11th, 13th, ...) with amplitudes ≈ I₁/h. A 12-pulse system
        cancels the 5th and 7th harmonics through 30° phase shifting, reducing
        THD from ~27% to ~12%.
        """
    )
    return


@app.cell
def _(np, plt):
    I1 = 100  # A fundamental current per bridge

    # 6-pulse harmonics: h = 6k±1
    harmonics_6p = [5, 7, 11, 13, 17, 19, 23, 25]
    I_6p = [I1 / h for h in harmonics_6p]

    # 12-pulse: 5th, 7th cancel; 11th, 13th double (relative to 200 A fund.)
    # Harmonics present: h = 12k±1
    harmonics_12p = [11, 13, 23, 25]
    I1_12p = 2 * I1  # total fundamental
    I_12p = [2 * I1 / h for h in harmonics_12p]

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 6))

    # 6-pulse
    bars6 = ax6a.bar(harmonics_6p, I_6p, width=1.2, color="steelblue", edgecolor="black", alpha=0.8)
    for b, h, _i in zip(bars6, harmonics_6p, I_6p):
        ax6a.text(h, _i + 0.5, f"{_i:.1f} A", ha="center", fontsize=9, fontweight="bold")
    thd_6p = np.sqrt(sum(i**2 for i in I_6p)) / I1 * 100
    ax6a.set_xlabel("Harmonic Order")
    ax6a.set_ylabel("Current Amplitude (A)")
    ax6a.set_title(f"6-Pulse Rectifier (I₁ = {I1} A)\nTHD ≈ {thd_6p:.1f}%")
    ax6a.set_xticks(harmonics_6p)
    ax6a.set_ylim(0, 25)
    ax6a.grid(True, alpha=0.3, axis="y")

    # 12-pulse
    # Show cancelled harmonics as dashed outlines
    cancelled = [5, 7, 17, 19]
    ax6b.bar(cancelled, [0] * len(cancelled), width=1.2, color="none",
             edgecolor="red", linestyle="--", linewidth=2, alpha=0.6)
    for h in cancelled:
        ax6b.text(h, 1, "×", ha="center", fontsize=14, color="red", fontweight="bold")

    bars12 = ax6b.bar(harmonics_12p, I_12p, width=1.2, color="darkorange", edgecolor="black", alpha=0.8)
    for b, h, _i in zip(bars12, harmonics_12p, I_12p):
        ax6b.text(h, _i + 0.5, f"{_i:.1f} A", ha="center", fontsize=9, fontweight="bold")
    thd_12p = np.sqrt(sum(i**2 for i in I_12p)) / I1_12p * 100
    ax6b.set_xlabel("Harmonic Order")
    ax6b.set_ylabel("Current Amplitude (A)")
    ax6b.set_title(f"12-Pulse Rectifier (I₁ = {I1_12p} A)\n5th & 7th cancelled, THD ≈ {thd_12p:.1f}%")
    ax6b.set_xticks(sorted(cancelled + harmonics_12p))
    ax6b.set_ylim(0, 25)
    ax6b.grid(True, alpha=0.3, axis="y")

    fig6.tight_layout()
    fig6
    return


# --- 10.4.3 Multilevel Inverters: 2-Level vs 3-Level ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.4.3 Multilevel Inverters — 2-Level vs 3-Level Output

        A three-level NPC inverter with V_dc = 800 V produces steps of ±400 V and 0,
        halving the dv/dt per transition compared to a two-level inverter that
        switches the full 800 V bus. The staircase waveform has lower THD and
        reduced filtering requirements.
        """
    )
    return


@app.cell
def _(np, plt):
    _Vdc = 800  # total DC bus
    f_fund = 60  # Hz
    f_carr = 5000  # Hz carrier
    ma_ml = 0.85  # modulation index
    t_ml = np.linspace(0, 1 / f_fund, 15000)

    # Reference signal
    v_ref_ml = ma_ml * np.sin(2 * np.pi * f_fund * t_ml)

    # Triangular carrier
    v_carr = 2 * np.abs(2 * (t_ml * f_carr - np.floor(t_ml * f_carr + 0.5))) - 1

    # 2-level PWM output
    v_2lvl = np.where(v_ref_ml > v_carr, _Vdc / 2, -_Vdc / 2)

    # 3-level NPC PWM: uses two carriers (upper and lower)
    # Upper carrier: 0 to 1, Lower carrier: -1 to 0
    v_carr_upper = np.where(v_carr >= 0, v_carr, 0)
    v_carr_lower = np.where(v_carr <= 0, v_carr, 0)

    v_3lvl = np.zeros_like(t_ml)
    for _i in range(len(t_ml)):
        if v_ref_ml[_i] >= 0:
            v_3lvl[_i] = _Vdc / 2 if v_ref_ml[_i] > v_carr_upper[_i] else 0
        else:
            v_3lvl[_i] = -_Vdc / 2 if v_ref_ml[_i] < v_carr_lower[_i] else 0

    # Fundamental component
    _V1_peak = ma_ml * _Vdc / 2

    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax7a.fill_between(t_ml * 1e3, v_2lvl, step="mid", alpha=0.2, color="blue")
    ax7a.step(t_ml * 1e3, v_2lvl, "b-", linewidth=0.5, where="mid", label="2-level PWM")
    ax7a.plot(t_ml * 1e3, _V1_peak * np.sin(2 * np.pi * f_fund * t_ml), "r--",
              linewidth=2, label=f"Fundamental ({_V1_peak:.0f} V peak)")
    ax7a.set_ylabel("Output Voltage (V)")
    ax7a.set_title(f"2-Level Inverter: Steps = ±{_Vdc//2} V (dv/dt = {_Vdc} V per transition)")
    ax7a.legend(fontsize=9)
    ax7a.grid(True, alpha=0.3)
    ax7a.set_ylim(-500, 500)

    ax7b.fill_between(t_ml * 1e3, v_3lvl, step="mid", alpha=0.2, color="darkorange")
    ax7b.step(t_ml * 1e3, v_3lvl, color="darkorange", linewidth=0.5, where="mid",
              label="3-level NPC")
    ax7b.plot(t_ml * 1e3, _V1_peak * np.sin(2 * np.pi * f_fund * t_ml), "r--",
              linewidth=2, label=f"Fundamental ({_V1_peak:.0f} V peak)")
    ax7b.set_xlabel("Time (ms)")
    ax7b.set_ylabel("Output Voltage (V)")
    ax7b.set_title(f"3-Level NPC Inverter: Steps = ±{_Vdc//2} V, 0 V (dv/dt = {_Vdc//2} V per transition)")
    ax7b.legend(fontsize=9)
    ax7b.grid(True, alpha=0.3)
    ax7b.set_ylim(-500, 500)

    fig7.tight_layout()
    fig7
    return


# --- 10.7.1 Active PFC: Input Current With and Without PFC ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.7.1 Active PFC — Input Current Shaping

        Without PFC, a capacitor-input rectifier draws peaky current pulses near the
        voltage peaks (THD > 100%, PF ≈ 0.6). With boost PFC, the input current
        is shaped to follow the sinusoidal voltage waveform (THD < 5%, PF > 0.99).
        """
    )
    return


@app.cell
def _(np, plt):
    f_ac_pfc = 50  # Hz
    Vrms_pfc = 230
    Vpk_pfc = Vrms_pfc * np.sqrt(2)
    P_load = 500  # W
    t_pfc = np.linspace(0, 2 / f_ac_pfc, 4000)

    _v_ac = Vpk_pfc * np.sin(2 * np.pi * f_ac_pfc * t_pfc)

    # Without PFC: capacitor-input rectifier draws peaky current
    # Current flows only when |v_ac| > V_cap (cap voltage near Vpk)
    V_cap = 0.92 * Vpk_pfc  # capacitor stays near peak
    i_no_pfc = np.zeros_like(t_pfc)
    for _i, _t in enumerate(t_pfc):
        v = abs(_v_ac[_i])
        if v > V_cap:
            # Current pulse proportional to voltage excess
            i_no_pfc[_i] = (v - V_cap) * 0.5  # arbitrary scaling for visual
        else:
            i_no_pfc[_i] = 0
    # Sign follows AC polarity
    i_no_pfc = i_no_pfc * np.sign(_v_ac)
    # Scale to match average power
    i_rms_no = np.sqrt(np.mean(i_no_pfc**2))
    if i_rms_no > 0:
        i_no_pfc = i_no_pfc * (P_load / Vrms_pfc) / i_rms_no * 2.5

    # With PFC: sinusoidal current in phase with voltage
    I_pk_pfc = P_load / Vrms_pfc * np.sqrt(2)
    i_with_pfc = I_pk_pfc * np.sin(2 * np.pi * f_ac_pfc * t_pfc)

    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Without PFC
    ax8a.plot(t_pfc * 1e3, _v_ac / Vpk_pfc * 8, "b-", linewidth=1, alpha=0.4,
              label="AC voltage (scaled)")
    ax8a.plot(t_pfc * 1e3, i_no_pfc, "r-", linewidth=1.5,
              label="Input current (no PFC)")
    ax8a.set_ylabel("Current (A)")
    ax8a.set_title("Without PFC: Peaky Current Pulses (THD > 100%, PF ≈ 0.6)")
    ax8a.legend(fontsize=9)
    ax8a.grid(True, alpha=0.3)

    # With PFC
    ax8b.plot(t_pfc * 1e3, _v_ac / Vpk_pfc * I_pk_pfc, "b-", linewidth=1, alpha=0.4,
              label="AC voltage (scaled)")
    ax8b.plot(t_pfc * 1e3, i_with_pfc, "g-", linewidth=2,
              label=f"Input current (with PFC, I_pk = {I_pk_pfc:.2f} A)")
    ax8b.set_xlabel("Time (ms)")
    ax8b.set_ylabel("Current (A)")
    ax8b.set_title("With Boost PFC: Sinusoidal Current (THD < 5%, PF > 0.99)")
    ax8b.legend(fontsize=9)
    ax8b.grid(True, alpha=0.3)

    fig8.tight_layout()
    fig8
    return


# --- 10.8.1 Battery Cell Characteristics: Pack Voltage vs SOC ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.8.1 Battery Cell Characteristics — Pack Voltage vs SOC

        A 96-series NMC pack (3.7 V nominal, 50 Ah) has an open-circuit voltage
        that varies with state of charge. Under a 150 A (3C) load, the terminal
        voltage drops by I × R_pack = 150 × 0.192 = 28.8 V due to internal
        resistance, reducing usable voltage across the full SOC range.
        """
    )
    return


@app.cell
def _(np, plt):
    # NMC single-cell OCV vs SOC (typical curve)
    soc_pct = np.linspace(0, 100, 200)
    soc_frac = soc_pct / 100.0
    # Empirical NMC OCV model: ~3.0 V at 0% to ~4.2 V at 100%
    ocv_cell = 3.0 + 1.2 * soc_frac - 0.35 * (1 - soc_frac) * np.exp(-20 * soc_frac) \
               + 0.05 * np.log(soc_frac + 0.001) + 0.15 * soc_frac**2

    n_cells = 96
    R_pack = 0.192  # Ω (96 × 2 mΩ)
    I_load = 150    # A (3C)

    ocv_pack = ocv_cell * n_cells
    v_loaded = ocv_pack - I_load * R_pack

    fig_bms1, ax_bms1 = plt.subplots(figsize=(10, 5))
    ax_bms1.plot(soc_pct, ocv_pack, "b-", linewidth=2, label="Open-Circuit Voltage (no load)")
    ax_bms1.plot(soc_pct, v_loaded, "r--", linewidth=2, label=f"Terminal Voltage (150 A load, IR = {I_load * R_pack:.1f} V)")

    # Annotate nominal point at ~50% SOC
    idx_50 = 100  # index for 50% SOC
    ax_bms1.annotate(f"Nominal: {ocv_pack[idx_50]:.0f} V",
                     xy=(50, ocv_pack[idx_50]), xytext=(60, ocv_pack[idx_50] + 15),
                     fontsize=10, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue"))
    ax_bms1.annotate(f"Loaded: {v_loaded[idx_50]:.0f} V",
                     xy=(50, v_loaded[idx_50]), xytext=(60, v_loaded[idx_50] - 15),
                     fontsize=10, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

    ax_bms1.fill_between(soc_pct, v_loaded, ocv_pack, alpha=0.1, color="orange", label="IR voltage drop")
    ax_bms1.set_xlabel("State of Charge (%)")
    ax_bms1.set_ylabel("Pack Voltage (V)")
    ax_bms1.set_title(f"96s NMC Battery Pack: OCV and Loaded Voltage vs SOC")
    ax_bms1.legend(fontsize=9)
    ax_bms1.grid(True, alpha=0.3)
    ax_bms1.set_xlim(0, 100)
    fig_bms1.tight_layout()
    fig_bms1
    return


# --- 10.8.2 Cell Balancing: Passive Balancing Convergence ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.8.2 Cell Balancing — Passive Balancing Convergence

        A 12-series pack has cells ranging from 3.95 V to 4.18 V at end of charge.
        Passive balancing with 33 Ω bleed resistors discharges the higher cells
        toward the lowest cell voltage, with each cell decaying exponentially
        through its bleed resistor.
        """
    )
    return


@app.cell
def _(np, plt):
    R_bleed = 33  # Ω
    C_cell_ah = 50  # Ah (large cell, slow voltage change)
    # For visualization, use a simplified model: voltage drops linearly with
    # charge removed, and balancing current = V/R
    # Time constant is very long for real cells; we compress for visualization

    # Initial cell voltages (12 cells, spread from 3.95 to 4.18 V)
    v_init = np.array([3.95, 3.97, 3.99, 4.01, 4.03, 4.05,
                       4.08, 4.10, 4.12, 4.14, 4.16, 4.18])
    v_target = v_init.min()  # balance to lowest cell

    _t_min = np.linspace(0, 30, 500)  # 30 minutes

    fig_bms2, ax_bms2 = plt.subplots(figsize=(10, 5))
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(v_init)))

    for idx, v0 in enumerate(v_init):
        if v0 > v_target:
            # Use a visualization-friendly time constant (~10 min for largest delta)
            # Real balancing of 50 Ah cells takes hours; compressed here for clarity
            tau = 10.0 * (v0 - v_target) / (v_init.max() - v_target)
            tau = max(tau, 3)
            v_t = v_target + (v0 - v_target) * np.exp(-_t_min / tau)
        else:
            v_t = np.full_like(_t_min, v0)
        ax_bms2.plot(_t_min, v_t, linewidth=1.5, color=colors[idx],
                     label=f"Cell {idx+1}: {v0:.2f} V" if idx % 3 == 0 or idx == 11 else None)

    ax_bms2.axhline(y=v_target, color="green", linestyle="--", alpha=0.6, linewidth=1.5,
                    label=f"Target: {v_target:.2f} V")
    ax_bms2.set_xlabel("Balancing Time (minutes)")
    ax_bms2.set_ylabel("Cell Voltage (V)")
    ax_bms2.set_title(f"Passive Cell Balancing: 12 Cells with {R_bleed} Ω Bleed Resistors")
    ax_bms2.legend(fontsize=9, loc="upper right")
    ax_bms2.grid(True, alpha=0.3)
    ax_bms2.set_ylim(3.93, 4.20)
    fig_bms2.tight_layout()
    fig_bms2
    return


# --- 10.8.3 State of Charge Estimation: Coulomb Counting with Drift ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.8.3 SOC Estimation — Coulomb Counting with Sensor Drift

        A 100 Ah pack starts at 85% SOC. Coulomb counting integrates measured
        current, but a ±0.5% sensor accuracy causes cumulative drift. After
        60 Ah discharged, the worst-case SOC error is ±0.30%, growing linearly
        with total charge transferred.
        """
    )
    return


@app.cell
def _(np, plt):
    Q_full = 100  # Ah rated capacity
    soc_init = 85  # %
    sensor_error = 0.005  # ±0.5%

    q_discharged = np.linspace(0, 80, 300)  # Ah discharged
    soc_true = soc_init - (q_discharged / Q_full) * 100

    # Drift error grows linearly with charge transferred
    drift_ah = sensor_error * q_discharged  # Ah of error
    drift_soc = (drift_ah / Q_full) * 100   # % SOC error

    # Exaggerate by 10× for visual clarity; annotation shows true values
    drift_viz = drift_soc * 10
    soc_estimated = soc_true + drift_viz * 0.7  # biased high

    fig_bms3, ax_bms3 = plt.subplots(figsize=(10, 5))
    ax_bms3.plot(q_discharged, soc_true, "b-", linewidth=2, label="True SOC")
    ax_bms3.plot(q_discharged, soc_estimated, "r--", linewidth=2, label="Estimated SOC (drift exaggerated 10×)")

    # Uncertainty band (exaggerated for visibility)
    ax_bms3.fill_between(q_discharged, soc_true - drift_viz, soc_true + drift_viz,
                         alpha=0.15, color="red", label="Uncertainty band (exaggerated 10×)")

    # Annotate at 60 Ah point with true error
    idx_60 = np.argmin(np.abs(q_discharged - 60))
    ax_bms3.annotate(f"At 60 Ah: true error = ±{drift_soc[idx_60]:.2f}%\n(shown 10× for visibility)",
                     xy=(60, soc_true[idx_60]), xytext=(30, soc_true[idx_60] + 12),
                     fontsize=10, color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax_bms3.set_xlabel("Charge Discharged (Ah)")
    ax_bms3.set_ylabel("State of Charge (%)")
    ax_bms3.set_title("Coulomb Counting SOC Estimation with ±0.5% Current Sensor Drift")
    ax_bms3.legend(fontsize=9)
    ax_bms3.grid(True, alpha=0.3)
    ax_bms3.set_xlim(0, 80)
    ax_bms3.set_ylim(0, 90)
    fig_bms3.tight_layout()
    fig_bms3
    return


# --- 10.9.3 BESS Frequency Regulation Droop Response ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.9.3 BESS Frequency Regulation — Droop Response

        A 20 MW BESS with 4% droop and ±0.036 Hz deadband responds to grid
        frequency deviations. The BESS discharges (positive power) when frequency
        drops below the deadband and charges (negative power) when frequency rises
        above it. The droop characteristic is linear up to rated power.
        """
    )
    return


@app.cell
def _(np, plt):
    P_rated_bess = 20
    R_droop = 0.04
    f_nom = 60.0
    deadband = 0.036

    freq = np.linspace(59.5, 60.5, 500)
    delta_f = freq - f_nom
    delta_p = np.zeros_like(delta_f)
    for _i, df in enumerate(delta_f):
        if abs(df) > deadband:
            df_eff = abs(df) - deadband
            _power = P_rated_bess * df_eff / (f_nom * R_droop)
            _power = min(_power, P_rated_bess)
            delta_p[_i] = -np.sign(df) * _power

    fig_droop, ax_droop = plt.subplots(figsize=(10, 5))
    ax_droop.plot(freq, delta_p, "b-", linewidth=2)
    ax_droop.axhline(y=0, color="black", linewidth=0.8)
    ax_droop.axvline(x=f_nom, color="gray", linestyle="--", alpha=0.5, label="Nominal 60 Hz")
    ax_droop.axvspan(f_nom - deadband, f_nom + deadband, alpha=0.1, color="green", label=f"Deadband ±{deadband} Hz")
    ax_droop.plot(59.85, 0.95, "ro", markersize=10, zorder=5)
    ax_droop.annotate("59.85 Hz → 0.95 MW\n(discharge)", xy=(59.85, 0.95), xytext=(59.6, 4),
                fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_droop.plot(60.10, -0.533, "bs", markersize=10, zorder=5)
    ax_droop.annotate("60.10 Hz → −0.533 MW\n(charge)", xy=(60.10, -0.533), xytext=(60.2, -4),
                fontsize=9, color="blue", arrowprops=dict(arrowstyle="->", color="blue"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_droop.fill_between(freq, 0, delta_p, where=(delta_p > 0), alpha=0.15, color="red", label="Discharge (support)")
    ax_droop.fill_between(freq, 0, delta_p, where=(delta_p < 0), alpha=0.15, color="blue", label="Charge (absorb)")
    ax_droop.set_xlabel("Grid Frequency (Hz)")
    ax_droop.set_ylabel("BESS Power Output (MW)")
    ax_droop.set_title(f"BESS Frequency Regulation: {P_rated_bess} MW, {R_droop*100:.0f}% Droop, ±{deadband} Hz Deadband")
    ax_droop.legend(fontsize=9, loc="upper right")
    ax_droop.grid(True, alpha=0.3)
    ax_droop.set_xlim(59.5, 60.5)
    ax_droop.set_ylim(-12, 12)
    fig_droop.tight_layout()
    fig_droop
    return


# --- 10.9.4 BESS Peak Shaving ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.9.4 BESS Peak Shaving — Load Profile

        A commercial facility with 5.0 MW peak demand uses a 1.5 MW / 3.0 MWh
        BESS to shave the peak to 3.5 MW. The shaded area shows the energy
        the BESS must discharge, saving $27,000/month in demand charges.
        """
    )
    return


@app.cell
def _(np, plt):
    hours = np.linspace(0, 24, 288)
    base_load = 2.0 + 0.5 * np.sin(2 * np.pi * (hours - 6) / 24)
    peak_bump = 2.5 * np.exp(-0.5 * ((hours - 14) / 1.5)**2)
    load_profile = base_load + peak_bump
    load_profile = np.clip(load_profile, 1.5, 5.0)

    threshold = 3.5
    shaved_load = np.minimum(load_profile, threshold)

    fig_ps, ax_ps = plt.subplots(figsize=(10, 5))
    ax_ps.plot(hours, load_profile, "b-", linewidth=2, label="Original Load Profile")
    ax_ps.plot(hours, shaved_load, "g-", linewidth=2, label=f"Load with BESS (capped at {threshold} MW)")
    ax_ps.fill_between(hours, shaved_load, load_profile, where=(load_profile > threshold),
                    alpha=0.3, color="red", label="BESS Discharge (peak shaved)")
    ax_ps.axhline(y=threshold, color="green", linestyle="--", alpha=0.6, linewidth=1.5)
    ax_ps.axhline(y=5.0, color="red", linestyle=":", alpha=0.5)
    ax_ps.annotate("Peak: 5.0 MW", xy=(14, 5.0), xytext=(16, 4.7),
                fontsize=10, color="red", arrowprops=dict(arrowstyle="->", color="red"))
    ax_ps.annotate(f"Target: {threshold} MW", xy=(8, threshold), xytext=(3, 4.0),
                fontsize=10, color="green", arrowprops=dict(arrowstyle="->", color="green"))
    ax_ps.annotate("Demand charge\nsavings: $27k/mo", xy=(14, 4.2), xytext=(18, 4.5),
                fontsize=9, color="darkred",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_ps.set_xlabel("Hour of Day")
    ax_ps.set_ylabel("Demand (MW)")
    ax_ps.set_title("Commercial Peak Shaving: 1.5 MW / 3.0 MWh BESS")
    ax_ps.legend(fontsize=9, loc="upper left")
    ax_ps.grid(True, alpha=0.3)
    ax_ps.set_xlim(0, 24)
    ax_ps.set_ylim(0, 5.5)
    fig_ps.tight_layout()
    fig_ps
    return


# --- 10.9.5 BESS Capacity Degradation and LCOS ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.9.5 BESS Capacity Degradation and LCOS

        A 100 MWh LFP BESS degrades at 2.5% per year over 15 years. The left
        panel shows capacity fade from 100 MWh to 62.5 MWh. The right panel
        shows the levelized cost of storage converging to $82.1/MWh, well below
        the $140/MWh stacked revenue target.
        """
    )
    return


@app.cell
def _(np, plt):
    years_bess = np.arange(0, 16)
    degradation_rate = 0.025
    capacity_bess = 100 * (1 - degradation_rate * years_bess)

    fig_deg, (ax_d1, ax_d2) = plt.subplots(1, 2, figsize=(12, 5))

    ax_d1.bar(years_bess, capacity_bess, color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.5)
    ax_d1.axhline(y=81.25, color="orange", linestyle="--", linewidth=1.5, label="15-yr average: 81.25 MWh")
    ax_d1.axhline(y=62.5, color="red", linestyle=":", linewidth=1.5, label="End-of-life: 62.5 MWh (37.5% fade)")
    ax_d1.set_xlabel("Year")
    ax_d1.set_ylabel("Usable Capacity (MWh)")
    ax_d1.set_title("100 MWh LFP BESS: 2.5%/yr Capacity Fade")
    ax_d1.legend(fontsize=9)
    ax_d1.grid(True, alpha=0.3, axis="y")
    ax_d1.set_xlim(-0.5, 15.5)
    ax_d1.set_ylim(0, 110)

    capital_bess = 28_000_000
    om_annual = np.array([cap * 0.80 * 365 * 6 for cap in capacity_bess[1:]])
    cum_energy = np.cumsum([cap * 0.80 * 365 for cap in capacity_bess[1:]])
    lcos_over_time = np.zeros(15)
    for n in range(15):
        pv_om = sum(om_annual[:n+1] / (1.08)**(np.arange(1, n+2)))
        lcos_over_time[n] = (capital_bess + pv_om) / cum_energy[n] if cum_energy[n] > 0 else 0

    ax_d2.plot(years_bess[1:], lcos_over_time, "r-", linewidth=2, marker="o", markersize=5)
    ax_d2.axhline(y=82.1, color="blue", linestyle="--", alpha=0.6, label="15-yr LCOS: $82.1/MWh")
    ax_d2.axhline(y=140, color="green", linestyle=":", alpha=0.6, label="Revenue: $140/MWh")
    ax_d2.fill_between(years_bess[1:], lcos_over_time, 140, where=(lcos_over_time < 140),
                     alpha=0.1, color="green", label="Profit margin")
    ax_d2.set_xlabel("Project Year")
    ax_d2.set_ylabel("LCOS ($/MWh)")
    ax_d2.set_title("Levelized Cost of Storage Over Project Life")
    ax_d2.legend(fontsize=9, loc="upper right")
    ax_d2.grid(True, alpha=0.3)
    ax_d2.set_xlim(0.5, 15.5)
    ax_d2.set_ylim(0, 300)

    fig_deg.tight_layout()
    fig_deg
    return


# ============================================================
# 10.10 Battery Charging
# ============================================================

# --- 10.10.1 CC/CV Charging Profile ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.10.1 CC/CV Charging Profile — 21700 NMC Cell

        A 5.0 Ah NMC cell is charged using the CC/CV algorithm at 0.5C (2.5 A)
        with a 4.20 V cutoff. During CC the voltage rises steadily; at the cutoff
        the charger switches to CV and the current tapers exponentially. Charge
        terminates at C/20 (0.25 A). The CC phase delivers ~80% of capacity in
        1.6 h; the CV tail adds the remaining ~20% over ~55 min.
        """
    )
    return


@app.cell
def _(np, plt):
    # CC/CV charging parameters (Example 10.10.1)
    C_rated = 5.0  # Ah
    I_cc = 2.5     # A (0.5C)
    V_cutoff = 4.20  # V
    V_start = 3.0    # V
    t_cc = 1.6       # hours to reach cutoff
    tau_cv = 0.4     # CV time constant (hours)
    I_term = 0.25    # termination current (C/20)

    # CC phase
    _dt = 0.001  # hours
    t_cc_arr = np.arange(0, t_cc, _dt)
    # Voltage rises approximately linearly from 3.0 to 4.20 V
    v_cc = V_start + (V_cutoff - V_start) * (t_cc_arr / t_cc) ** 0.7
    i_cc = np.full_like(t_cc_arr, I_cc)

    # CV phase — exponential taper
    t_cv_end = -tau_cv * np.log(I_term / I_cc)  # 0.92 h
    t_cv_arr = np.arange(0, t_cv_end, _dt)
    i_cv = I_cc * np.exp(-t_cv_arr / tau_cv)
    v_cv = np.full_like(t_cv_arr, V_cutoff)

    # Combine
    t_total = np.concatenate([t_cc_arr, t_cc + t_cv_arr])
    v_total = np.concatenate([v_cc, v_cv])
    i_total = np.concatenate([i_cc, i_cv])

    # Cumulative charge (Ah)
    q_total = np.cumsum(i_total * _dt)

    fig_cc, (ax_v, ax_i, ax_q) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Voltage
    ax_v.plot(t_total, v_total, "b-", linewidth=2)
    ax_v.axhline(y=V_cutoff, color="red", linestyle="--", alpha=0.5, label=f"Cutoff {V_cutoff} V")
    ax_v.axvline(x=t_cc, color="gray", linestyle=":", alpha=0.7)
    ax_v.set_ylabel("Cell Voltage (V)")
    ax_v.set_title("CC/CV Charging Profile — 5.0 Ah NMC 21700 Cell at 0.5C")
    ax_v.legend(fontsize=9)
    ax_v.grid(True, alpha=0.3)
    ax_v.set_ylim(2.8, 4.4)
    ax_v.text(t_cc / 2, 3.1, "CC Phase", ha="center", fontsize=11, fontweight="bold", color="green")
    ax_v.text(t_cc + t_cv_end / 2, 3.1, "CV Phase", ha="center", fontsize=11, fontweight="bold", color="purple")

    # Current
    ax_i.plot(t_total, i_total, "r-", linewidth=2)
    ax_i.axhline(y=I_term, color="orange", linestyle="--", alpha=0.5, label=f"Termination {I_term} A (C/20)")
    ax_i.axvline(x=t_cc, color="gray", linestyle=":", alpha=0.7)
    ax_i.set_ylabel("Charge Current (A)")
    ax_i.legend(fontsize=9)
    ax_i.grid(True, alpha=0.3)
    ax_i.set_ylim(0, 3.0)

    # Cumulative charge
    ax_q.plot(t_total, q_total, "g-", linewidth=2)
    ax_q.axhline(y=C_rated, color="gray", linestyle="--", alpha=0.5, label=f"Rated {C_rated} Ah")
    ax_q.axvline(x=t_cc, color="gray", linestyle=":", alpha=0.7)
    q_at_cc = I_cc * t_cc
    ax_q.annotate(f"CC: {q_at_cc:.2f} Ah ({q_at_cc/C_rated*100:.0f}%)",
                  xy=(t_cc, q_at_cc), xytext=(0.5, 4.2),
                  fontsize=9, arrowprops=dict(arrowstyle="->", color="green"),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_q.set_xlabel("Time (hours)")
    ax_q.set_ylabel("Charge Delivered (Ah)")
    ax_q.legend(fontsize=9)
    ax_q.grid(True, alpha=0.3)

    fig_cc.tight_layout()
    fig_cc
    return


# --- 10.10.3 EV Charging Level Comparison ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.10.3 EV Charging Level Comparison

        Time to charge a 75 kWh battery from 10% to 80% SOC across Level 1
        (1.4 kW), Level 2 (7.7 kW), and DCFC (150 kW). The dramatic difference
        in charge times illustrates why DC fast charging is essential for
        long-distance travel, while Level 2 is adequate for overnight home charging.
        """
    )
    return


@app.cell
def _(np, plt):
    battery_kwh = 75
    soc_start = 0.10
    soc_end = 0.80
    energy_needed = battery_kwh * (soc_end - soc_start)  # 52.5 kWh

    levels = ["Level 1\n(1.4 kW)", "Level 2\n(7.7 kW)", "DCFC\n(150 kW)"]
    powers = [1.4, 7.7, 150]
    times_h = [energy_needed / p for p in powers]

    colors_ev = ["#CC6666", "#6699CC", "#33AA66"]

    fig_ev, (ax_bar, ax_soc) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart — charge times
    bars = ax_bar.bar(levels, times_h, color=colors_ev, edgecolor="black", linewidth=0.8)
    for _bar, _t in zip(bars, times_h):
        if _t > 1:
            label = f"{_t:.1f} h"
        else:
            label = f"{_t*60:.0f} min"
        ax_bar.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.5,
                    label, ha="center", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Charge Time (hours)")
    ax_bar.set_title(f"Time to Charge {battery_kwh} kWh Battery: {soc_start*100:.0f}% → {soc_end*100:.0f}% SOC")
    ax_bar.grid(True, alpha=0.3, axis="y")
    ax_bar.set_ylim(0, max(times_h) * 1.15)

    # SOC vs time curves
    t_max = 10  # hours
    t_arr = np.linspace(0, t_max, 1000)
    for name, _power, color in zip(levels, powers, colors_ev):
        soc = soc_start + (_power / battery_kwh) * t_arr
        soc = np.minimum(soc, soc_end)
        lbl = name.replace("\n", " ")
        ax_soc.plot(t_arr, soc * 100, linewidth=2, color=color, label=lbl)
    ax_soc.axhline(y=soc_end * 100, color="gray", linestyle="--", alpha=0.5, label="Target 80%")
    ax_soc.set_xlabel("Time (hours)")
    ax_soc.set_ylabel("State of Charge (%)")
    ax_soc.set_title("SOC vs Time by Charging Level")
    ax_soc.legend(fontsize=9)
    ax_soc.grid(True, alpha=0.3)
    ax_soc.set_xlim(0, t_max)
    ax_soc.set_ylim(0, 100)

    fig_ev.tight_layout()
    fig_ev
    return


# --- 10.10.4 Fast Charging Thermal Analysis ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.10.4 Fast Charging — Thermal Comparison at 25°C vs 5°C

        A 280 Ah LFP cell charged at 1C (280 A) generates 47 W at 25°C
        (R_int = 0.6 mΩ) but 117.6 W at 5°C (R_int = 1.5 mΩ) due to
        increased electrolyte viscosity. The adiabatic temperature rise is
        2.5× faster at low temperature, with the cell reaching 45°C in just
        34 minutes — creating a risk of lithium plating if not pre-conditioned.
        """
    )
    return


@app.cell
def _(np, plt):
    # Cell parameters (Example 10.10.4)
    capacity = 280  # Ah
    I_charge = 280  # A (1C)
    R_25 = 0.6e-3   # Ω at 25°C
    R_5 = 1.5e-3    # Ω at 5°C
    mass = 5.5       # kg
    cp = 1100        # J/(kg·°C)
    T_max_cell = 45  # °C

    Q_25 = I_charge**2 * R_25  # 47.0 W
    Q_5 = I_charge**2 * R_5    # 117.6 W

    _t_min = np.linspace(0, 60, 500)  # minutes
    t_sec = _t_min * 60

    # Adiabatic temperature rise
    T_25 = 25 + Q_25 * t_sec / (mass * cp)
    T_5 = 5 + Q_5 * t_sec / (mass * cp)

    # Time to reach 45°C from 5°C
    t_limit_5 = mass * cp * (T_max_cell - 5) / Q_5 / 60  # minutes

    fig_th, (ax_temp, ax_heat) = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature rise
    ax_temp.plot(_t_min, T_25, "b-", linewidth=2, label="Start at 25°C (R = 0.6 mΩ)")
    ax_temp.plot(_t_min, T_5, "r-", linewidth=2, label="Start at 5°C (R = 1.5 mΩ)")
    ax_temp.axhline(y=T_max_cell, color="red", linestyle="--", alpha=0.6, label=f"Max safe temp {T_max_cell}°C")
    ax_temp.axvline(x=t_limit_5, color="red", linestyle=":", alpha=0.5)
    ax_temp.annotate(f"Limit at {t_limit_5:.1f} min",
                     xy=(t_limit_5, T_max_cell), xytext=(t_limit_5 + 5, 50),
                     fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"),
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_temp.fill_between(_t_min, T_max_cell, 70, alpha=0.08, color="red", label="Plating risk zone")
    ax_temp.set_xlabel("Time (minutes)")
    ax_temp.set_ylabel("Cell Temperature (°C)")
    ax_temp.set_title(f"Adiabatic Temperature Rise — 280 Ah LFP at 1C")
    ax_temp.legend(fontsize=9, loc="upper left")
    ax_temp.grid(True, alpha=0.3)
    ax_temp.set_xlim(0, 60)
    ax_temp.set_ylim(0, 70)

    # Heat generation comparison bar chart
    conditions = ["25°C\n(0.6 mΩ)", "5°C\n(1.5 mΩ)"]
    heats = [Q_25, Q_5]
    colors_th = ["#6699CC", "#CC4444"]
    bars_th = ax_heat.bar(conditions, heats, color=colors_th, edgecolor="black", linewidth=0.8, width=0.5)
    for _bar, q in zip(bars_th, heats):
        ax_heat.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 2,
                     f"{q:.1f} W", ha="center", fontsize=12, fontweight="bold")
    ax_heat.set_ylabel("I²R Heat Generation (W)")
    ax_heat.set_title("Heat Generation: I²R at 280 A (1C)")
    ax_heat.grid(True, alpha=0.3, axis="y")
    ax_heat.set_ylim(0, 140)
    ax_heat.annotate(f"2.5× increase", xy=(1, Q_5 / 2), fontsize=11, fontweight="bold",
                     color="darkred", ha="center")

    fig_th.tight_layout()
    fig_th
    return


# --- 10.10.5 Wireless Charging — Efficiency vs Coupling Coefficient ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.10.5 Wireless Charging — Efficiency vs Coupling Coefficient

        Maximum theoretical coil-to-coil efficiency for a wireless charging
        system depends strongly on the coupling coefficient k and the coil
        quality factors Q₁, Q₂. For the SAE J2954 example (Q₁ = 712, Q₂ = 890),
        efficiency remains above 90% even at k = 0.1, but drops sharply for
        lower-Q consumer (Qi) coils. The shaded region shows the typical
        coupling range for EV wireless charging (k = 0.1–0.3).
        """
    )
    return


@app.cell
def _(np, plt):
    k_range = np.linspace(0.01, 0.8, 500)

    # SAE J2954 EV system (Example 10.10.5)
    Q1_ev, Q2_ev = 712, 890

    # Qi consumer system (Problem 10.10.10)
    Q1_qi, Q2_qi = 31.4, 11.3

    # Mid-range system
    Q1_mid, Q2_mid = 200, 250

    def eta_max(k, Q1, Q2):
        kQQ = k**2 * Q1 * Q2
        return kQQ / (1 + np.sqrt(1 + kQQ))**2

    eta_ev = eta_max(k_range, Q1_ev, Q2_ev) * 100
    eta_qi = eta_max(k_range, Q1_qi, Q2_qi) * 100
    eta_mid = eta_max(k_range, Q1_mid, Q2_mid) * 100

    fig_wpt, ax_wpt = plt.subplots(figsize=(10, 6))
    ax_wpt.plot(k_range, eta_ev, "b-", linewidth=2, label=f"EV (Q₁={Q1_ev}, Q₂={Q2_ev})")
    ax_wpt.plot(k_range, eta_mid, "g-", linewidth=2, label=f"Mid-range (Q₁={Q1_mid}, Q₂={Q2_mid})")
    ax_wpt.plot(k_range, eta_qi, "r-", linewidth=2, label=f"Qi consumer (Q₁={Q1_qi:.0f}, Q₂={Q2_qi:.0f})")

    # EV coupling range
    ax_wpt.axvspan(0.1, 0.3, alpha=0.1, color="blue", label="EV coupling range (k = 0.1–0.3)")
    # Qi coupling range
    ax_wpt.axvspan(0.5, 0.8, alpha=0.1, color="red", label="Qi coupling range (k = 0.5–0.8)")

    # Mark the example points
    k_ex = 0.20
    eta_ex = eta_max(k_ex, Q1_ev, Q2_ev) * 100
    ax_wpt.plot(k_ex, eta_ex, "bo", markersize=10, zorder=5)
    ax_wpt.annotate(f"Example 10.10.5\nk = {k_ex}, η = {eta_ex:.1f}%",
                    xy=(k_ex, eta_ex), xytext=(0.35, 96),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="blue"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    k_qi_ex = 0.55
    eta_qi_ex = eta_max(k_qi_ex, Q1_qi, Q2_qi) * 100
    ax_wpt.plot(k_qi_ex, eta_qi_ex, "rs", markersize=10, zorder=5)
    ax_wpt.annotate(f"Problem 10.10.10\nk = {k_qi_ex}, η = {eta_qi_ex:.1f}%",
                    xy=(k_qi_ex, eta_qi_ex), xytext=(0.55, 70),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="red"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax_wpt.set_xlabel("Coupling Coefficient (k)")
    ax_wpt.set_ylabel("Maximum Theoretical Efficiency (%)")
    ax_wpt.set_title("Wireless Charging Efficiency vs Coupling Coefficient")
    ax_wpt.legend(fontsize=9, loc="lower right")
    ax_wpt.grid(True, alpha=0.3)
    ax_wpt.set_xlim(0, 0.8)
    ax_wpt.set_ylim(0, 100)

    fig_wpt.tight_layout()
    fig_wpt
    return


# --- 10.11.2 Ragone Plot — Energy Storage Technology Comparison ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.11.2 Ragone Plot — Energy Storage Technology Comparison

        A Ragone plot compares energy storage technologies by plotting specific
        power (W/kg) vs specific energy (Wh/kg) on a log-log scale.  Each
        shaded region shows the characteristic performance envelope of a
        technology.  Supercapacitors (EDLC) bridge the gap between conventional
        capacitors (extreme power, minimal energy) and batteries (high energy,
        limited power).  Lithium-ion capacitors (LIC) extend the EDLC region
        toward higher energy density.
        """
    )
    return


@app.cell
def _(plt):
    import matplotlib.patches as _mpatches

    # Technology regions: [E_min, E_max, P_min, P_max, label, color]
    _technologies = [
        (1e-3,  0.05,   1e6,   1e8,   "Conventional\nCapacitor",  "#9B59B6"),
        (1,     15,     500,   1e4,   "EDLC\n(Supercapacitor)",   "#2980B9"),
        (5,     30,     200,   5e3,   "Pseudocapacitor /\nHybrid", "#27AE60"),
        (10,    30,     100,   2e3,   "Lithium-Ion\nCapacitor",    "#1ABC9C"),
        (20,    40,     50,    300,   "Lead-Acid\nBattery",        "#E67E22"),
        (100,   250,    100,   1000,  "Li-Ion\nBattery",           "#E74C3C"),
        (150,   400,    50,    500,   "Li-S / Li-Air\n(emerging)", "#C0392B"),
    ]

    fig_rag, _ax_rag = plt.subplots(figsize=(10, 7))

    for _emin, _emax, _pmin, _pmax, _lbl, _col in _technologies:
        _rect = _mpatches.FancyBboxPatch(
            (_emin, _pmin),
            _emax - _emin, _pmax - _pmin,
            boxstyle="round,pad=0",
            linewidth=1.5,
            edgecolor=_col,
            facecolor=_col,
            alpha=0.20,
            transform=_ax_rag.transData,
        )
        _ax_rag.add_patch(_rect)
        # Label inside the box (geometric center on log scale)
        import numpy as _np_rag
        _xe = (_emin * _emax) ** 0.5
        _xp = (_pmin * _pmax) ** 0.5
        _ax_rag.text(_xe, _xp, _lbl, ha="center", va="center",
                     fontsize=8.5, color=_col, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    # Mark the Example 10.11.1 cell (750 F, 2.7 V, ~60 g → E≈7.5 Wh/kg)
    _e_ex = 0.759 / 0.060 * 1e-3 * 1e3   # 12.65 Wh/kg (using 60 g cell from §10.11.2 example)
    _e_ex = 5.9   # from Example 10.11.2
    _p_ex = 1312 / 0.060   # P_max from §10.11.3 example, ~22 kW/kg
    _p_ex = min(_p_ex, 8000)  # cap for visibility
    _ax_rag.plot(_e_ex, _p_ex, "b^", markersize=11, zorder=6,
                 label="Example 10.11.2 EDLC\n(5.9 Wh/kg, est. power)")
    _ax_rag.annotate(
        "§10.11.2\nEDLC example",
        xy=(_e_ex, _p_ex), xytext=(2, 12000),
        fontsize=8, color="#2980B9",
        arrowprops=dict(arrowstyle="->", color="#2980B9"),
    )

    _ax_rag.set_xscale("log")
    _ax_rag.set_yscale("log")
    _ax_rag.set_xlabel("Specific Energy (Wh/kg)", fontsize=11)
    _ax_rag.set_ylabel("Specific Power (W/kg)", fontsize=11)
    _ax_rag.set_title(
        "Ragone Plot — Energy Storage Technology Comparison\n"
        "(Approximate performance envelopes; actual values vary by product and condition)",
        fontsize=11,
    )
    _ax_rag.set_xlim(5e-4, 600)
    _ax_rag.set_ylim(30, 2e8)
    _ax_rag.grid(True, which="both", alpha=0.25)

    fig_rag.tight_layout()
    fig_rag
    return fig_rag


# --- 10.11.3 Supercapacitor Constant-Current Discharge Voltage Profile ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 10.11.3 Supercapacitor Discharge Voltage Profile

        During constant-current discharge the capacitor voltage drops linearly:
        V_C(t) = V₀ − (I/C)·t.  The terminal voltage is shifted down by the
        ESR drop: V_terminal = V_C − I·ESR.  Unlike a battery's flat discharge
        curve, the full voltage swing must be managed — typically with a
        DC-DC converter — or the usable range is limited to V_max/2 to
        extract 75% of stored energy.

        The chart reproduces the conditions of Example 10.11.3:
        C = 10 F, ESR = 50 mΩ, V₀ = 16.2 V, I = 20 A.
        """
    )
    return


@app.cell
def _(plt):
    import numpy as _np_dis

    _C    = 10.0        # F
    _ESR  = 0.050       # Ω
    _V0   = 16.2        # V (6 cells × 2.7 V)
    _I    = 20.0        # A constant discharge current
    _Vmin = _V0 / 2     # 8.1 V — discharge to half voltage

    # Time until V_C reaches Vmin
    _t_end = (_V0 - _Vmin) / (_I / _C)   # 4.05 s

    _t = _np_dis.linspace(0, _t_end + 0.5, 500)

    # Capacitor voltage (ideal, linear)
    _Vc = _np_dis.where(_t <= _t_end, _V0 - (_I / _C) * _t, _Vmin)

    # Terminal voltage (ESR drop during discharge)
    _Vterm = _np_dis.where(_t <= _t_end, _Vc - _I * _ESR, _Vmin)

    # Battery reference (flat discharge at mean voltage for comparison)
    _Vbat = _np_dis.full_like(_t, _np_dis.mean(_Vc[_t <= _t_end]))

    fig_dis, _ax_dis = plt.subplots(figsize=(10, 6))

    _ax_dis.plot(_t, _Vc,   "b-",  linewidth=2.5, label="V_C (capacitor voltage, ideal)")
    _ax_dis.plot(_t, _Vterm, "r--", linewidth=2.0, label=f"V_terminal (with ESR = {_ESR*1000:.0f} mΩ)")
    _ax_dis.plot(_t, _Vbat,  "g:",  linewidth=1.5, alpha=0.7,
                 label=f"Battery reference (flat at {_Vbat[0]:.1f} V)")

    # ESR drop annotation
    _ax_dis.annotate(
        f"ESR drop = I × ESR\n= {_I:.0f} × {_ESR:.3f} = {_I*_ESR:.1f} V",
        xy=(0.1, _V0 - _I * _ESR),
        xytext=(0.8, _V0 - _I * _ESR - 2.5),
        fontsize=9, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    # Midpoint annotation
    _ax_dis.axvline(x=_t_end, color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
    _ax_dis.axhline(y=_Vmin, color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
    _ax_dis.annotate(
        f"t = {_t_end:.2f} s\nV_C = V₀/2 = {_Vmin:.1f} V\n(75% energy extracted)",
        xy=(_t_end, _Vmin),
        xytext=(_t_end - 1.8, _Vmin + 2.5),
        fontsize=9, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    _ax_dis.set_xlabel("Time (s)", fontsize=11)
    _ax_dis.set_ylabel("Voltage (V)", fontsize=11)
    _ax_dis.set_title(
        f"Supercapacitor Constant-Current Discharge (C = {_C} F, ESR = {_ESR*1000:.0f} mΩ, "
        f"V₀ = {_V0} V, I = {_I:.0f} A)\n"
        "Linear V_C profile compared to flat battery discharge",
        fontsize=11,
    )
    _ax_dis.legend(fontsize=9)
    _ax_dis.grid(True, alpha=0.3)
    _ax_dis.set_xlim(0, _t_end + 0.5)
    _ax_dis.set_ylim(0, _V0 + 2)

    fig_dis.tight_layout()
    fig_dis
    return fig_dis


if __name__ == "__main__":
    app.run()
