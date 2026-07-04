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
        # Chapter 13: Operational Amplifiers — Example Visualizations

        Interactive graphs for selected example problems from Chapter 13,
        covering integrator/differentiator circuits, active filters,
        Schmitt trigger hysteresis, and slew rate limiting.
        """
    )
    return


# --- 13.2.3 Integrator: Ramp output ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.2.3 Integrator — Ramp Output from Step Input

        An integrator with R = 10 kΩ and C = 0.1 μF produces a linear ramp
        when driven by a constant −2 V input. The output ramps from 0 V to 12 V
        in t = R × C × ΔV / V_in = 6 ms, giving a slew rate of 2.0 V/ms.
        """
    )
    return


@app.cell
def _(np, plt):
    R_int = 10e3  # Ω
    C_int = 0.1e-6  # F
    Vin_int = -2  # V (constant)
    tau_int = R_int * C_int  # 1 ms

    t_int = np.linspace(0, 8e-3, 1000)
    # V_out = -(1/RC) * integral(Vin dt) = -(Vin/(RC)) * t = 2/0.001 * t = 2000t
    Vout_int = -(Vin_int / (R_int * C_int)) * t_int
    Vout_int = np.minimum(Vout_int, 12)  # clip at 12V (practical saturation)

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    ax1a.plot(t_int * 1e3, np.full_like(t_int, Vin_int), "b-", linewidth=2)
    ax1a.set_ylabel("V_in (V)")
    ax1a.set_title("Integrator: Input Step")
    ax1a.set_ylim(-3, 1)
    ax1a.grid(True, alpha=0.3)

    ax1b.plot(t_int * 1e3, Vout_int, "r-", linewidth=2, label="V_out(t)")
    ax1b.axhline(y=12, color="gray", linestyle="--", alpha=0.5, label="V_out = 12 V")
    ax1b.plot(6, 12, "go", markersize=10, zorder=5)
    ax1b.annotate("t = 6 ms, V_out = 12 V",
                  xy=(6, 12), xytext=(6.3, 9),
                  fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
    ax1b.annotate(f"Slope = {abs(Vin_int)/(R_int*C_int*1e3):.1f} V/ms",
                  xy=(3, 6), fontsize=11, color="red")
    ax1b.set_xlabel("Time (ms)")
    ax1b.set_ylabel("V_out (V)")
    ax1b.set_title("Integrator: Output Ramp (R = 10 kΩ, C = 0.1 μF)")
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1
    return


# --- 13.2.4 Differentiator: Triangular to square ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.2.4 Differentiator — Triangular to Square Wave

        A differentiator with R = 100 kΩ and C = 0.01 μF converts a
        1 kHz triangular wave (2 V peak-to-peak) into a square wave.
        The output is V_out = −RC × dV_in/dt.
        """
    )
    return


@app.cell
def _(np, plt):
    R_diff = 100e3  # Ω
    C_diff = 0.01e-6  # F
    f_diff = 1000  # Hz
    Vpp_diff = 2  # V peak-to-peak

    t_diff = np.linspace(0, 3e-3, 3000)  # 3 cycles

    # Triangular wave (±1 V)
    period_diff = 1 / f_diff
    v_tri = (2 * Vpp_diff / period_diff) * (period_diff / 2 - np.abs(t_diff % period_diff - period_diff / 2)) - Vpp_diff / 2

    # Derivative of triangular wave = square wave
    # Slope of rising: +2*Vpp/T, slope of falling: -2*Vpp/T
    dv_dt = np.gradient(v_tri, t_diff)
    v_out_diff = -R_diff * C_diff * dv_dt

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax2a.plot(t_diff * 1e3, v_tri, "b-", linewidth=2, label="V_in (triangular)")
    ax2a.set_ylabel("V_in (V)")
    ax2a.set_title("Differentiator Input: 1 kHz Triangular Wave (2 Vpp)")
    ax2a.legend()
    ax2a.grid(True, alpha=0.3)

    ax2b.plot(t_diff * 1e3, v_out_diff, "r-", linewidth=2, label="V_out = −RC × dV_in/dt")
    expected_level = R_diff * C_diff * 2 * Vpp_diff * f_diff
    ax2b.axhline(y=expected_level, color="gray", linestyle=":", alpha=0.5,
                 label=f"±{expected_level:.1f} V")
    ax2b.axhline(y=-expected_level, color="gray", linestyle=":", alpha=0.5)
    ax2b.set_xlabel("Time (ms)")
    ax2b.set_ylabel("V_out (V)")
    ax2b.set_title("Differentiator Output: Square Wave (R = 100 kΩ, C = 0.01 μF)")
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2
    return


# --- 13.5.2 Sallen-Key: Second-order Butterworth ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.5.2 Sallen-Key Filter — 2nd-Order Butterworth Response

        A Sallen-Key lowpass with f_c = 1 kHz and Q = 0.707 (Butterworth) produces
        a maximally flat passband with −40 dB/decade roll-off in the stopband.
        The −3 dB point occurs precisely at 1 kHz.
        """
    )
    return


@app.cell
def _(np, plt):
    fc_sk = 1000  # Hz
    Q_sk = 0.707  # Butterworth
    f_sk = np.logspace(1, 5, 1000)  # 10 Hz to 100 kHz
    omega_sk = 2 * np.pi * f_sk
    omega_c_sk = 2 * np.pi * fc_sk

    # 2nd order lowpass: H(jω) = 1 / (1 - (ω/ω_c)² + j(ω/ω_c)/Q)
    s_norm = 1j * f_sk / fc_sk
    H_sk = 1 / (1 + s_norm / Q_sk + s_norm**2)
    H_mag_sk = 20 * np.log10(np.abs(H_sk))
    H_phase_sk = np.degrees(np.angle(H_sk))

    fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 7))

    ax3a.semilogx(f_sk, H_mag_sk, "b-", linewidth=2)
    ax3a.axhline(y=-3, color="red", linestyle="--", alpha=0.6, label="−3 dB")
    ax3a.axvline(x=fc_sk, color="red", linestyle=":", alpha=0.6, label=f"f_c = {fc_sk} Hz")
    # Show -40 dB/decade slope
    f_slope = np.array([5000, 50000])
    slope_line = -40 * np.log10(f_slope / fc_sk) - 3
    ax3a.plot(f_slope, slope_line, "g--", linewidth=1.5, alpha=0.7, label="−40 dB/decade")
    ax3a.set_ylabel("Magnitude (dB)")
    ax3a.set_title("Sallen-Key 2nd-Order Butterworth Lowpass (f_c = 1 kHz, Q = 0.707)")
    ax3a.set_ylim(-80, 5)
    ax3a.legend()
    ax3a.grid(True, alpha=0.3, which="both")

    ax3b.semilogx(f_sk, H_phase_sk, "r-", linewidth=2)
    ax3b.axhline(y=-90, color="gray", linestyle=":", alpha=0.5, label="−90° at f_c")
    ax3b.axvline(x=fc_sk, color="red", linestyle=":", alpha=0.6)
    ax3b.set_xlabel("Frequency (Hz)")
    ax3b.set_ylabel("Phase (°)")
    ax3b.set_ylim(-180, 0)
    ax3b.legend()
    ax3b.grid(True, alpha=0.3, which="both")

    fig3.tight_layout()
    fig3
    return


# --- 13.6.2 Schmitt Trigger: Hysteresis ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.6.2 Schmitt Trigger — Hysteresis Transfer Characteristic

        A Schmitt trigger with V_TH = 3.0 V and V_TL = 2.0 V creates a 1 V
        hysteresis band that prevents output chatter when the input signal is noisy.
        The output switches to +V_sat when V_in rises above V_TH and to −V_sat
        when V_in falls below V_TL.
        """
    )
    return


@app.cell
def _(np, plt):
    VTH = 3.0  # V (upper threshold)
    VTL = 2.0  # V (lower threshold)
    Vsat_pos = 13.5  # V
    Vsat_neg = -13.5  # V

    # Simulate input: slow sine wave
    t_schmitt = np.linspace(0, 4e-3, 4000)
    Vin_schmitt = 2.5 + 1.5 * np.sin(2 * np.pi * 500 * t_schmitt)  # centered at 2.5V

    # Compute Schmitt trigger output
    Vout_schmitt = np.zeros_like(Vin_schmitt)
    state = -1  # start low
    for _i in range(len(Vin_schmitt)):
        if state == -1 and Vin_schmitt[_i] > VTH:
            state = 1
        elif state == 1 and Vin_schmitt[_i] < VTL:
            state = -1
        Vout_schmitt[_i] = Vsat_pos if state == 1 else Vsat_neg

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

    # Transfer characteristic (hysteresis loop)
    ax4a.plot([0, VTL, VTL], [Vsat_neg, Vsat_neg, Vsat_pos], "b-", linewidth=2, label="Rising")
    ax4a.plot([VTH, VTH, 5], [Vsat_pos, Vsat_neg, Vsat_neg], "r-", linewidth=2, label="Falling")
    ax4a.plot([VTL, VTH], [Vsat_pos, Vsat_pos], "b-", linewidth=2)
    ax4a.plot([0, VTL], [Vsat_neg, Vsat_neg], "r-", linewidth=2)
    # Arrows to show direction
    ax4a.annotate("", xy=(VTL, Vsat_pos - 1), xytext=(VTL, Vsat_neg + 1),
                  arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax4a.annotate("", xy=(VTH, Vsat_neg + 1), xytext=(VTH, Vsat_pos - 1),
                  arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax4a.axvline(x=VTL, color="gray", linestyle=":", alpha=0.4)
    ax4a.axvline(x=VTH, color="gray", linestyle=":", alpha=0.4)
    ax4a.annotate(f"V_TL = {VTL} V", xy=(VTL, -16), fontsize=10, ha="center", color="blue")
    ax4a.annotate(f"V_TH = {VTH} V", xy=(VTH, -16), fontsize=10, ha="center", color="red")
    ax4a.set_xlabel("V_in (V)")
    ax4a.set_ylabel("V_out (V)")
    ax4a.set_title("Hysteresis Transfer Characteristic")
    ax4a.set_xlim(-0.5, 5.5)
    ax4a.legend()
    ax4a.grid(True, alpha=0.3)

    # Time domain
    ax4b.plot(t_schmitt * 1e3, Vin_schmitt, "b-", linewidth=1.5, label="V_in")
    ax4b.axhline(y=VTH, color="red", linestyle="--", alpha=0.5, label=f"V_TH = {VTH} V")
    ax4b.axhline(y=VTL, color="blue", linestyle="--", alpha=0.5, label=f"V_TL = {VTL} V")
    ax4b_twin = ax4b.twinx()
    ax4b_twin.plot(t_schmitt * 1e3, Vout_schmitt, "r-", linewidth=1.5, alpha=0.7, label="V_out")
    ax4b.set_xlabel("Time (ms)")
    ax4b.set_ylabel("V_in (V)", color="blue")
    ax4b_twin.set_ylabel("V_out (V)", color="red")
    ax4b.set_title("Schmitt Trigger Time Response")
    lines1, labels1 = ax4b.get_legend_handles_labels()
    lines2, labels2 = ax4b_twin.get_legend_handles_labels()
    ax4b.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax4b.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4
    return


# --- 13.7.2 Slew Rate Limiting ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.7.2 Slew Rate — Limited Square Wave Response

        An op-amp with a slew rate of 13 V/μs cannot track the instantaneous edges
        of a square wave. The output ramps linearly at the slew rate limit, taking
        1.54 μs to traverse the 20 Vpp swing (±10 V). At higher frequencies the
        output becomes a triangular wave rather than a square wave.
        """
    )
    return


@app.cell
def _(np, plt):
    SR = 13e6  # V/s (13 V/μs)
    Vpp_sr = 20  # V peak-to-peak (±10 V)
    f_sr = 200e3  # 200 kHz square wave

    t_sr = np.linspace(0, 10e-6, 5000)  # 10 μs window (2 cycles)
    T_sr = 1 / f_sr

    # Ideal square wave
    v_ideal = Vpp_sr / 2 * np.sign(np.sin(2 * np.pi * f_sr * t_sr))

    # Slew-rate limited output
    v_slew = np.zeros_like(t_sr)
    v_slew[0] = v_ideal[0]
    dt_sr = t_sr[1] - t_sr[0]
    for _i in range(1, len(t_sr)):
        desired = v_ideal[_i]
        delta = desired - v_slew[_i - 1]
        max_delta = SR * dt_sr
        if abs(delta) > max_delta:
            v_slew[_i] = v_slew[_i - 1] + np.sign(delta) * max_delta
        else:
            v_slew[_i] = desired

    t_rise_sr = Vpp_sr / SR

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(t_sr * 1e6, v_ideal, "b--", linewidth=1, alpha=0.5, label="Ideal square wave")
    ax5.plot(t_sr * 1e6, v_slew, "r-", linewidth=2, label=f"Slew-limited (SR = 13 V/μs)")
    ax5.annotate(f"Rise time = {t_rise_sr*1e6:.2f} μs",
                 xy=(T_sr / 2 * 1e6 + t_rise_sr / 2 * 1e6, 0),
                 xytext=(T_sr / 2 * 1e6 + 2, -4),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
    ax5.set_xlabel("Time (μs)")
    ax5.set_ylabel("Voltage (V)")
    ax5.set_title(f"Slew Rate Limiting: 200 kHz Square Wave, SR = 13 V/μs, Rise Time = {t_rise_sr*1e6:.2f} μs")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5
    return


# --- 13.2.5 Log Amplifier Transfer Curve ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.2.5 Log Amplifier — Logarithmic Transfer Curve

        A log amplifier with a BJT feedback element (I_S = 10⁻¹⁴ A, R_in = 10 kΩ)
        produces V_out = -(kT/q) × ln(V_in / (I_S × R_in)). The output changes
        by −59.2 mV per decade of input, compressing a 3-decade input range
        into less than 200 mV of output swing.
        """
    )
    return


@app.cell
def _(np, plt):
    kT_q = 0.025693  # V at 25°C (kT/q = 25.69 mV at 298.15 K)
    I_S = 1e-14  # A
    R_in_log = 10e3  # Ω

    V_in_log = np.logspace(-2, 1, 500)  # 10 mV to 10 V
    I_in = V_in_log / R_in_log
    V_out_log = -kT_q * np.log(I_in / I_S)

    # Marked points from the example
    V_points = [0.01, 0.1, 1.0, 10.0]
    V_out_points = [-kT_q * np.log(v / R_in_log / I_S) for v in V_points]

    fig_log, ax_log = plt.subplots(figsize=(10, 5))
    ax_log.semilogx(V_in_log, V_out_log * 1000, "b-", linewidth=2,
                     label="V_out = -(kT/q) ln(V_in/(I_S R))")
    for v, vo in zip(V_points, V_out_points):
        ax_log.plot(v, vo * 1000, "ro", markersize=8, zorder=5)
        ax_log.annotate(f"{vo*1000:.0f} mV", xy=(v, vo*1000),
                        xytext=(v * 1.5, vo*1000 + 8), fontsize=9, color="red")
    ax_log.set_xlabel("V_in (V)")
    ax_log.set_ylabel("V_out (mV)")
    ax_log.set_title("Log Amplifier Transfer Curve (I_S = 10⁻¹⁴ A, R_in = 10 kΩ, T = 25°C)")
    ax_log.annotate("−59.2 mV/decade", xy=(0.1, -532), fontsize=11, color="green",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    ax_log.legend()
    ax_log.grid(True, alpha=0.3, which="both")
    fig_log.tight_layout()
    fig_log
    return


# --- 13.5.4 MFB Bandpass Filter Frequency Response ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.5.4 Multiple Feedback (MFB) Bandpass Filter

        An MFB bandpass filter with f₀ = 1 kHz, Q = 10, and midband gain |A₀| = 5
        (using C = 10 nF, R₁ = 15.9 kΩ, R₂ = 318 kΩ, R₃ = 7.96 kΩ) provides a
        very narrow bandpass response with BW = f₀/Q = 100 Hz centered at 1 kHz.
        """
    )
    return


@app.cell
def _(np, plt):
    # MFB bandpass parameters from Example 13.5.4 (scaled to f0=1kHz, Q=10, |A0|=5)
    f0_mfb = 1000.0   # Hz center frequency
    Q_mfb = 10.0      # quality factor
    A0_mfb = 5.0      # midband gain magnitude

    # MFB bandpass transfer function: H(s) = -(A0 * ω0/Q * s) / (s² + (ω0/Q)*s + ω0²)
    # |H(jω)| = A0 * (ω0/Q * ω) / sqrt((ω0² - ω²)² + (ω0/Q * ω)²)
    f_mfb = np.logspace(1.5, 4.5, 2000)   # 31.6 Hz to 31.6 kHz
    omega0_mfb = 2 * np.pi * f0_mfb
    omega_mfb = 2 * np.pi * f_mfb

    # Normalized frequency u = ω/ω0
    u_mfb = f_mfb / f0_mfb
    # Bandpass magnitude: |H(ju)| = A0 / sqrt(1 + Q²*(u - 1/u)²)
    H_mag_mfb = A0_mfb / np.sqrt(1 + Q_mfb**2 * (u_mfb - 1.0 / u_mfb)**2)
    H_dB_mfb = 20 * np.log10(H_mag_mfb)

    BW_mfb = f0_mfb / Q_mfb   # 100 Hz
    f_low_mfb = f0_mfb * (np.sqrt(1 + 1 / (4 * Q_mfb**2)) - 1 / (2 * Q_mfb))   # lower -3 dB
    f_high_mfb = f0_mfb * (np.sqrt(1 + 1 / (4 * Q_mfb**2)) + 1 / (2 * Q_mfb))  # upper -3 dB
    A0_dB_mfb = 20 * np.log10(A0_mfb)

    fig_mfb, ax_mfb = plt.subplots(figsize=(10, 5))

    ax_mfb.semilogx(f_mfb, H_dB_mfb, "b-", linewidth=2, label=f"MFB Bandpass (Q = {Q_mfb:.0f})")

    # Mark center frequency and peak gain
    ax_mfb.axvline(x=f0_mfb, color="red", linestyle="--", alpha=0.7, label=f"f₀ = {f0_mfb:.0f} Hz")
    ax_mfb.plot(f0_mfb, A0_dB_mfb, "ro", markersize=8, zorder=5)
    ax_mfb.annotate(f"f₀ = {f0_mfb:.0f} Hz\n|A₀| = {A0_dB_mfb:.1f} dB",
                    xy=(f0_mfb, A0_dB_mfb), xytext=(f0_mfb * 2.5, A0_dB_mfb - 4),
                    fontsize=10, color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))

    # Mark -3 dB bandwidth
    A3dB_mfb = A0_dB_mfb - 3
    ax_mfb.axhline(y=A3dB_mfb, color="green", linestyle=":", alpha=0.7,
                   label=f"−3 dB level ({A3dB_mfb:.1f} dB)")
    ax_mfb.axvline(x=f_low_mfb, color="green", linestyle=":", alpha=0.5)
    ax_mfb.axvline(x=f_high_mfb, color="green", linestyle=":", alpha=0.5)
    ax_mfb.annotate("", xy=(f_high_mfb, A3dB_mfb - 1.5), xytext=(f_low_mfb, A3dB_mfb - 1.5),
                    arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
    ax_mfb.annotate(f"BW = {BW_mfb:.0f} Hz",
                    xy=((f_low_mfb + f_high_mfb) / 2, A3dB_mfb - 2.5),
                    ha="center", fontsize=10, color="green")

    # Annotate Q factor
    ax_mfb.annotate(f"Q = f₀/BW = {Q_mfb:.0f}",
                    xy=(150, A0_dB_mfb - 8), fontsize=11,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    ax_mfb.set_xlabel("Frequency (Hz)")
    ax_mfb.set_ylabel("Magnitude (dB)")
    ax_mfb.set_title(f"MFB Bandpass Filter Frequency Response (f₀ = {f0_mfb:.0f} Hz, Q = {Q_mfb:.0f}, |A₀| = {A0_mfb:.0f})")
    ax_mfb.set_ylim(A0_dB_mfb - 35, A0_dB_mfb + 5)
    ax_mfb.legend()
    ax_mfb.grid(True, alpha=0.3, which="both")

    fig_mfb.tight_layout()
    fig_mfb
    return


# --- 13.5.5 Notch Filter Frequency Response ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.5.5 Active Twin-T Notch Filter — 60 Hz Rejection

        An active Twin-T notch filter with f₀ = 60 Hz and positive feedback
        k = 0.96 achieves Q = 6.25 and BW = 9.6 Hz. The deep notch at 60 Hz
        rejects power-line hum while preserving adjacent frequencies.
        """
    )
    return


@app.cell
def _(np, plt):
    f0_notch = 60  # Hz
    k_notch = 0.96
    Q_notch = 1 / (4 * (1 - k_notch))  # 6.25

    f_notch = np.linspace(1, 200, 5000)
    # Twin-T notch transfer function: H(s) = (s² + ω₀²) / (s² + (ω₀/Q)s + ω₀²)
    s_n = 1j * f_notch / f0_notch  # normalized
    H_notch = (s_n**2 + 1) / (s_n**2 + s_n / Q_notch + 1)
    H_mag_notch = 20 * np.log10(np.abs(H_notch))

    fig_notch, ax_notch = plt.subplots(figsize=(10, 5))
    ax_notch.plot(f_notch, H_mag_notch, "b-", linewidth=2)
    ax_notch.axvline(x=60, color="red", linestyle="--", alpha=0.6, label="f₀ = 60 Hz")
    BW_notch = f0_notch / Q_notch
    ax_notch.axvspan(60 - BW_notch/2, 60 + BW_notch/2, alpha=0.15, color="orange",
                      label=f"BW = {BW_notch:.1f} Hz (Q = {Q_notch:.2f})")
    ax_notch.axhline(y=-3, color="gray", linestyle=":", alpha=0.5, label="−3 dB")
    ax_notch.set_xlabel("Frequency (Hz)")
    ax_notch.set_ylabel("Magnitude (dB)")
    ax_notch.set_title("Active Twin-T Notch Filter (f₀ = 60 Hz, k = 0.96)")
    ax_notch.set_ylim(-50, 5)
    ax_notch.legend()
    ax_notch.grid(True, alpha=0.3)
    fig_notch.tight_layout()
    fig_notch
    return


# --- 13.6.4 Relaxation Oscillator Waveforms ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 13.6.4 Relaxation Oscillator — Capacitor and Output Waveforms

        An astable multivibrator with R = 22 kΩ, C = 10 nF, and β = 0.5 (R₁ = R₂)
        oscillates at f = 1/(2RC ln(3)) ≈ 2.07 kHz. The capacitor charges
        exponentially between ±6 V thresholds while the output is a ±12 V square wave.
        """
    )
    return


@app.cell
def _(np, plt):
    R_relax = 22e3  # Ω
    C_relax = 10e-9  # F
    beta_relax = 0.5
    V_sat_relax = 12  # V
    tau_relax = R_relax * C_relax  # 220 μs

    V_TH_relax = beta_relax * V_sat_relax   # +6 V
    V_TL_relax = -beta_relax * V_sat_relax  # -6 V
    f_relax = 1 / (2 * tau_relax * np.log((1 + beta_relax) / (1 - beta_relax)))

    # Simulate 4 cycles
    dt_relax = 0.5e-6
    t_end = 4 / f_relax
    t_relax = np.arange(0, t_end, dt_relax)
    v_cap = np.zeros_like(t_relax)
    v_out_relax = np.zeros_like(t_relax)

    v_cap[0] = V_TL_relax  # start at lower threshold
    v_out_relax[0] = V_sat_relax  # output high, cap charging up

    for _i in range(1, len(t_relax)):
        target = v_out_relax[_i-1]  # capacitor charges toward output
        v_cap[_i] = target + (v_cap[_i-1] - target) * np.exp(-dt_relax / tau_relax)
        if v_out_relax[_i-1] > 0 and v_cap[_i] >= V_TH_relax:
            v_out_relax[_i] = -V_sat_relax
        elif v_out_relax[_i-1] < 0 and v_cap[_i] <= V_TL_relax:
            v_out_relax[_i] = V_sat_relax
        else:
            v_out_relax[_i] = v_out_relax[_i-1]

    fig_relax, (ax_r1, ax_r2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_r1.plot(t_relax * 1e3, v_out_relax, "r-", linewidth=1.5, label="V_out (square wave)")
    ax_r1.set_ylabel("V_out (V)")
    ax_r1.set_title(f"Relaxation Oscillator Output (f = {f_relax:.0f} Hz)")
    ax_r1.set_ylim(-15, 15)
    ax_r1.legend()
    ax_r1.grid(True, alpha=0.3)

    ax_r2.plot(t_relax * 1e3, v_cap, "b-", linewidth=2, label="V_cap (exponential)")
    ax_r2.axhline(y=V_TH_relax, color="red", linestyle="--", alpha=0.6, label=f"V_TH = +{V_TH_relax:.0f} V")
    ax_r2.axhline(y=V_TL_relax, color="blue", linestyle="--", alpha=0.6, label=f"V_TL = {V_TL_relax:.0f} V")
    ax_r2.set_xlabel("Time (ms)")
    ax_r2.set_ylabel("V_cap (V)")
    ax_r2.set_title(f"Capacitor Voltage (R = 22 kΩ, C = 10 nF, β = {beta_relax})")
    ax_r2.legend()
    ax_r2.grid(True, alpha=0.3)

    fig_relax.tight_layout()
    fig_relax
    return


if __name__ == "__main__":
    app.run()
