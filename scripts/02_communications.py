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
    # Chapter 2: Communications Engineering — Example Visualizations

    Interactive graphs for selected example problems from Chapter 2,
    covering AM modulation, digital modulation constellation diagrams,
    and Shannon channel capacity.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2.1.1 Amplitude Modulation (AM)

    A 1 MHz carrier is modulated by a 1 kHz message signal with modulation
    index m = 0.75 (as in Example 2.1.1). The top three subplots show the
    carrier, message, and resulting AM waveform with its envelope; the AM
    waveform panel draws the carrier at an illustrative 25 kHz so that its
    cycles are visible over the 2 ms window. The bottom subplot shows the
    frequency spectrum: the carrier line at 1 MHz (A_c = 1.0) and the two
    sidebands at f_c +/- f_m, each m * A_c / 2 = 0.375, symmetric about f_c.
    The FFT length puts all three tones exactly on bin centres, so the line
    heights are exact.
    """)
    return


@app.cell
def _(np, plt):
    # Parameters match Example 2.1.1 (m = 0.75); 1 V carrier, 1 kHz tone.
    fc = 1e6       # carrier frequency: 1 MHz
    fm = 1e3       # message frequency: 1 kHz
    m_idx = 0.75   # modulation index (Example 2.1.1)
    Ac = 1.0       # carrier amplitude

    t_duration = 2e-3
    t = np.linspace(0, t_duration, 4000, endpoint=False)
    message = m_idx * Ac * np.cos(2 * np.pi * fm * t)

    t_zoom = np.linspace(0, 10e-6, 2000, endpoint=False)
    carrier_zoom = Ac * np.cos(2 * np.pi * fc * t_zoom)

    # AM waveform panel: a 1 MHz carrier would put 2,000 cycles in the 2 ms view
    # and fill the panel solid, so this panel alone uses an illustrative 25 kHz
    # carrier, sampled at 2 MHz (80 samples per carrier cycle).
    fc_show = 25e3
    t_am = np.arange(0, t_duration, 1 / 2e6)
    env_am = Ac * (1 + m_idx * np.cos(2 * np.pi * fm * t_am))
    am_show = env_am * np.cos(2 * np.pi * fc_show * t_am)

    # Spectrum: fs = 10 MHz and N = 10^6 give 10 Hz bins, so f_c and f_c +/- f_m
    # fall exactly on bin centres (no scalloping, no leakage, no window needed).
    N_fft = 1_000_000
    fs_fft = 10e6
    t_fft = np.arange(N_fft) / fs_fft
    am_fft_signal = (Ac + m_idx * Ac * np.cos(2 * np.pi * fm * t_fft)) * np.cos(2 * np.pi * fc * t_fft)
    spectrum = np.abs(np.fft.rfft(am_fft_signal)) / N_fft * 2
    freqs = np.fft.rfftfreq(N_fft, 1 / fs_fft)

    fig1, axes = plt.subplots(4, 1, figsize=(12, 10))

    axes[0].plot(t_zoom * 1e6, carrier_zoom, "b-", linewidth=0.8)
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].set_title("Carrier Signal (f_c = 1 MHz)")
    axes[0].set_xlabel("Time (µs)")
    axes[0].set_xlim(0, 10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t * 1e3, message, "g-", linewidth=2)
    axes[1].set_ylabel("Amplitude (V)")
    axes[1].set_title(f"Message Signal (f_m = 1 kHz, m = {m_idx})")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_xlim(0, t_duration * 1e3)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_am * 1e3, am_show, "b-", linewidth=0.7, label="AM signal")
    axes[2].plot(t_am * 1e3, env_am, "r--", linewidth=2, label="Envelope")
    axes[2].plot(t_am * 1e3, -env_am, "r--", linewidth=2)
    axes[2].set_ylabel("Amplitude (V)")
    axes[2].set_title("AM Waveform with Envelope (carrier drawn at an illustrative "
                      "25 kHz so its cycles are visible; m = 0.75)")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_xlim(0, t_duration * 1e3)
    axes[2].set_ylim(-2.0, 2.0)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    mask = (freqs >= 0.995e6) & (freqs <= 1.005e6)
    axes[3].plot(freqs[mask] / 1e6, spectrum[mask], "b-", linewidth=1.5)
    axes[3].set_ylabel("Magnitude (V)")
    axes[3].set_xlabel("Frequency (MHz)")
    axes[3].set_title("AM Frequency Spectrum (carrier A_c = 1.0, each sideband "
                      "m·A_c/2 = 0.375, symmetric about f_c)")
    axes[3].set_ylim(0, 1.25)
    axes[3].grid(True, alpha=0.3)

    carrier_idx = np.argmin(np.abs(freqs - fc))
    lsb_idx = np.argmin(np.abs(freqs - (fc - fm)))
    usb_idx = np.argmin(np.abs(freqs - (fc + fm)))
    axes[3].annotate(f"Carrier {fc/1e6:.3f} MHz\n{spectrum[carrier_idx]:.3f} V",
                     xy=(fc / 1e6, spectrum[carrier_idx]),
                     xytext=(fc / 1e6 + 0.0008, 1.02),
                     fontsize=9, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue"))
    axes[3].annotate(f"LSB {(fc-fm)/1e6:.3f} MHz\n{spectrum[lsb_idx]:.3f} V",
                     xy=((fc - fm) / 1e6, spectrum[lsb_idx]),
                     xytext=((fc - fm) / 1e6 - 0.0032, 0.62),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
    axes[3].annotate(f"USB {(fc+fm)/1e6:.3f} MHz\n{spectrum[usb_idx]:.3f} V",
                     xy=((fc + fm) / 1e6, spectrum[usb_idx]),
                     xytext=((fc + fm) / 1e6 + 0.0008, 0.62),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

    fig1.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2.3 Digital Modulation — Constellation Diagrams

    Constellation diagrams show the mapping of digital symbols to points in
    the I/Q (In-phase / Quadrature) plane. BPSK uses 2 points on the real
    axis, QPSK uses 4 points at 45/135/225/315 degrees (the π/4-rotated
    convention an I/Q modulator driven by ±1 on each arm produces — the same
    modulation as the 0/90/180/270 set, and not to be confused with offset
    QPSK, which staggers the Q arm in time), 8-PSK places 8
    points on the unit circle, and 16-QAM arranges 16 points in a 4x4 grid
    with varying amplitude and phase on three amplitude rings, Gray coded
    (the first two bits follow I and the last two follow Q).
    """)
    return


@app.cell
def _(np, plt):
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 10))

    marker_kwargs = dict(marker="o", markersize=10, linestyle="none", color="blue", zorder=5)
    circle_kwargs = dict(fill=False, edgecolor="gray", linestyle="--", linewidth=1, alpha=0.5)

    # --- BPSK ---
    ax = axes2[0, 0]
    bpsk_i = np.array([-1, 1])
    bpsk_q = np.array([0, 0])
    ax.plot(bpsk_i, bpsk_q, **marker_kwargs)
    ax.add_patch(plt.Circle((0, 0), 1, **circle_kwargs))
    for i, _label in enumerate(["0", "1"]):
        ax.annotate(_label, (bpsk_i[i], bpsk_q[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("BPSK (1 bit/symbol)", fontsize=12)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # --- QPSK ---
    ax = axes2[0, 1]
    qpsk_angles = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
    qpsk_i = np.cos(qpsk_angles)
    qpsk_q = np.sin(qpsk_angles)
    ax.plot(qpsk_i, qpsk_q, **marker_kwargs)
    ax.add_patch(plt.Circle((0, 0), 1, **circle_kwargs))
    qpsk_labels = ["00", "01", "11", "10"]
    for i, _label in enumerate(qpsk_labels):
        offset_x = 15 if qpsk_i[i] > 0 else -15
        offset_y = 15 if qpsk_q[i] > 0 else -15
        ax.annotate(_label, (qpsk_i[i], qpsk_q[i]), textcoords="offset points",
                    xytext=(offset_x, offset_y), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("QPSK (2 bits/symbol)", fontsize=12)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # --- 8-PSK ---
    ax = axes2[1, 0]
    psk8_angles = np.arange(8) * 2 * np.pi / 8
    psk8_i = np.cos(psk8_angles)
    psk8_q = np.sin(psk8_angles)
    ax.plot(psk8_i, psk8_q, **marker_kwargs)
    ax.add_patch(plt.Circle((0, 0), 1, **circle_kwargs))
    psk8_labels = ["000", "001", "011", "010", "110", "111", "101", "100"]
    for i, _label in enumerate(psk8_labels):
        offset_x = 20 * np.cos(psk8_angles[i])
        offset_y = 20 * np.sin(psk8_angles[i])
        ax.annotate(_label, (psk8_i[i], psk8_q[i]), textcoords="offset points",
                    xytext=(offset_x, offset_y), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("8-PSK (3 bits/symbol)", fontsize=12)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # --- 16-QAM ---
    ax = axes2[1, 1]
    qam_vals = np.array([-3, -1, 1, 3])
    qam_i_pts = []
    qam_q_pts = []
    for qi in qam_vals:
        for qq in qam_vals:
            qam_i_pts.append(qi)
            qam_q_pts.append(qq)
    qam_i_pts = np.array(qam_i_pts)
    qam_q_pts = np.array(qam_q_pts)
    ax.plot(qam_i_pts, qam_q_pts, **marker_kwargs)
    # Gray labels: 2-bit Gray sequence 00, 01, 11, 10 for -3, -1, +1, +3 on
    # each axis (I bits first, then Q bits), so adjacent points differ in 1 bit.
    qam_gray = {-3: "00", -1: "01", 1: "11", 3: "10"}
    for _qi, _qq in zip(qam_i_pts, qam_q_pts):
        ax.annotate(qam_gray[_qi] + qam_gray[_qq], (_qi, _qq),
                    textcoords="offset points", xytext=(0, 11), ha="center",
                    fontsize=8, fontweight="bold")
    # Draw circles at the three amplitude levels: sqrt(2), sqrt(10), sqrt(18)
    for r in [np.sqrt(2), np.sqrt(10), np.sqrt(18)]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor="gray",
                                linestyle="--", linewidth=1, alpha=0.5))
    ax.set_title("16-QAM (4 bits/symbol, Gray coded)", fontsize=12)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-4.6, 4.6)
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    fig2.suptitle("Digital Modulation Constellation Diagrams", fontsize=14, y=1.01)
    fig2.tight_layout()
    fig2
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2.6 Shannon Channel Capacity

    The Shannon-Hartley theorem gives the maximum data rate for a noisy
    channel: C = B * log2(1 + SNR). The first subplot shows capacity vs
    SNR for three bandwidths (1 MHz, 10 MHz, 100 MHz) on a log axis. The
    second subplot shows spectral efficiency C/B vs SNR, which is independent
    of bandwidth; the squares mark the Shannon-minimum SNR (2^eta - 1) for
    eta = 1...8 bits/s/Hz and the triangles mark where uncoded BPSK...256-QAM
    need to operate for BER = 1e-5, showing the gap that coding must close.
    The -1.59 dB limit bounds Eb/N0, not SNR, so it is stated as a note.
    """)
    return


@app.cell
def _(np, plt):
    snr_db = np.linspace(-10, 40, 500)
    snr_linear = 10 ** (snr_db / 10)

    bandwidths = [1e6, 10e6, 100e6]
    bw_labels = ["1 MHz", "10 MHz", "100 MHz"]
    bw_colors = ["blue", "green", "red"]

    fig3, (ax_cap, ax_eff) = plt.subplots(2, 1, figsize=(12, 9))

    # --- Subplot 1: Channel Capacity vs SNR ---
    for bw, _label, color in zip(bandwidths, bw_labels, bw_colors):
        capacity = bw * np.log2(1 + snr_linear)
        ax_cap.semilogy(snr_db, capacity / 1e6, linewidth=2, color=color, label=f"B = {_label}")

    ax_cap.set_xlabel("SNR (dB)")
    ax_cap.set_ylabel("Channel Capacity (Mbps, log scale)")
    ax_cap.set_title("Shannon Channel Capacity: C = B * log2(1 + SNR)")
    ax_cap.legend(fontsize=10, loc="lower right")
    ax_cap.grid(True, which="both", alpha=0.3)
    ax_cap.set_xlim(-10, 40)
    ax_cap.set_ylim(0.1, 2000)

    # Annotate a specific point
    snr_example = 20  # dB
    snr_lin_ex = 10 ** (snr_example / 10)
    cap_10mhz = 10e6 * np.log2(1 + snr_lin_ex)
    ax_cap.plot(snr_example, cap_10mhz / 1e6, "ko", markersize=8, zorder=5)
    ax_cap.annotate(f"SNR = 20 dB, B = 10 MHz\nC = {cap_10mhz/1e6:.1f} Mbps",
                    xy=(snr_example, cap_10mhz / 1e6),
                    xytext=(snr_example + 3, 8),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- Subplot 2: Spectral Efficiency vs SNR ---
    spectral_eff = np.log2(1 + snr_linear)  # bits/s/Hz

    ax_eff.plot(snr_db, spectral_eff, "b-", linewidth=2, label="C/B = log2(1 + SNR)")

    # The -1.59 dB Shannon limit is a bound on Eb/N0, NOT on SNR, so it must not
    # be drawn as a vertical line on this SNR axis. The two are related by
    # SNR = (Eb/N0) x (C/B); as the spectral efficiency C/B approaches zero the
    # required Eb/N0 approaches ln(2) = -1.59 dB while the required SNR itself
    # approaches -infinity dB. Stated as a note rather than plotted.
    ax_eff.text(0.02, 0.97,
                "Shannon bound on energy per bit: Eb/N0 $\\to$ ln 2 = $-$1.59 dB\n"
                "as C/B $\\to$ 0. On this SNR axis that limit is at $-\\infty$ dB,\n"
                "since SNR = (Eb/N0) $\\times$ (C/B).",
                transform=ax_eff.transAxes, fontsize=9, color="darkred",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow", alpha=0.85))

    # Squares: the Shannon-minimum SNR = 2^eta - 1 for eta = 1...8 bits/s/Hz (on
    # the bound). Triangles: where uncoded BPSK...256-QAM actually need to operate
    # for BER = 1e-5 (SNR = Eb/N0 + 10 log10(eta), Eb/N0 from the Section 2.3.5
    # table), showing the gap that coding has to close.
    mod_schemes = [
        ("BPSK", 1.0, 9.6),
        ("QPSK", 2.0, 9.6),
        ("16-QAM", 4.0, 13.4),
        ("64-QAM", 6.0, 17.8),
        ("256-QAM", 8.0, 22.5),
    ]
    for k, (name, eff, ebn0_db) in enumerate(mod_schemes):
        snr_required = 10 * np.log10(2 ** eff - 1)
        snr_uncoded = ebn0_db + 10 * np.log10(eff)
        ax_eff.plot(snr_required, eff, "s", markersize=8, color="darkgreen", zorder=5,
                    label="Shannon-minimum SNR for η = 1…8" if k == 0 else None)
        ax_eff.plot(snr_uncoded, eff, "^", markersize=9, color="#CC4444", zorder=5,
                    label="Uncoded, BER = 10⁻⁵ (§2.3.5 table)" if k == 0 else None)
        ax_eff.plot([snr_required, snr_uncoded], [eff, eff], ":", color="gray",
                    linewidth=1, zorder=4)
        ax_eff.annotate(name, (snr_required, eff), textcoords="offset points",
                        xytext=(-8, 3), ha="right", fontsize=9, color="darkgreen")
        ax_eff.annotate(f"{snr_uncoded - snr_required:.1f} dB gap",
                        (snr_uncoded, eff), textcoords="offset points",
                        xytext=(8, -3), fontsize=8, color="#CC4444")

    ax_eff.set_xlabel("SNR (dB)")
    ax_eff.set_ylabel("Spectral Efficiency (bits/s/Hz)")
    ax_eff.set_title("Spectral Efficiency vs SNR with Shannon Bound")
    ax_eff.legend(fontsize=9, loc="lower right")
    ax_eff.grid(True, alpha=0.3)
    ax_eff.set_xlim(-10, 40)
    ax_eff.set_ylim(0, 14)

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
