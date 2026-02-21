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
    index m = 0.8. The top three subplots show the carrier, message, and
    resulting AM waveform with its envelope. The bottom subplot shows the
    frequency spectrum with the carrier peak at 1 MHz and symmetric
    sidebands at f_c +/- f_m.
    """)
    return


@app.cell
def _(np, plt):
    # Parameters
    fc = 1e6       # carrier frequency: 1 MHz
    fm = 1e3       # message frequency: 1 kHz
    m_idx = 0.8    # modulation index
    Ac = 1.0       # carrier amplitude

    # Time vector: show 2 cycles of the message signal (2 ms)
    # Sample at 10x carrier frequency for smooth rendering, but limit points
    # We only need to resolve the envelope, so sample at ~50 kHz for display
    # and compute the AM math analytically
    t_duration = 2e-3  # 2 ms = 2 message cycles
    fs_display = 200e3  # 200 kHz display sample rate (fast enough for envelope)
    t = np.linspace(0, t_duration, int(fs_display * t_duration), endpoint=False)

    # Signals
    carrier = Ac * np.cos(2 * np.pi * fc * t)
    message = m_idx * Ac * np.cos(2 * np.pi * fm * t)
    am_signal = (Ac + message) * np.cos(2 * np.pi * fc * t)
    envelope_upper = Ac + message
    envelope_lower = -(Ac + message)

    # For the carrier subplot, show a zoomed-in portion (first 10 us)
    t_zoom = np.linspace(0, 10e-6, 2000, endpoint=False)
    carrier_zoom = Ac * np.cos(2 * np.pi * fc * t_zoom)

    # Frequency spectrum via FFT
    N_fft = 2**18  # enough resolution
    fs_fft = 10e6  # 10 MHz sample rate for FFT
    t_fft = np.arange(N_fft) / fs_fft
    am_fft_signal = (Ac + m_idx * Ac * np.cos(2 * np.pi * fm * t_fft)) * np.cos(2 * np.pi * fc * t_fft)
    spectrum = np.abs(np.fft.rfft(am_fft_signal)) / N_fft * 2
    freqs = np.fft.rfftfreq(N_fft, 1 / fs_fft)

    fig1, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Carrier (zoomed to show individual cycles)
    axes[0].plot(t_zoom * 1e6, carrier_zoom, "b-", linewidth=0.8)
    axes[0].set_ylabel("Amplitude (V)")
    axes[0].set_title("Carrier Signal (f_c = 1 MHz)")
    axes[0].set_xlabel("Time (us)")
    axes[0].set_xlim(0, 10)
    axes[0].grid(True, alpha=0.3)

    # Message signal
    axes[1].plot(t * 1e3, message, "g-", linewidth=2)
    axes[1].set_ylabel("Amplitude (V)")
    axes[1].set_title(f"Message Signal (f_m = 1 kHz, m = {m_idx})")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_xlim(0, t_duration * 1e3)
    axes[1].grid(True, alpha=0.3)

    # AM signal with envelope
    axes[2].plot(t * 1e3, am_signal, "b-", linewidth=0.3, alpha=0.6, label="AM signal")
    axes[2].plot(t * 1e3, envelope_upper, "r--", linewidth=2, label="Envelope")
    axes[2].plot(t * 1e3, envelope_lower, "r--", linewidth=2)
    axes[2].set_ylabel("Amplitude (V)")
    axes[2].set_title("AM Waveform with Envelope")
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_xlim(0, t_duration * 1e3)
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Frequency spectrum
    # Focus on region around carrier
    mask = (freqs >= 0.990e6) & (freqs <= 1.010e6)
    axes[3].plot(freqs[mask] / 1e6, spectrum[mask], "b-", linewidth=1.5)
    axes[3].set_ylabel("Magnitude")
    axes[3].set_xlabel("Frequency (MHz)")
    axes[3].set_title("AM Frequency Spectrum")
    axes[3].grid(True, alpha=0.3)

    # Annotate carrier and sidebands
    carrier_idx = np.argmin(np.abs(freqs - fc))
    lsb_idx = np.argmin(np.abs(freqs - (fc - fm)))
    usb_idx = np.argmin(np.abs(freqs - (fc + fm)))
    axes[3].annotate(f"Carrier\n{fc/1e6:.3f} MHz",
                     xy=(fc / 1e6, spectrum[carrier_idx]),
                     xytext=(fc / 1e6 + 0.003, spectrum[carrier_idx] * 0.9),
                     fontsize=9, color="blue",
                     arrowprops=dict(arrowstyle="->", color="blue"))
    axes[3].annotate(f"LSB\n{(fc-fm)/1e6:.3f} MHz",
                     xy=((fc - fm) / 1e6, spectrum[lsb_idx]),
                     xytext=((fc - fm) / 1e6 - 0.005, spectrum[lsb_idx] * 1.3),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
    axes[3].annotate(f"USB\n{(fc+fm)/1e6:.3f} MHz",
                     xy=((fc + fm) / 1e6, spectrum[usb_idx]),
                     xytext=((fc + fm) / 1e6 + 0.002, spectrum[usb_idx] * 1.3),
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
    axis, QPSK uses 4 points at 45/135/225/315 degrees, 8-PSK places 8
    points on the unit circle, and 16-QAM arranges 16 points in a 4x4 grid
    with varying amplitude and phase.
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
    # Draw circles at the two amplitude levels
    for r in [np.sqrt(2), np.sqrt(10), np.sqrt(18)]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor="gray",
                                linestyle=":", linewidth=0.5, alpha=0.4))
    ax.set_title("16-QAM (4 bits/symbol)", fontsize=12)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
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
    SNR for three bandwidths (1 MHz, 10 MHz, 100 MHz). The second subplot
    shows spectral efficiency C/B vs SNR, which is independent of bandwidth
    and approaches the Shannon limit at Eb/N0 = -1.59 dB.
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
        ax_cap.plot(snr_db, capacity / 1e6, linewidth=2, color=color, label=f"B = {_label}")

    ax_cap.set_xlabel("SNR (dB)")
    ax_cap.set_ylabel("Channel Capacity (Mbps)")
    ax_cap.set_title("Shannon Channel Capacity: C = B * log2(1 + SNR)")
    ax_cap.legend(fontsize=10)
    ax_cap.grid(True, alpha=0.3)
    ax_cap.set_xlim(-10, 40)

    # Annotate a specific point
    snr_example = 20  # dB
    snr_lin_ex = 10 ** (snr_example / 10)
    cap_10mhz = 10e6 * np.log2(1 + snr_lin_ex)
    ax_cap.plot(snr_example, cap_10mhz / 1e6, "ko", markersize=8, zorder=5)
    ax_cap.annotate(f"SNR = 20 dB, B = 10 MHz\nC = {cap_10mhz/1e6:.1f} Mbps",
                    xy=(snr_example, cap_10mhz / 1e6),
                    xytext=(snr_example + 5, cap_10mhz / 1e6 + 50),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # --- Subplot 2: Spectral Efficiency vs SNR ---
    spectral_eff = np.log2(1 + snr_linear)  # bits/s/Hz

    ax_eff.plot(snr_db, spectral_eff, "b-", linewidth=2, label="C/B = log2(1 + SNR)")

    # Shannon limit: minimum Eb/N0 = -1.59 dB = ln(2) = 0.693 linear
    # At the Shannon limit, spectral efficiency approaches 0
    shannon_limit_db = -1.59
    ax_eff.axvline(x=shannon_limit_db, color="red", linestyle="--", linewidth=1.5,
                   label=f"Shannon limit (Eb/N0 = {shannon_limit_db} dB)")
    ax_eff.annotate(f"Shannon Limit\nEb/N0 = -1.59 dB",
                    xy=(shannon_limit_db, 0.5),
                    xytext=(shannon_limit_db + 8, 1.0),
                    fontsize=10, color="red",
                    arrowprops=dict(arrowstyle="->", color="red"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Mark common modulation scheme efficiencies
    mod_schemes = [
        ("BPSK", 1.0),
        ("QPSK", 2.0),
        ("16-QAM", 4.0),
        ("64-QAM", 6.0),
        ("256-QAM", 8.0),
    ]
    for name, eff in mod_schemes:
        snr_required = 10 * np.log10(2 ** eff - 1)
        ax_eff.plot(snr_required, eff, "s", markersize=8, color="darkgreen", zorder=5)
        ax_eff.annotate(name, (snr_required, eff), textcoords="offset points",
                        xytext=(8, 3), fontsize=9, color="darkgreen")

    ax_eff.set_xlabel("SNR (dB)")
    ax_eff.set_ylabel("Spectral Efficiency (bits/s/Hz)")
    ax_eff.set_title("Spectral Efficiency vs SNR with Shannon Bound")
    ax_eff.legend(fontsize=10, loc="upper left")
    ax_eff.grid(True, alpha=0.3)
    ax_eff.set_xlim(-10, 40)
    ax_eff.set_ylim(0, 14)

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
