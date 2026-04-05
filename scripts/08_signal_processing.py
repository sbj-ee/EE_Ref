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
        # Chapter 8: Signal Processing — Example Visualizations

        Interactive graphs for selected example problems from Chapter 8,
        covering Fourier analysis, digital filters, and spectral analysis.
        """
    )
    return


# --- 8.2.1 Fourier Series: Square wave harmonics ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.2.1 Fourier Series — Square Wave Harmonics

        A 5 V peak-to-peak square wave at 500 Hz is decomposed into its first three
        non-zero harmonics (1st, 3rd, 5th). Each harmonic is a sine wave with amplitude
        4A/(nπ). The composite of three harmonics begins to approximate the square wave.
        """
    )
    return


@app.cell
def _(np, plt):
    f_sq = 500  # Hz
    A_sq = 5  # V peak-to-peak, so amplitude = 2.5 V
    t_sq = np.linspace(0, 4e-3, 2000)

    h1 = (4 * 2.5 / (1 * np.pi)) * np.sin(2 * np.pi * 1 * f_sq * t_sq)
    h3 = (4 * 2.5 / (3 * np.pi)) * np.sin(2 * np.pi * 3 * f_sq * t_sq)
    h5 = (4 * 2.5 / (5 * np.pi)) * np.sin(2 * np.pi * 5 * f_sq * t_sq)
    composite = h1 + h3 + h5

    # Ideal square wave for reference
    sq_ideal = 2.5 * np.sign(np.sin(2 * np.pi * f_sq * t_sq))

    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(10, 8))

    # Individual harmonics
    ax1a.plot(t_sq * 1e3, h1, "b-", linewidth=1.5, label=f"1st (500 Hz): {4*2.5/np.pi:.3f} V", alpha=0.8)
    ax1a.plot(t_sq * 1e3, h3, "r-", linewidth=1.5, label=f"3rd (1500 Hz): {4*2.5/(3*np.pi):.3f} V", alpha=0.8)
    ax1a.plot(t_sq * 1e3, h5, "g-", linewidth=1.5, label=f"5th (2500 Hz): {4*2.5/(5*np.pi):.3f} V", alpha=0.8)
    ax1a.set_ylabel("Voltage (V)")
    ax1a.set_title("Individual Fourier Harmonics of a 500 Hz Square Wave")
    ax1a.legend()
    ax1a.grid(True, alpha=0.3)

    # Composite vs ideal
    ax1b.plot(t_sq * 1e3, sq_ideal, "k--", linewidth=1, label="Ideal square wave", alpha=0.5)
    ax1b.plot(t_sq * 1e3, composite, "b-", linewidth=2, label="Sum of 3 harmonics")
    ax1b.set_xlabel("Time (ms)")
    ax1b.set_ylabel("Voltage (V)")
    ax1b.set_title("Composite Waveform (1st + 3rd + 5th Harmonic)")
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1
    return


# --- 8.2.2 Fourier Transform: Rectangular pulse and sinc spectrum ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.2.2 Fourier Transform — Rectangular Pulse Spectrum

        A rectangular pulse x(t) = 1 for |t| ≤ 0.5 ms has a Fourier Transform
        that is a sinc function. The first null (zero crossing) occurs at f = 1 kHz,
        equal to 1/pulse_width.
        """
    )
    return


@app.cell
def _(np, plt):
    # Time domain: rectangular pulse
    t_pulse = np.linspace(-2e-3, 2e-3, 1000)
    pulse_width = 1e-3  # total width = 1 ms (±0.5 ms)
    x_pulse = np.where(np.abs(t_pulse) <= pulse_width / 2, 1.0, 0.0)

    # Frequency domain: sinc spectrum
    f_spec = np.linspace(-5000, 5000, 2000)
    # X(f) = pulse_width * sinc(f * pulse_width)
    X_spec = pulse_width * np.sinc(f_spec * pulse_width)

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 7))

    ax2a.plot(t_pulse * 1e3, x_pulse, "b-", linewidth=2)
    ax2a.fill_between(t_pulse * 1e3, x_pulse, alpha=0.2)
    ax2a.set_xlabel("Time (ms)")
    ax2a.set_ylabel("Amplitude")
    ax2a.set_title("Rectangular Pulse (width = 1 ms)")
    ax2a.set_ylim(-0.2, 1.3)
    ax2a.grid(True, alpha=0.3)

    ax2b.plot(f_spec / 1e3, np.abs(X_spec) * 1e3, "r-", linewidth=2)
    ax2b.axvline(x=1, color="green", linestyle="--", alpha=0.7, label="First null at 1 kHz")
    ax2b.axvline(x=-1, color="green", linestyle="--", alpha=0.7)
    ax2b.set_xlabel("Frequency (kHz)")
    ax2b.set_ylabel("|X(f)| (mV·s → mV at 1 kHz)")
    ax2b.set_title("Magnitude Spectrum |X(f)| = τ|sinc(fτ)|")
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2
    return


# --- 8.2.3 DFT: Discrete frequency bins ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.2.3 Discrete Fourier Transform — Frequency Bins

        A 512-point DFT at f_s = 16 kHz has frequency resolution Δf = 31.25 Hz.
        A 1500 Hz tone appears at bin k = 48 (= 1500 / 31.25). The stem plot shows
        the discrete nature of the DFT spectrum.
        """
    )
    return


@app.cell
def _(np, plt):
    N_dft = 512
    fs_dft = 16000  # Hz
    delta_f = fs_dft / N_dft  # 31.25 Hz
    f_tone = 1500  # Hz
    k_tone = int(f_tone / delta_f)  # bin 48

    # Generate signal: 1500 Hz tone
    n_dft = np.arange(N_dft)
    x_dft = np.sin(2 * np.pi * f_tone / fs_dft * n_dft)

    # Compute DFT magnitude (single-sided)
    X_dft = np.fft.fft(x_dft)
    X_mag = 2 * np.abs(X_dft[:N_dft // 2]) / N_dft
    f_bins = np.arange(N_dft // 2) * delta_f

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    markerline, stemlines, baseline = ax3.stem(f_bins / 1e3, X_mag, linefmt="b-", markerfmt="b.", basefmt="k-")
    stemlines.set_linewidth(0.5)
    markerline.set_markersize(2)

    # Highlight the tone bin
    ax3.plot(f_bins[k_tone] / 1e3, X_mag[k_tone], "ro", markersize=10, zorder=5,
             label=f"Bin {k_tone}: f = {f_bins[k_tone]:.0f} Hz")
    ax3.annotate(f"k = {k_tone}\nf = {f_bins[k_tone]:.0f} Hz",
                 xy=(f_bins[k_tone] / 1e3, X_mag[k_tone]),
                 xytext=(f_bins[k_tone] / 1e3 + 0.8, X_mag[k_tone] - 0.1),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")

    ax3.set_xlabel("Frequency (kHz)")
    ax3.set_ylabel("Magnitude")
    ax3.set_title(f"512-point DFT (f_s = 16 kHz, Δf = {delta_f} Hz)")
    ax3.set_xlim(0, fs_dft / 2 / 1e3)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3
    return


# --- 8.5.1 FIR Filter: Lowpass magnitude response ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.5.1 FIR Filter — Lowpass Magnitude Response

        An 11-tap FIR lowpass filter with a cutoff at f_s/4 (Hamming window).
        The magnitude response shows the passband, transition band, and stopband
        attenuation characteristic of windowed FIR designs.
        """
    )
    return


@app.cell
def _(np, plt):
    # Design an 11-tap FIR lowpass using windowed sinc method
    M_fir = 11
    fc_fir = 0.25  # normalized cutoff (f_s/4)
    n_fir = np.arange(M_fir)
    mid = (M_fir - 1) / 2

    # Sinc filter coefficients
    h_fir = np.where(n_fir == mid, 2 * fc_fir,
                     np.sin(2 * np.pi * fc_fir * (n_fir - mid)) / (np.pi * (n_fir - mid)))
    # Hamming window
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n_fir / (M_fir - 1))
    h_fir = h_fir * window
    h_fir = h_fir / np.sum(h_fir)  # normalize

    # Frequency response
    N_freq = 1024
    w_fir = np.linspace(0, np.pi, N_freq)
    H_fir = np.zeros(N_freq, dtype=complex)
    for k_idx in range(M_fir):
        H_fir += h_fir[k_idx] * np.exp(-1j * w_fir * k_idx)

    H_mag_fir = 20 * np.log10(np.abs(H_fir) + 1e-12)
    f_norm = w_fir / np.pi  # 0 to 1 (normalized to f_s/2)

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 7))

    # Coefficients
    markerline4, stemlines4, baseline4 = ax4a.stem(n_fir, h_fir, linefmt="b-", markerfmt="bo", basefmt="k-")
    ax4a.set_xlabel("Tap Index")
    ax4a.set_ylabel("Coefficient Value")
    ax4a.set_title("11-Tap FIR Lowpass Filter Coefficients (Hamming Window)")
    ax4a.grid(True, alpha=0.3)

    # Magnitude response
    ax4b.plot(f_norm, H_mag_fir, "b-", linewidth=2)
    ax4b.axvline(x=0.5, color="red", linestyle="--", alpha=0.6, label="Cutoff (f_s/4)")
    ax4b.axhline(y=-3, color="gray", linestyle=":", alpha=0.6, label="-3 dB")
    ax4b.set_xlabel("Normalized Frequency (×f_s/2)")
    ax4b.set_ylabel("Magnitude (dB)")
    ax4b.set_title("FIR Lowpass Magnitude Response")
    ax4b.set_ylim(-80, 5)
    ax4b.legend()
    ax4b.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4
    return


# --- 8.5.2 IIR Filter: First-order lowpass ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.5.2 IIR Filter — First-Order Lowpass

        A first-order IIR lowpass filter H(z) = 0.1 / (1 − 0.9z⁻¹) has a pole at
        z = 0.9. The magnitude response rolls off from unity at DC to strong attenuation
        at the Nyquist frequency.
        """
    )
    return


@app.cell
def _(np, plt):
    # H(z) = b0 / (1 - a1 * z^-1)
    b0_iir = 0.1
    a1_iir = 0.9

    w_iir = np.linspace(0, np.pi, 1000)
    z_iir = np.exp(1j * w_iir)
    H_iir = b0_iir / (1 - a1_iir * z_iir**(-1))
    H_mag_iir = np.abs(H_iir)
    H_mag_iir_db = 20 * np.log10(H_mag_iir + 1e-12)

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    ax5.plot(w_iir / np.pi, H_mag_iir_db, "b-", linewidth=2)
    ax5.axhline(y=-3, color="gray", linestyle=":", alpha=0.6, label="-3 dB")

    # Mark DC and Nyquist values
    H_dc = b0_iir / (1 - a1_iir)
    H_nyq = b0_iir / (1 + a1_iir)
    ax5.plot(0, 20 * np.log10(H_dc), "go", markersize=8, label=f"DC: |H| = {H_dc:.1f} ({20*np.log10(H_dc):.1f} dB)")
    ax5.plot(1, 20 * np.log10(H_nyq), "ro", markersize=8,
             label=f"Nyquist: |H| = {H_nyq:.4f} ({20*np.log10(H_nyq):.1f} dB)")

    ax5.set_xlabel("Normalized Frequency (×π rad/sample)")
    ax5.set_ylabel("Magnitude (dB)")
    ax5.set_title("IIR Lowpass Filter: H(z) = 0.1 / (1 − 0.9z⁻¹)")
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    fig5.tight_layout()
    fig5
    return


# --- 8.6.1 PSD: White noise power spectral density ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.6.1 Power Spectral Density — White Noise

        White noise with PSD = 2 × 10⁻⁶ V²/Hz over a 10 kHz bandwidth produces
        total noise power of 0.02 V² (RMS voltage = 141 mV). Reducing the measurement
        bandwidth to 2 kHz cuts the total noise power to 0.004 V² (RMS = 63 mV).
        """
    )
    return


@app.cell
def _(np, plt):
    psd_val = 2e-6  # V²/Hz
    bw_full = 10e3  # Hz
    bw_reduced = 2e3  # Hz

    f_psd = np.linspace(0, 15e3, 1000)
    psd_flat = np.full_like(f_psd, psd_val)

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    ax6.plot(f_psd / 1e3, psd_flat * 1e6, "b-", linewidth=2, label="PSD = 2 μV²/Hz")

    # Shade full bandwidth
    mask_full = f_psd <= bw_full
    ax6.fill_between(f_psd[mask_full] / 1e3, psd_flat[mask_full] * 1e6, alpha=0.15, color="blue",
                     label=f"10 kHz BW: P = {psd_val*bw_full*1e3:.0f} mV², Vrms = {np.sqrt(psd_val*bw_full)*1e3:.0f} mV")

    # Shade reduced bandwidth
    mask_reduced = f_psd <= bw_reduced
    ax6.fill_between(f_psd[mask_reduced] / 1e3, psd_flat[mask_reduced] * 1e6, alpha=0.3, color="green",
                     label=f"2 kHz BW: P = {psd_val*bw_reduced*1e3:.0f} mV², Vrms = {np.sqrt(psd_val*bw_reduced)*1e3:.0f} mV")

    ax6.set_xlabel("Frequency (kHz)")
    ax6.set_ylabel("PSD (μV²/Hz)")
    ax6.set_title("White Noise Power Spectral Density")
    ax6.set_ylim(0, 4)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    fig6.tight_layout()
    fig6
    return


# --- 8.1.5 Sampling Theorem and Aliasing ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.1.5 Sampling Theorem and Aliasing

        An audio signal sampled at f_s = 44.1 kHz (CD standard) has a Nyquist frequency
        of 22.05 kHz. A 25 kHz ultrasonic tone that leaks past the anti-aliasing filter
        aliases to f_alias = 44,100 − 25,000 = 19,100 Hz — an audible artifact
        indistinguishable from a genuine 19.1 kHz tone.
        """
    )
    return


@app.cell
def _(np, plt):
    fs_alias = 44100  # Hz
    f_signal = 25000  # Hz  (above Nyquist)
    f_alias = fs_alias - f_signal  # 19100 Hz
    f_legit = 3000  # a legitimate audio tone for context

    # Continuous-time representation (high sample rate for smooth curves)
    t_cont = np.linspace(0, 2e-3, 50000)
    x_legit = np.sin(2 * np.pi * f_legit * t_cont)
    x_ultra = 0.5 * np.sin(2 * np.pi * f_signal * t_cont)
    x_alias_cont = 0.5 * np.sin(2 * np.pi * f_alias * t_cont)

    # Sampled points
    n_samp = np.arange(0, 2e-3, 1 / fs_alias)
    x_legit_samp = np.sin(2 * np.pi * f_legit * n_samp)
    x_ultra_samp = 0.5 * np.sin(2 * np.pi * f_signal * n_samp)

    # Frequency domain illustration
    f_axis = np.linspace(0, fs_alias, 2000)
    f_nyq = fs_alias / 2

    fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(10, 8))

    # Time domain: show the 25 kHz tone and its 19.1 kHz alias
    ax7a.plot(t_cont * 1e3, x_ultra, "r-", linewidth=0.5, alpha=0.3, label="25 kHz original (continuous)")
    ax7a.plot(t_cont * 1e3, x_alias_cont, "b-", linewidth=1.5, alpha=0.7, label="19.1 kHz alias (reconstructed)")
    ax7a.plot(n_samp[:40] * 1e3, x_ultra_samp[:40], "ko", markersize=4, zorder=5, label="Sample points at 44.1 kHz")
    ax7a.set_xlabel("Time (ms)")
    ax7a.set_ylabel("Amplitude")
    ax7a.set_title("Aliasing: 25 kHz Tone Sampled at 44.1 kHz Appears as 19.1 kHz")
    ax7a.set_xlim(0, 0.5)
    ax7a.legend(fontsize=9)
    ax7a.grid(True, alpha=0.3)

    # Frequency domain: spectral folding diagram
    ax7b.axvline(x=f_nyq / 1e3, color="red", linestyle="--", linewidth=2, label=f"Nyquist = {f_nyq/1e3:.2f} kHz")
    ax7b.annotate("", xy=(f_alias / 1e3, 0.6), xytext=(f_signal / 1e3, 0.6),
                  arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
    ax7b.annotate("Folds back", xy=((f_alias + f_signal) / 2 / 1e3, 0.65),
                  fontsize=10, ha="center", color="purple")
    # Original signal bar
    ax7b.bar(f_signal / 1e3, 0.5, width=0.3, color="red", alpha=0.7, label="25 kHz (original)")
    # Alias bar
    ax7b.bar(f_alias / 1e3, 0.5, width=0.3, color="blue", alpha=0.7, label="19.1 kHz (alias)")
    # Audio band
    ax7b.axvspan(0, 20, alpha=0.08, color="green", label="Audible band (0–20 kHz)")
    ax7b.set_xlabel("Frequency (kHz)")
    ax7b.set_ylabel("Magnitude")
    ax7b.set_title("Spectral Folding Around the Nyquist Frequency")
    ax7b.set_xlim(0, 30)
    ax7b.set_ylim(0, 1)
    ax7b.legend(fontsize=9)
    ax7b.grid(True, alpha=0.3)

    fig7.tight_layout()
    fig7
    return


# --- 8.2.6 Hilbert Transform and Analytic Signals ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.2.6 Hilbert Transform — Envelope Extraction

        An amplitude-modulated vibration signal x(t) = (1 + 0.5 cos(2π·8t)) × cos(2π·200t)
        has a 200 Hz carrier modulated at 8 Hz. The analytic signal z(t) = x(t) + jx̂(t)
        recovers the envelope A(t) = |z(t)| = 1 + 0.5 cos(2π·8t) cleanly.
        """
    )
    return


@app.cell
def _(np, plt):
    fs_ht = 2048  # Hz
    t_ht = np.arange(0, 1.0, 1 / fs_ht)
    fc_ht = 200  # carrier
    fm_ht = 8  # modulation

    # AM signal
    envelope_true = 1 + 0.5 * np.cos(2 * np.pi * fm_ht * t_ht)
    x_am = envelope_true * np.cos(2 * np.pi * fc_ht * t_ht)

    # Compute analytic signal via FFT
    X_fft = np.fft.fft(x_am)
    N_ht = len(X_fft)
    h_ht = np.zeros(N_ht)
    h_ht[0] = 1
    h_ht[1:N_ht // 2] = 2
    h_ht[N_ht // 2] = 1
    z_ht = np.fft.ifft(X_fft * h_ht)
    envelope_extracted = np.abs(z_ht)

    fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(10, 7))

    ax8a.plot(t_ht, x_am, "b-", linewidth=0.5, alpha=0.6, label="AM signal x(t)")
    ax8a.plot(t_ht, envelope_true, "r-", linewidth=2, label="True envelope: 1 + 0.5cos(2π·8t)")
    ax8a.plot(t_ht, -envelope_true, "r-", linewidth=2, alpha=0.5)
    ax8a.set_ylabel("Amplitude")
    ax8a.set_title("AM Signal: 200 Hz Carrier Modulated at 8 Hz (m = 50%)")
    ax8a.legend(fontsize=9)
    ax8a.grid(True, alpha=0.3)

    ax8b.plot(t_ht, envelope_extracted, "b-", linewidth=2, label="Extracted envelope |z(t)|")
    ax8b.plot(t_ht, envelope_true, "r--", linewidth=1.5, alpha=0.7, label="True envelope")
    ax8b.axhline(y=1.5, color="gray", linestyle=":", alpha=0.5, label="Max = 1.5")
    ax8b.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Min = 0.5")
    ax8b.set_xlabel("Time (s)")
    ax8b.set_ylabel("Envelope Amplitude")
    ax8b.set_title("Hilbert Transform Envelope Extraction")
    ax8b.legend(fontsize=9)
    ax8b.grid(True, alpha=0.3)

    fig8.tight_layout()
    fig8
    return


# --- 8.4.4 Bilinear Transform ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.4.4 Bilinear Transform — Analog vs Digital Filter

        A first-order analog lowpass prototype with cutoff at 2 kHz is digitized using the
        bilinear transform at f_s = 10 kHz. Pre-warping places the −3 dB point exactly at
        2 kHz in the digital domain. The frequency warping compresses the infinite analog
        axis into [0, f_s/2].
        """
    )
    return


@app.cell
def _(np, plt):
    fs_bt = 10000  # Hz
    T_bt = 1 / fs_bt
    fc_bt = 2000  # desired digital cutoff

    # Pre-warped analog cutoff
    wc_dig = 2 * np.pi * fc_bt / fs_bt  # 0.4π rad/sample
    Wc_analog = (2 / T_bt) * np.tan(wc_dig / 2)  # pre-warped

    # Analog prototype: Ha(s) = Wc / (s + Wc)
    f_analog = np.logspace(1, 4.5, 1000)  # Hz
    s_a = 1j * 2 * np.pi * f_analog
    Ha = Wc_analog / (s_a + Wc_analog)
    Ha_dB = 20 * np.log10(np.abs(Ha))

    # Digital filter from bilinear transform
    # H(z) = 0.4208(1 + z⁻¹) / (1 − 0.1584z⁻¹)
    b0_bt = 0.4208
    a1_bt = -0.1584
    f_digital = np.linspace(1, fs_bt / 2, 1000)
    w_dig = 2 * np.pi * f_digital / fs_bt
    z_bt = np.exp(1j * w_dig)
    Hd = b0_bt * (1 + z_bt**(-1)) / (1 + a1_bt * z_bt**(-1))
    Hd_dB = 20 * np.log10(np.abs(Hd) + 1e-12)

    fig9, ax9 = plt.subplots(figsize=(10, 6))
    ax9.semilogx(f_analog, Ha_dB, "b-", linewidth=2, label="Analog prototype H_a(s)")
    ax9.semilogx(f_digital, Hd_dB, "r-", linewidth=2, label="Digital filter H(z) via bilinear")
    ax9.axhline(y=-3, color="gray", linestyle=":", alpha=0.6, label="−3 dB")
    ax9.axvline(x=fc_bt, color="green", linestyle="--", alpha=0.7, label=f"f_c = {fc_bt} Hz")
    ax9.axvline(x=fs_bt / 2, color="orange", linestyle="--", alpha=0.5, label=f"Nyquist = {fs_bt//2} Hz")

    # Mark −3 dB point on digital filter
    idx_3db = np.argmin(np.abs(Hd_dB - (-3)))
    ax9.plot(f_digital[idx_3db], Hd_dB[idx_3db], "ro", markersize=8, zorder=5)
    ax9.annotate(f"−3 dB at {f_digital[idx_3db]:.0f} Hz", xy=(f_digital[idx_3db], -3),
                 xytext=(f_digital[idx_3db] * 1.5, -8), fontsize=10,
                 arrowprops=dict(arrowstyle="->", color="red"), color="red")

    ax9.set_xlabel("Frequency (Hz)")
    ax9.set_ylabel("Magnitude (dB)")
    ax9.set_title("Bilinear Transform: Analog Prototype vs Digital Implementation (f_s = 10 kHz)")
    ax9.set_xlim(10, 10000)
    ax9.set_ylim(-40, 5)
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3, which="both")

    fig9.tight_layout()
    fig9
    return


# --- 8.5.7 Allpass Filter — Group Delay Equalization ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.5.7 Allpass Filter — Group Delay Equalization

        An allpass filter has unity magnitude at all frequencies (|H(e^jω)| = 1)
        but reshapes the phase response. A second-order allpass section with
        coefficients a₁ = −1.38, a₂ = 0.69 demonstrates flat magnitude with a
        peaked group delay — used to equalize IIR filter phase distortion.
        """
    )
    return


@app.cell
def _(np, plt):
    # Second-order allpass: H(z) = (a2 + a1*z^-1 + z^-2) / (1 + a1*z^-1 + a2*z^-2)
    a1_ap = -1.38
    a2_ap = 0.69

    w_ap = np.linspace(0.001, np.pi, 1000)
    z_ap = np.exp(1j * w_ap)

    H_ap = (a2_ap + a1_ap * z_ap**(-1) + z_ap**(-2)) / (1 + a1_ap * z_ap**(-1) + a2_ap * z_ap**(-2))
    H_ap_mag = np.abs(H_ap)
    H_ap_phase = np.unwrap(np.angle(H_ap))

    # Group delay = -d(phase)/d(omega)
    dw = w_ap[1] - w_ap[0]
    grp_delay = -np.gradient(H_ap_phase, dw)

    fig10, (ax10a, ax10b) = plt.subplots(2, 1, figsize=(10, 7))

    # Magnitude (should be ~1 everywhere)
    ax10a.plot(w_ap / np.pi, 20 * np.log10(H_ap_mag + 1e-12), "b-", linewidth=2)
    ax10a.set_ylabel("Magnitude (dB)")
    ax10a.set_title("Second-Order Allpass Filter (a₁ = −1.38, a₂ = 0.69)")
    ax10a.set_ylim(-1, 1)
    ax10a.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="0 dB (unity gain)")
    ax10a.legend()
    ax10a.grid(True, alpha=0.3)

    # Group delay
    ax10b.plot(w_ap / np.pi, grp_delay, "r-", linewidth=2, label="Group delay τ(ω)")
    peak_idx = np.argmax(grp_delay)
    ax10b.plot(w_ap[peak_idx] / np.pi, grp_delay[peak_idx], "go", markersize=8, zorder=5)
    ax10b.annotate(f"Peak: {grp_delay[peak_idx]:.1f} samples\nat ω = {w_ap[peak_idx]/np.pi:.2f}π",
                   xy=(w_ap[peak_idx] / np.pi, grp_delay[peak_idx]),
                   xytext=(w_ap[peak_idx] / np.pi + 0.15, grp_delay[peak_idx] - 1),
                   fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
    ax10b.set_xlabel("Normalized Frequency (×π rad/sample)")
    ax10b.set_ylabel("Group Delay (samples)")
    ax10b.set_title("Group Delay — Used to Equalize IIR Filter Phase Distortion")
    ax10b.legend()
    ax10b.grid(True, alpha=0.3)

    fig10.tight_layout()
    fig10
    return


# --- 8.6.6 Wavelet Transform — Multi-Level DWT Decomposition ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 8.6.6 Wavelet Transform — Multi-Level Decomposition

        A 3-level discrete wavelet transform decomposes a signal (f_s = 8 kHz) into
        frequency bands by iteratively splitting the approximation coefficients.
        Each level halves the bandwidth: Level 1 detail = 2–4 kHz,
        Level 2 = 1–2 kHz, Level 3 = 0.5–1 kHz, Level 3 approx = 0–0.5 kHz.
        """
    )
    return


@app.cell
def _(np, plt):
    fs_wt = 8000  # Hz
    N_wt = 1024
    t_wt = np.arange(N_wt) / fs_wt

    # Create a test signal: chirp + transient pulse
    # Chirp from 100 Hz to 3000 Hz
    f0_wt = 100
    f1_wt = 3000
    chirp = np.sin(2 * np.pi * (f0_wt * t_wt + (f1_wt - f0_wt) / (2 * t_wt[-1]) * t_wt**2))
    # Add a transient pulse at t = 0.06 s
    pulse_center = int(0.06 * fs_wt)
    pulse = np.zeros(N_wt)
    pulse[pulse_center - 4:pulse_center + 4] = 2.0
    signal_wt = chirp + pulse

    # Simple Haar wavelet DWT (no scipy dependency)
    def haar_dwt_level(x):
        """One level of Haar DWT: returns (approx, detail)."""
        n = len(x) // 2 * 2  # ensure even
        x = x[:n]
        approx = (x[0::2] + x[1::2]) / np.sqrt(2)
        detail = (x[0::2] - x[1::2]) / np.sqrt(2)
        return approx, detail

    # 3-level decomposition
    a1, d1 = haar_dwt_level(signal_wt)   # Level 1
    a2, d2 = haar_dwt_level(a1)           # Level 2
    a3, d3 = haar_dwt_level(a2)           # Level 3

    fig11, axes11 = plt.subplots(5, 1, figsize=(10, 10), sharex=False)

    # Original signal
    axes11[0].plot(t_wt * 1e3, signal_wt, "b-", linewidth=0.8)
    axes11[0].set_ylabel("Amplitude")
    axes11[0].set_title("Original Signal (chirp 100–3000 Hz + transient pulse)")
    axes11[0].set_xlim(0, t_wt[-1] * 1e3)
    axes11[0].grid(True, alpha=0.3)

    # Level 1 detail (2–4 kHz)
    t_d1 = np.linspace(0, t_wt[-1] * 1e3, len(d1))
    axes11[1].plot(t_d1, d1, "r-", linewidth=0.8)
    axes11[1].set_ylabel("d₁")
    axes11[1].set_title("Level 1 Detail: 2–4 kHz (512 coefficients)")
    axes11[1].grid(True, alpha=0.3)

    # Level 2 detail (1–2 kHz)
    t_d2 = np.linspace(0, t_wt[-1] * 1e3, len(d2))
    axes11[2].plot(t_d2, d2, "g-", linewidth=0.8)
    axes11[2].set_ylabel("d₂")
    axes11[2].set_title("Level 2 Detail: 1–2 kHz (256 coefficients)")
    axes11[2].grid(True, alpha=0.3)

    # Level 3 detail (0.5–1 kHz)
    t_d3 = np.linspace(0, t_wt[-1] * 1e3, len(d3))
    axes11[3].plot(t_d3, d3, "m-", linewidth=0.8)
    axes11[3].set_ylabel("d₃")
    axes11[3].set_title("Level 3 Detail: 0.5–1 kHz (128 coefficients)")
    axes11[3].grid(True, alpha=0.3)

    # Level 3 approximation (0–0.5 kHz)
    t_a3 = np.linspace(0, t_wt[-1] * 1e3, len(a3))
    axes11[4].plot(t_a3, a3, "k-", linewidth=0.8)
    axes11[4].set_ylabel("a₃")
    axes11[4].set_title("Level 3 Approximation: 0–0.5 kHz (128 coefficients)")
    axes11[4].set_xlabel("Time (ms)")
    axes11[4].grid(True, alpha=0.3)

    fig11.tight_layout()
    fig11
    return


if __name__ == "__main__":
    app.run()
