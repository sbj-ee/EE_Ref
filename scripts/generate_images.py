"""Generate all PNG images for the EE-Book chapter and appendix examples."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

IMG_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
os.makedirs(IMG_DIR, exist_ok=True)

def save(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Chapter 1: Power Engineering
# ============================================================
print("Chapter 1: Power Engineering")

# --- 1.1 U.S. Electricity Generation by Source ---
sources = [
    "Natural Gas",
    "Coal",
    "Nuclear",
    "Wind",
    "Conv. Hydro",
    "Solar",
    "Biomass — Wood",
    "Petroleum",
    "Biomass — Waste",
    "Geothermal",
    "Other Gases",
    "Pumped Storage",
]
values = np.array([
    1_579_190,
    897_999,
    779_645,
    380_300,
    251_585,
    163_550,
    36_463,
    19_173,
    17_790,
    15_975,
    11_397,
    -5_112,
])

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

order = np.argsort(values)
sorted_sources = [sources[i] for i in order]
sorted_values = values[order]
sorted_colors = [colors[i] for i in order]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(range(len(sorted_sources)), sorted_values / 1000,
                color=sorted_colors, edgecolor="white", linewidth=0.5)

ax.set_yticks(range(len(sorted_sources)))
ax.set_yticklabels(sorted_sources, fontsize=9)
ax.set_xlabel("Net Generation (Thousand GWh)")
ax.set_title("Y2022 U.S. Electricity Net Generation by Source")
ax.grid(True, axis="x", alpha=0.3)

for i, (val, bar) in enumerate(zip(sorted_values, bars)):
    if val >= 0:
        ax.text(val / 1000 + 15, i, f"{val:,.0f}", va="center", fontsize=8)
    else:
        ax.text(val / 1000 - 15, i, f"{val:,.0f}", va="center", fontsize=8,
                 ha="right")

legend_elements = [
    Patch(facecolor="#CC3333", label="Fossil Fuels"),
    Patch(facecolor="#E68A00", label="Nuclear"),
    Patch(facecolor="#33AA33", label="Renewable"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

ax.set_xlim(-100, 1750)
fig.tight_layout()
save(fig, "ch01_energy_mix.png")

# --- 1.3.6 Power Factor Correction ---
P_load = 500  # kW

pf_example = 0.85
theta_ex = np.arccos(pf_example)
S_ex = P_load / pf_example  # kVA
Q_ex = P_load * np.tan(theta_ex)  # kVAR

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))

# Draw power triangle
ax_a.annotate("", xy=(P_load, 0), xytext=(0, 0),
              arrowprops=dict(arrowstyle="->", color="blue", linewidth=2))
ax_a.annotate("", xy=(P_load, -Q_ex), xytext=(P_load, 0),
              arrowprops=dict(arrowstyle="->", color="red", linewidth=2))
ax_a.annotate("", xy=(P_load, -Q_ex), xytext=(0, 0),
              arrowprops=dict(arrowstyle="->", color="green", linewidth=2))

ax_a.text(P_load / 2, 15, f"P = {P_load} kW", ha="center", fontsize=11,
          color="blue", fontweight="bold")
ax_a.text(P_load + 15, -Q_ex / 2, f"Q = {Q_ex:.0f}\nkVAR", ha="left",
          fontsize=10, color="red", fontweight="bold")
ax_a.text(P_load / 2 - 40, -Q_ex / 2 - 20,
          f"S = {S_ex:.0f} kVA", ha="center", fontsize=10,
          color="green", fontweight="bold")

arc_angles = np.linspace(0, -theta_ex, 30)
arc_r = 80
ax_a.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), "k-",
          linewidth=1)
ax_a.text(90, -25, f"\u03b8 = {np.degrees(theta_ex):.1f}\u00b0", fontsize=9)

ax_a.set_xlim(-50, P_load + 100)
ax_a.set_ylim(-Q_ex - 50, 60)
ax_a.set_aspect("equal")
ax_a.set_title(f"Power Triangle at PF = {pf_example}")
ax_a.set_xlabel("Real Power (kW)")
ax_a.set_ylabel("Reactive Power (kVAR)")
ax_a.grid(True, alpha=0.3)
ax_a.axhline(y=0, color="black", linewidth=0.5)
ax_a.axvline(x=0, color="black", linewidth=0.5)

pf_range = np.linspace(0.70, 0.999, 200)
theta_range = np.arccos(pf_range)
Q_original = P_load * np.tan(theta_range)
Q_correction = Q_original

ax_b.plot(pf_range, Q_correction, "b-", linewidth=2)
ax_b.fill_between(pf_range, Q_correction, alpha=0.1, color="blue")

for pf_mark in [0.70, 0.80, 0.85, 0.90, 0.95]:
    q_mark = P_load * np.tan(np.arccos(pf_mark))
    ax_b.plot(pf_mark, q_mark, "ro", markersize=7, zorder=5)
    ax_b.annotate(f"{q_mark:.0f} kVAR",
                  xy=(pf_mark, q_mark),
                  xytext=(pf_mark - 0.03, q_mark + 20),
                  fontsize=8, ha="center")

ax_b.set_xlabel("Original Power Factor")
ax_b.set_ylabel("Capacitive kVAR for Unity PF Correction")
ax_b.set_title(f"Correction kVAR vs Power Factor ({P_load} kW Load)")
ax_b.grid(True, alpha=0.3)
ax_b.set_xlim(0.68, 1.02)

fig.tight_layout()
save(fig, "ch01_power_factor.png")

# --- 1.5.1 Harmonics and THD ---
f0 = 60  # fundamental frequency (Hz)
t = np.linspace(0, 2 / f0, 1000)  # two full cycles

fundamental = np.sin(2 * np.pi * f0 * t)
h3 = 0.20 * np.sin(2 * np.pi * 3 * f0 * t)
h5 = 0.15 * np.sin(2 * np.pi * 5 * f0 * t)
h7 = 0.10 * np.sin(2 * np.pi * 7 * f0 * t)

distorted = fundamental + h3 + h5 + h7

thd = np.sqrt(0.20**2 + 0.15**2 + 0.10**2) * 100  # percent

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

ax_a.plot(t * 1000, fundamental, "b-", linewidth=1.5, alpha=0.6,
          label="Fundamental (60 Hz)")
ax_a.plot(t * 1000, distorted, "r-", linewidth=2,
          label="Distorted Waveform")
ax_a.set_xlabel("Time (ms)")
ax_a.set_ylabel("Amplitude (per-unit)")
ax_a.set_title("Fundamental vs Distorted Waveform")
ax_a.legend(fontsize=9, loc="upper right")
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim(0, 2 / f0 * 1000)

harmonics = [1, 3, 5, 7]
magnitudes = [100.0, 20.0, 15.0, 10.0]
bar_colors = ["#2266CC", "#CC4444", "#CC4444", "#CC4444"]
labels = ["1st\n(60 Hz)", "3rd\n(180 Hz)", "5th\n(300 Hz)", "7th\n(420 Hz)"]

bars = ax_b.bar(labels, magnitudes, color=bar_colors, edgecolor="white",
                width=0.6)

for bar, mag in zip(bars, magnitudes):
    ax_b.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
              f"{mag:.0f}%", ha="center", fontsize=10, fontweight="bold")

ax_b.text(0.95, 0.92, f"THD = {thd:.1f}%",
          transform=ax_b.transAxes, fontsize=12, fontweight="bold",
          ha="right", va="top",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                    edgecolor="gray", alpha=0.9))

ax_b.set_ylabel("Magnitude (% of Fundamental)")
ax_b.set_title("Harmonic Spectrum")
ax_b.grid(True, axis="y", alpha=0.3)
ax_b.set_ylim(0, 115)

fig.tight_layout()
save(fig, "ch01_harmonics_thd.png")


# ============================================================
# Chapter 2: Communications Engineering
# ============================================================
print("Chapter 2: Communications Engineering")

# --- 2.1.1 AM Modulation ---
fc = 1e6       # carrier frequency: 1 MHz
fm = 1e3       # message frequency: 1 kHz
m_idx = 0.8    # modulation index
Ac = 1.0       # carrier amplitude

t_duration = 2e-3
fs_display = 200e3
t = np.linspace(0, t_duration, int(fs_display * t_duration), endpoint=False)

carrier = Ac * np.cos(2 * np.pi * fc * t)
message = m_idx * Ac * np.cos(2 * np.pi * fm * t)
am_signal = (Ac + message) * np.cos(2 * np.pi * fc * t)
envelope_upper = Ac + message
envelope_lower = -(Ac + message)

t_zoom = np.linspace(0, 10e-6, 2000, endpoint=False)
carrier_zoom = Ac * np.cos(2 * np.pi * fc * t_zoom)

N_fft = 2**18
fs_fft = 10e6
t_fft = np.arange(N_fft) / fs_fft
am_fft_signal = (Ac + m_idx * Ac * np.cos(2 * np.pi * fm * t_fft)) * np.cos(2 * np.pi * fc * t_fft)
spectrum = np.abs(np.fft.rfft(am_fft_signal)) / N_fft * 2
freqs = np.fft.rfftfreq(N_fft, 1 / fs_fft)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(t_zoom * 1e6, carrier_zoom, "b-", linewidth=0.8)
axes[0].set_ylabel("Amplitude (V)")
axes[0].set_title("Carrier Signal (f_c = 1 MHz)")
axes[0].set_xlabel("Time (us)")
axes[0].set_xlim(0, 10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t * 1e3, message, "g-", linewidth=2)
axes[1].set_ylabel("Amplitude (V)")
axes[1].set_title(f"Message Signal (f_m = 1 kHz, m = {m_idx})")
axes[1].set_xlabel("Time (ms)")
axes[1].set_xlim(0, t_duration * 1e3)
axes[1].grid(True, alpha=0.3)

axes[2].plot(t * 1e3, am_signal, "b-", linewidth=0.3, alpha=0.6, label="AM signal")
axes[2].plot(t * 1e3, envelope_upper, "r--", linewidth=2, label="Envelope")
axes[2].plot(t * 1e3, envelope_lower, "r--", linewidth=2)
axes[2].set_ylabel("Amplitude (V)")
axes[2].set_title("AM Waveform with Envelope")
axes[2].set_xlabel("Time (ms)")
axes[2].set_xlim(0, t_duration * 1e3)
axes[2].legend(fontsize=9, loc="upper right")
axes[2].grid(True, alpha=0.3)

mask = (freqs >= 0.990e6) & (freqs <= 1.010e6)
axes[3].plot(freqs[mask] / 1e6, spectrum[mask], "b-", linewidth=1.5)
axes[3].set_ylabel("Magnitude")
axes[3].set_xlabel("Frequency (MHz)")
axes[3].set_title("AM Frequency Spectrum")
axes[3].grid(True, alpha=0.3)

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

fig.tight_layout()
save(fig, "ch02_am_modulation.png")

# --- 2.3 Digital Modulation Constellation Diagrams ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

marker_kwargs = dict(marker="o", markersize=10, linestyle="none", color="blue", zorder=5)
circle_kwargs = dict(fill=False, edgecolor="gray", linestyle="--", linewidth=1, alpha=0.5)

# BPSK
ax = axes[0, 0]
bpsk_i = np.array([-1, 1])
bpsk_q = np.array([0, 0])
ax.plot(bpsk_i, bpsk_q, **marker_kwargs)
ax.add_patch(plt.Circle((0, 0), 1, **circle_kwargs))
for i, label in enumerate(["0", "1"]):
    ax.annotate(label, (bpsk_i[i], bpsk_q[i]), textcoords="offset points",
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

# QPSK
ax = axes[0, 1]
qpsk_angles = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4])
qpsk_i = np.cos(qpsk_angles)
qpsk_q = np.sin(qpsk_angles)
ax.plot(qpsk_i, qpsk_q, **marker_kwargs)
ax.add_patch(plt.Circle((0, 0), 1, **circle_kwargs))
qpsk_labels = ["00", "01", "11", "10"]
for i, label in enumerate(qpsk_labels):
    offset_x = 15 if qpsk_i[i] > 0 else -15
    offset_y = 15 if qpsk_q[i] > 0 else -15
    ax.annotate(label, (qpsk_i[i], qpsk_q[i]), textcoords="offset points",
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

# 8-PSK
ax = axes[1, 0]
psk8_angles = np.arange(8) * 2 * np.pi / 8
psk8_i = np.cos(psk8_angles)
psk8_q = np.sin(psk8_angles)
ax.plot(psk8_i, psk8_q, **marker_kwargs)
ax.add_patch(plt.Circle((0, 0), 1, **circle_kwargs))
psk8_labels = ["000", "001", "011", "010", "110", "111", "101", "100"]
for i, label in enumerate(psk8_labels):
    offset_x = 20 * np.cos(psk8_angles[i])
    offset_y = 20 * np.sin(psk8_angles[i])
    ax.annotate(label, (psk8_i[i], psk8_q[i]), textcoords="offset points",
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

# 16-QAM
ax = axes[1, 1]
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

fig.suptitle("Digital Modulation Constellation Diagrams", fontsize=14, y=1.01)
fig.tight_layout()
save(fig, "ch02_constellation.png")

# --- 2.6 Shannon Channel Capacity ---
snr_db = np.linspace(-10, 40, 500)
snr_linear = 10 ** (snr_db / 10)

bandwidths = [1e6, 10e6, 100e6]
bw_labels = ["1 MHz", "10 MHz", "100 MHz"]
bw_colors = ["blue", "green", "red"]

fig, (ax_cap, ax_eff) = plt.subplots(2, 1, figsize=(12, 9))

for bw, label, color in zip(bandwidths, bw_labels, bw_colors):
    capacity = bw * np.log2(1 + snr_linear)
    ax_cap.plot(snr_db, capacity / 1e6, linewidth=2, color=color, label=f"B = {label}")

ax_cap.set_xlabel("SNR (dB)")
ax_cap.set_ylabel("Channel Capacity (Mbps)")
ax_cap.set_title("Shannon Channel Capacity: C = B * log2(1 + SNR)")
ax_cap.legend(fontsize=10)
ax_cap.grid(True, alpha=0.3)
ax_cap.set_xlim(-10, 40)

snr_example = 20
snr_lin_ex = 10 ** (snr_example / 10)
cap_10mhz = 10e6 * np.log2(1 + snr_lin_ex)
ax_cap.plot(snr_example, cap_10mhz / 1e6, "ko", markersize=8, zorder=5)
ax_cap.annotate(f"SNR = 20 dB, B = 10 MHz\nC = {cap_10mhz/1e6:.1f} Mbps",
                xy=(snr_example, cap_10mhz / 1e6),
                xytext=(snr_example + 5, cap_10mhz / 1e6 + 50),
                fontsize=9,
                arrowprops=dict(arrowstyle="->", color="black"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

spectral_eff = np.log2(1 + snr_linear)

ax_eff.plot(snr_db, spectral_eff, "b-", linewidth=2, label="C/B = log2(1 + SNR)")

shannon_limit_db = -1.59
ax_eff.axvline(x=shannon_limit_db, color="red", linestyle="--", linewidth=1.5,
               label=f"Shannon limit (Eb/N0 = {shannon_limit_db} dB)")
ax_eff.annotate(f"Shannon Limit\nEb/N0 = -1.59 dB",
                xy=(shannon_limit_db, 0.5),
                xytext=(shannon_limit_db + 8, 1.0),
                fontsize=10, color="red",
                arrowprops=dict(arrowstyle="->", color="red"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

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

fig.tight_layout()
save(fig, "ch02_shannon_capacity.png")


# ============================================================
# Chapter 3: Semiconductors
# ============================================================
print("Chapter 3: Semiconductors")

# --- 3.1.1 Energy Band Diagram ---
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

materials = [
    ("Conductor\n(Copper)", 0, True),
    ("Semiconductor\n(Silicon, Eg = 1.12 eV)", 1.12, False),
    ("Insulator\n(SiO\u2082, Eg \u2248 9 eV)", 9.0, False),
]

for ax, (title, eg, overlap) in zip(axes, materials):
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 14)
    ax.set_xticks([])
    ax.set_yticks([])
    if ax == axes[0]:
        ax.set_ylabel("Energy (eV)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")

    if overlap:
        ax.fill_between([1, 9], 2, 6, color="#4a90d9", alpha=0.5)
        ax.text(5, 4.0, "Valence\nBand", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#1a3d6e")
        ax.fill_between([1, 9], 4.5, 8.5, color="#e8a040", alpha=0.4)
        ax.text(5, 7.5, "Conduction\nBand", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#8b5e00")
        ax.annotate("Overlap", xy=(5, 5.25), fontsize=9, ha="center",
                    va="center", color="#cc3333", fontweight="bold")
        ax.annotate("", xy=(7.5, 6), xytext=(7.5, 4.5),
                    arrowprops=dict(arrowstyle="<->", color="#cc3333",
                                    linewidth=1.5))
        ax.axhline(y=5.25, xmin=0.1, xmax=0.9, color="black",
                    linestyle="--", linewidth=1.2)
        ax.text(9.2, 5.25, "E\u1da0", fontsize=9, va="center")
    else:
        vb_top = 4.0
        vb_bottom = 1.0
        ax.fill_between([1, 9], vb_bottom, vb_top, color="#4a90d9",
                        alpha=0.5)
        ax.text(5, (vb_top + vb_bottom) / 2, "Valence\nBand",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#1a3d6e")

        cb_bottom = vb_top + eg * (8.0 / 9.0)
        cb_top = cb_bottom + 3.0
        ax.fill_between([1, 9], cb_bottom, cb_top, color="#e8a040",
                        alpha=0.3)
        ax.text(5, (cb_top + cb_bottom) / 2, "Conduction\nBand",
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="#8b5e00")

        mid_x = 5
        ax.annotate("", xy=(mid_x + 3, cb_bottom),
                    xytext=(mid_x + 3, vb_top),
                    arrowprops=dict(arrowstyle="<->", color="#cc3333",
                                    linewidth=1.5))
        gap_label = f"E\u2091 = {eg} eV" if eg < 5 else f"E\u2091 \u2248 {eg:.0f} eV"
        ax.text(mid_x + 3.2, (vb_top + cb_bottom) / 2, gap_label,
                fontsize=9, va="center", ha="left", color="#cc3333",
                fontweight="bold")

        ax.fill_between([1, 9], vb_top, cb_bottom, color="#ffcccc",
                        alpha=0.3, hatch="//")
        ax.text(2.5, (vb_top + cb_bottom) / 2, "Bandgap",
                ha="center", va="center", fontsize=8, color="#cc3333",
                style="italic")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

fig.suptitle("Energy Band Diagrams: Conductor vs Semiconductor vs Insulator",
              fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "ch03_energy_bands.png")

# --- 3.3 PN Junction I-V Curve ---
I_S = 2e-14   # reverse saturation current (A)
V_T = 0.02585  # thermal voltage at 300 K (V)

V_fwd = np.linspace(0, 0.85, 2000)
V_rev = np.linspace(-2.0, 0, 500)
V_full = np.concatenate([V_rev, V_fwd[1:]])

I_n1_full = I_S * (np.exp(V_full / (1 * V_T)) - 1)
I_n2_full = I_S * (np.exp(V_full / (2 * V_T)) - 1)

I_n1_clip = np.clip(I_n1_full, -1e-10, 0.1)
I_n2_clip = np.clip(I_n2_full, -1e-10, 0.1)

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

ax_a.plot(V_full * 1000, I_n1_clip * 1000, "b-", linewidth=2,
          label="n = 1 (ideal)")
ax_a.plot(V_full * 1000, I_n2_clip * 1000, "r--", linewidth=2,
          label="n = 2 (recombination)")
ax_a.axhline(y=0, color="black", linewidth=0.5)
ax_a.axvline(x=0, color="black", linewidth=0.5)

knee_v = 700  # mV
knee_idx = np.argmin(np.abs(V_full * 1000 - knee_v))
knee_i = I_n1_clip[knee_idx] * 1000
ax_a.plot(knee_v, knee_i, "ko", markersize=8, zorder=5)
ax_a.annotate(f"Knee \u2248 0.7 V\n({knee_i:.1f} mA)",
              xy=(knee_v, knee_i),
              xytext=(450, knee_i * 0.85),
              fontsize=9, color="black",
              arrowprops=dict(arrowstyle="->", color="black"),
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                        alpha=0.8))

ax_a.annotate(f"I\u209b \u2248 {I_S:.0e} A\n(leakage)",
              xy=(-1500, -I_S * 1000),
              xytext=(-1500, 20),
              fontsize=9, color="gray",
              arrowprops=dict(arrowstyle="->", color="gray"))

ax_a.set_xlabel("Voltage (mV)", fontsize=10)
ax_a.set_ylabel("Current (mA)", fontsize=10)
ax_a.set_title("PN Junction I-V (Linear Scale)", fontsize=11,
                fontweight="bold")
ax_a.legend(fontsize=9, loc="upper left")
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim(-2100, 900)
ax_a.set_ylim(-5, 100)

V_log = np.linspace(0.1, 0.80, 1500)
I_n1_log = I_S * (np.exp(V_log / (1 * V_T)) - 1)
I_n2_log = I_S * (np.exp(V_log / (2 * V_T)) - 1)

ax_b.semilogy(V_log * 1000, I_n1_log, "b-", linewidth=2,
              label="n = 1 (ideal)")
ax_b.semilogy(V_log * 1000, I_n2_log, "r--", linewidth=2,
              label="n = 2 (recombination)")

ax_b.annotate("Slope = q/(nkT)\nn=1: 38.7 /V\nn=2: 19.3 /V",
              xy=(350, 1e-6), fontsize=9,
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        alpha=0.8))

ax_b.axvline(x=knee_v, color="gray", linestyle=":", alpha=0.7)
ax_b.text(knee_v + 10, 1e-12, "0.7 V", fontsize=9, color="gray",
          rotation=90, va="bottom")

ax_b.set_xlabel("Voltage (mV)", fontsize=10)
ax_b.set_ylabel("Current (A)", fontsize=10)
ax_b.set_title("PN Junction I-V (Log Scale \u2014 Forward Bias)",
                fontsize=11, fontweight="bold")
ax_b.legend(fontsize=9, loc="upper left")
ax_b.grid(True, alpha=0.3, which="both")
ax_b.set_xlim(100, 820)
ax_b.set_ylim(1e-14, 1)

fig.tight_layout()
save(fig, "ch03_pn_junction.png")

# --- 3.5.2 MOSFET Characteristics ---
V_th = 1.0       # threshold voltage (V)
k_n = 1.0e-3     # transconductance parameter (A/V^2)

V_DS = np.linspace(0, 8, 500)
V_GS_values = [2.0, 3.0, 4.0, 5.0]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

for v_gs, color in zip(V_GS_values, colors):
    v_ov = v_gs - V_th
    I_D = np.where(
        V_DS < v_ov,
        k_n * ((v_gs - V_th) * V_DS - 0.5 * V_DS**2),
        0.5 * k_n * (v_gs - V_th)**2
    )
    ax_a.plot(V_DS, I_D * 1000, color=color, linewidth=2,
              label=f"V\u0047\u0053 = {v_gs:.0f} V")

v_ds_boundary = np.linspace(0, max(V_GS_values) - V_th, 200)
i_boundary = 0.5 * k_n * v_ds_boundary**2
ax_a.plot(v_ds_boundary, i_boundary * 1000, "k--", linewidth=1.5,
          alpha=0.6, label="V\u1d30\u209b = V\u1d33\u209b \u2212 V\u209c\u2095")

ax_a.text(1.0, 7.0, "Linear\n(Triode)", fontsize=9, ha="center",
          style="italic", color="gray")
ax_a.text(5.5, 7.0, "Saturation", fontsize=9, ha="center",
          style="italic", color="gray")

ax_a.set_xlabel("V\u1d30\u209b (V)", fontsize=10)
ax_a.set_ylabel("I\u1d30 (mA)", fontsize=10)
ax_a.set_title("MOSFET Output Characteristics", fontsize=11,
                fontweight="bold")
ax_a.legend(fontsize=9, loc="center right")
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim(0, 8)
ax_a.set_ylim(0, 9)

V_GS_sweep = np.linspace(0, 6, 500)
V_DS_fixed = 5.0

I_D_transfer = np.where(
    V_GS_sweep <= V_th,
    0,
    np.where(
        V_DS_fixed < (V_GS_sweep - V_th),
        k_n * ((V_GS_sweep - V_th) * V_DS_fixed - 0.5 * V_DS_fixed**2),
        0.5 * k_n * (V_GS_sweep - V_th)**2
    )
)

ax_b.plot(V_GS_sweep, I_D_transfer * 1000, "b-", linewidth=2.5)

ax_b.axvline(x=V_th, color="red", linestyle="--", linewidth=1.2,
             alpha=0.7)
ax_b.annotate(f"V\u209c\u2095 = {V_th:.0f} V",
              xy=(V_th, 0.3),
              xytext=(V_th + 0.8, 1.5),
              fontsize=10, color="red", fontweight="bold",
              arrowprops=dict(arrowstyle="->", color="red"))

ax_b.annotate("I\u1d30 = (k\u2099/2)(V\u1d33\u209b \u2212 V\u209c\u2095)\u00b2",
              xy=(3.5, 0.5 * k_n * (3.5 - V_th)**2 * 1000),
              xytext=(1.5, 7.5),
              fontsize=10, color="blue",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                        alpha=0.8))

for v_gs in [2.0, 3.0, 4.0, 5.0]:
    i_d = 0.5 * k_n * (v_gs - V_th)**2 * 1000
    ax_b.plot(v_gs, i_d, "ko", markersize=5, zorder=5)
    ax_b.text(v_gs + 0.1, i_d + 0.2, f"{i_d:.1f} mA", fontsize=8,
              color="black")

ax_b.set_xlabel("V\u1d33\u209b (V)", fontsize=10)
ax_b.set_ylabel("I\u1d30 (mA)", fontsize=10)
ax_b.set_title(f"MOSFET Transfer Characteristic (V\u1d30\u209b = {V_DS_fixed:.0f} V)",
                fontsize=11, fontweight="bold")
ax_b.grid(True, alpha=0.3)
ax_b.set_xlim(0, 6)
ax_b.set_ylim(0, 13)

fig.tight_layout()
save(fig, "ch03_mosfet.png")


# ============================================================
# Chapter 7: Circuit Analysis
# ============================================================
print("Chapter 7: Circuit Analysis")

# --- 7.5.1 Impedance ---
R_rl = 100; L_rl = 50e-3
f_rl = np.linspace(1, 1000, 1000)
omega_rl = 2 * np.pi * f_rl
X_L = omega_rl * L_rl
Z_mag_rl = np.sqrt(R_rl**2 + X_L**2)
Z_phase_rl = np.degrees(np.arctan2(X_L, R_rl))
f_ex = 60; omega_ex = 2*np.pi*f_ex
Z_ex = np.sqrt(R_rl**2 + (omega_ex*L_rl)**2)
theta_ex = np.degrees(np.arctan2(omega_ex*L_rl, R_rl))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(f_rl, Z_mag_rl, "b-", linewidth=2)
ax1.axhline(y=R_rl, color="gray", linestyle="--", alpha=0.6, label="R = 100 Ω")
ax1.plot(f_ex, Z_ex, "ro", markersize=8, label=f"60 Hz: |Z| = {Z_ex:.1f} Ω")
ax1.set_ylabel("|Z| (Ω)"); ax1.set_title("Series RL Impedance vs Frequency (R = 100 Ω, L = 50 mH)")
ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(f_rl, Z_phase_rl, "r-", linewidth=2)
ax2.plot(f_ex, theta_ex, "ro", markersize=8, label=f"60 Hz: θ = {theta_ex:.1f}°")
ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Phase Angle (°)"); ax2.set_ylim(0, 90)
ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch07_impedance_rl.png")

# --- 7.5.2 Resonance ---
R_r = 10; L_r = 1e-3; C_r = 10e-9
f0_r = 1/(2*np.pi*np.sqrt(L_r*C_r))
Q_r = (1/R_r)*np.sqrt(L_r/C_r); BW_r = f0_r/Q_r
f_r = np.linspace(10e3, 100e3, 5000)
omega_r = 2*np.pi*f_r
X_r = omega_r*L_r - 1/(omega_r*C_r)
Z_mag_r = np.sqrt(R_r**2 + X_r**2)
Z_phase_r = np.degrees(np.arctan2(X_r, R_r))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.semilogy(f_r/1e3, Z_mag_r, "b-", linewidth=2)
ax1.axvline(x=f0_r/1e3, color="red", linestyle="--", alpha=0.6, label=f"f₀ = {f0_r/1e3:.1f} kHz")
ax1.axhline(y=R_r, color="gray", linestyle=":", alpha=0.6, label=f"R = {R_r} Ω (minimum)")
f_low=(f0_r-BW_r/2)/1e3; f_high=(f0_r+BW_r/2)/1e3
ax1.axvspan(f_low, f_high, alpha=0.15, color="green", label=f"BW = {BW_r:.0f} Hz, Q = {Q_r:.1f}")
ax1.set_ylabel("|Z| (Ω)"); ax1.set_title(f"Series RLC Resonance (R={R_r} Ω, L={L_r*1e3} mH, C={C_r*1e9} nF)")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, which="both")
ax2.plot(f_r/1e3, Z_phase_r, "r-", linewidth=2)
ax2.axhline(y=0, color="gray", linestyle=":", alpha=0.6)
ax2.axvline(x=f0_r/1e3, color="red", linestyle="--", alpha=0.6)
ax2.set_xlabel("Frequency (kHz)"); ax2.set_ylabel("Phase Angle (°)"); ax2.set_ylim(-90, 90)
ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch07_resonance_rlc.png")

# --- 7.6.1 RC Charging ---
R_rc=47e3; C_rc=10e-6; Vs_rc=9; tau_rc=R_rc*C_rc
t_rc = np.linspace(0, 5*tau_rc, 500)
Vc_rc = Vs_rc*(1-np.exp(-t_rc/tau_rc))
Vc_at_tau = Vs_rc*(1-np.exp(-1)); Vc_05 = Vs_rc*(1-np.exp(-0.5/tau_rc))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_rc, Vc_rc, "b-", linewidth=2, label="Vc(t) = 9(1 − e⁻ᵗ/⁰·⁴⁷)")
ax.axhline(y=Vs_rc, color="gray", linestyle="--", alpha=0.5, label="Vs = 9 V")
ax.plot(tau_rc, Vc_at_tau, "go", markersize=10, zorder=5)
ax.annotate(f"τ = {tau_rc} s\nVc = {Vc_at_tau:.2f} V (63.2%)", xy=(tau_rc, Vc_at_tau),
            xytext=(tau_rc+0.3, Vc_at_tau-1.5), fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.plot(0.5, Vc_05, "ro", markersize=10, zorder=5)
ax.annotate(f"t = 0.5 s\nVc = {Vc_05:.2f} V", xy=(0.5, Vc_05), xytext=(0.8, Vc_05-2),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.axvline(x=5*tau_rc, color="orange", linestyle=":", alpha=0.6, label=f"5τ = {5*tau_rc:.2f} s (≈99.3%)")
ax.set_xlabel("Time (s)"); ax.set_ylabel("Capacitor Voltage (V)")
ax.set_title("RC Charging Curve (R = 47 kΩ, C = 10 μF, Vs = 9 V)")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlim(0, 5*tau_rc); ax.set_ylim(0, 10)
fig.tight_layout()
save(fig, "ch07_rc_charging.png")

# --- 7.6.2 RL Transient ---
R_rl2=50; L_rl2=200e-3; Vs_rl2=24; tau_rl2=L_rl2/R_rl2; Iss=Vs_rl2/R_rl2
t_rl2 = np.linspace(0, 5*tau_rl2, 500)
I_rl2 = Iss*(1-np.exp(-t_rl2/tau_rl2)); VL_rl2 = Vs_rl2*np.exp(-t_rl2/tau_rl2)
t_ex2=2e-3; I_ex2=Iss*(1-np.exp(-t_ex2/tau_rl2)); VL_ex2=Vs_rl2*np.exp(-t_ex2/tau_rl2)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()
l1, = ax1.plot(t_rl2*1e3, I_rl2*1e3, "b-", linewidth=2, label="I(t)")
ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Current (mA)", color="blue"); ax1.tick_params(axis="y", labelcolor="blue")
l2, = ax2.plot(t_rl2*1e3, VL_rl2, "r-", linewidth=2, label="V_L(t)")
ax2.set_ylabel("Inductor Voltage (V)", color="red"); ax2.tick_params(axis="y", labelcolor="red")
ax1.plot(2, I_ex2*1e3, "bo", markersize=8, zorder=5)
ax2.plot(2, VL_ex2, "ro", markersize=8, zorder=5)
ax1.annotate(f"t=2 ms: I={I_ex2*1e3:.1f} mA", xy=(2, I_ex2*1e3), xytext=(6, I_ex2*1e3),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="blue"), color="blue")
ax2.annotate(f"t=2 ms: V_L={VL_ex2:.2f} V", xy=(2, VL_ex2), xytext=(6, VL_ex2+2),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax1.set_title("RL Circuit Transient (R = 50 Ω, L = 200 mH, Vs = 24 V, τ = 4 ms)")
ax1.legend([l1, l2], ["I(t)", "V_L(t)"], loc="center right"); ax1.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch07_rl_transient.png")

# --- 7.6.3 RLC Underdamped ---
R_rlc=100; L_rlc=10e-3; C_rlc=1e-6
omega0=1/np.sqrt(L_rlc*C_rlc); zeta=R_rlc/(2*np.sqrt(L_rlc/C_rlc))
omega_d=omega0*np.sqrt(1-zeta**2); sigma=zeta*omega0
t_rlc = np.linspace(0, 2e-3, 1000)
env = np.exp(-sigma*t_rlc); phi=np.arccos(zeta)
Vc = 1.0*(1-(1/np.sqrt(1-zeta**2))*env*np.sin(omega_d*t_rlc+phi))
env_u = 1.0*(1+(1/np.sqrt(1-zeta**2))*env)
env_l = 1.0*(1-(1/np.sqrt(1-zeta**2))*env)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_rlc*1e3, Vc, "b-", linewidth=2, label="Vc(t) — underdamped (ζ = 0.5)")
ax.plot(t_rlc*1e3, env_u, "r--", alpha=0.5, linewidth=1, label="Envelope")
ax.plot(t_rlc*1e3, env_l, "r--", alpha=0.5, linewidth=1)
ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6, label="Steady state")
T_d=2*np.pi/omega_d
ax.annotate(f"T_d = {T_d*1e3:.3f} ms\n(f_d = {omega_d/(2*np.pi):.0f} Hz)",
            xy=(T_d*1e3, Vc[int(T_d/t_rlc[-1]*len(t_rlc))]), xytext=(1.2, 1.35),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Normalized Capacitor Voltage")
ax.set_title(f"Underdamped RLC Step Response (ζ = {zeta}, ω₀ = {omega0:.0f} rad/s, ω_d = {omega_d:.0f} rad/s)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch07_rlc_underdamped.png")


# ============================================================
# Chapter 8: Signal Processing
# ============================================================
print("Chapter 8: Signal Processing")

# --- 8.2.1 Fourier Series ---
f_sq=500; t_sq=np.linspace(0, 4e-3, 2000)
h1=(4*2.5/(1*np.pi))*np.sin(2*np.pi*1*f_sq*t_sq)
h3=(4*2.5/(3*np.pi))*np.sin(2*np.pi*3*f_sq*t_sq)
h5=(4*2.5/(5*np.pi))*np.sin(2*np.pi*5*f_sq*t_sq)
composite=h1+h3+h5; sq_ideal=2.5*np.sign(np.sin(2*np.pi*f_sq*t_sq))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(t_sq*1e3, h1, "b-", linewidth=1.5, label=f"1st (500 Hz): {4*2.5/np.pi:.3f} V", alpha=0.8)
ax1.plot(t_sq*1e3, h3, "r-", linewidth=1.5, label=f"3rd (1500 Hz): {4*2.5/(3*np.pi):.3f} V", alpha=0.8)
ax1.plot(t_sq*1e3, h5, "g-", linewidth=1.5, label=f"5th (2500 Hz): {4*2.5/(5*np.pi):.3f} V", alpha=0.8)
ax1.set_ylabel("Voltage (V)"); ax1.set_title("Individual Fourier Harmonics of a 500 Hz Square Wave")
ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(t_sq*1e3, sq_ideal, "k--", linewidth=1, label="Ideal square wave", alpha=0.5)
ax2.plot(t_sq*1e3, composite, "b-", linewidth=2, label="Sum of 3 harmonics")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Voltage (V)")
ax2.set_title("Composite Waveform (1st + 3rd + 5th Harmonic)"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_fourier_series.png")

# --- 8.2.2 Rectangular pulse spectrum ---
t_p=np.linspace(-2e-3, 2e-3, 1000); pw=1e-3
x_p=np.where(np.abs(t_p)<=pw/2, 1.0, 0.0)
f_s=np.linspace(-5000, 5000, 2000); X_s=pw*np.sinc(f_s*pw)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(t_p*1e3, x_p, "b-", linewidth=2); ax1.fill_between(t_p*1e3, x_p, alpha=0.2)
ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Amplitude"); ax1.set_title("Rectangular Pulse (width = 1 ms)")
ax1.set_ylim(-0.2, 1.3); ax1.grid(True, alpha=0.3)
ax2.plot(f_s/1e3, np.abs(X_s)*1e3, "r-", linewidth=2)
ax2.axvline(x=1, color="green", linestyle="--", alpha=0.7, label="First null at 1 kHz")
ax2.axvline(x=-1, color="green", linestyle="--", alpha=0.7)
ax2.set_xlabel("Frequency (kHz)"); ax2.set_ylabel("|X(f)| (×10⁻³)")
ax2.set_title("Magnitude Spectrum |X(f)| = τ|sinc(fτ)|"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_fourier_transform.png")

# --- 8.2.3 DFT ---
N_d=512; fs_d=16000; delta_f=fs_d/N_d; f_tone=1500; k_tone=int(f_tone/delta_f)
n_d=np.arange(N_d); x_d=np.sin(2*np.pi*f_tone/fs_d*n_d)
X_d=np.fft.fft(x_d); X_mag=2*np.abs(X_d[:N_d//2])/N_d; f_bins=np.arange(N_d//2)*delta_f

fig, ax = plt.subplots(figsize=(10, 5))
ml, sl, bl = ax.stem(f_bins/1e3, X_mag, linefmt="b-", markerfmt="b.", basefmt="k-")
sl.set_linewidth(0.5); ml.set_markersize(2)
ax.plot(f_bins[k_tone]/1e3, X_mag[k_tone], "ro", markersize=10, zorder=5, label=f"Bin {k_tone}: f = {f_bins[k_tone]:.0f} Hz")
ax.annotate(f"k = {k_tone}\nf = {f_bins[k_tone]:.0f} Hz", xy=(f_bins[k_tone]/1e3, X_mag[k_tone]),
            xytext=(f_bins[k_tone]/1e3+0.8, X_mag[k_tone]-0.1), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("Magnitude")
ax.set_title(f"512-point DFT (f_s = 16 kHz, Δf = {delta_f} Hz)")
ax.set_xlim(0, fs_d/2/1e3); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_dft_bins.png")

# --- 8.5.1 FIR ---
M_f=11; fc_f=0.25; n_f=np.arange(M_f); mid=(M_f-1)/2
h_f=np.where(n_f==mid, 2*fc_f, np.sin(2*np.pi*fc_f*(n_f-mid))/(np.pi*(n_f-mid)))
win=0.54-0.46*np.cos(2*np.pi*n_f/(M_f-1)); h_f=h_f*win; h_f=h_f/np.sum(h_f)
N_fr=1024; w_f=np.linspace(0, np.pi, N_fr)
H_f=np.zeros(N_fr, dtype=complex)
for k in range(M_f): H_f+=h_f[k]*np.exp(-1j*w_f*k)
H_mag_f=20*np.log10(np.abs(H_f)+1e-12); f_norm=w_f/np.pi

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.stem(n_f, h_f, linefmt="b-", markerfmt="bo", basefmt="k-")
ax1.set_xlabel("Tap Index"); ax1.set_ylabel("Coefficient"); ax1.set_title("11-Tap FIR Lowpass Coefficients (Hamming)")
ax1.grid(True, alpha=0.3)
ax2.plot(f_norm, H_mag_f, "b-", linewidth=2)
ax2.axvline(x=0.5, color="red", linestyle="--", alpha=0.6, label="Cutoff (f_s/4)")
ax2.axhline(y=-3, color="gray", linestyle=":", alpha=0.6, label="-3 dB")
ax2.set_xlabel("Normalized Frequency (×f_s/2)"); ax2.set_ylabel("Magnitude (dB)")
ax2.set_title("FIR Lowpass Magnitude Response"); ax2.set_ylim(-80, 5); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_fir_filter.png")

# --- 8.5.2 IIR ---
b0=0.1; a1=0.9; w_i=np.linspace(0, np.pi, 1000)
z_i=np.exp(1j*w_i); H_i=b0/(1-a1*z_i**(-1))
H_dc=b0/(1-a1); H_nyq=b0/(1+a1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(w_i/np.pi, 20*np.log10(np.abs(H_i)+1e-12), "b-", linewidth=2)
ax.axhline(y=-3, color="gray", linestyle=":", alpha=0.6, label="-3 dB")
ax.plot(0, 20*np.log10(H_dc), "go", markersize=8, label=f"DC: |H| = {H_dc:.1f} ({20*np.log10(H_dc):.1f} dB)")
ax.plot(1, 20*np.log10(H_nyq), "ro", markersize=8, label=f"Nyquist: |H| = {H_nyq:.4f} ({20*np.log10(H_nyq):.1f} dB)")
ax.set_xlabel("Normalized Frequency (×π rad/sample)"); ax.set_ylabel("Magnitude (dB)")
ax.set_title("IIR Lowpass: H(z) = 0.1 / (1 − 0.9z⁻¹)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_iir_filter.png")

# --- 8.6.1 PSD ---
psd_val=2e-6; bw_full=10e3; bw_red=2e3
f_psd=np.linspace(0, 15e3, 1000); psd_flat=np.full_like(f_psd, psd_val)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(f_psd/1e3, psd_flat*1e6, "b-", linewidth=2, label="PSD = 2 μV²/Hz")
m1=f_psd<=bw_full; m2=f_psd<=bw_red
ax.fill_between(f_psd[m1]/1e3, psd_flat[m1]*1e6, alpha=0.15, color="blue",
                label=f"10 kHz BW: Vrms = {np.sqrt(psd_val*bw_full)*1e3:.0f} mV")
ax.fill_between(f_psd[m2]/1e3, psd_flat[m2]*1e6, alpha=0.3, color="green",
                label=f"2 kHz BW: Vrms = {np.sqrt(psd_val*bw_red)*1e3:.0f} mV")
ax.set_xlabel("Frequency (kHz)"); ax.set_ylabel("PSD (μV²/Hz)")
ax.set_title("White Noise Power Spectral Density"); ax.set_ylim(0, 4); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_psd.png")

# --- 8.1.5 Sampling and Aliasing ---
fs_al=44100; f_sig=25000; f_al=fs_al-f_sig; f_nyq_al=fs_al/2
t_c=np.linspace(0, 2e-3, 50000)
x_ultra=0.5*np.sin(2*np.pi*f_sig*t_c)
x_alias_c=0.5*np.sin(2*np.pi*f_al*t_c)
n_s=np.arange(0, 2e-3, 1/fs_al)
x_ultra_s=0.5*np.sin(2*np.pi*f_sig*n_s)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(t_c*1e3, x_ultra, "r-", linewidth=0.5, alpha=0.3, label="25 kHz original (continuous)")
ax1.plot(t_c*1e3, x_alias_c, "b-", linewidth=1.5, alpha=0.7, label="19.1 kHz alias (reconstructed)")
ax1.plot(n_s[:40]*1e3, x_ultra_s[:40], "ko", markersize=4, zorder=5, label="Sample points at 44.1 kHz")
ax1.set_xlabel("Time (ms)"); ax1.set_ylabel("Amplitude")
ax1.set_title("Aliasing: 25 kHz Tone Sampled at 44.1 kHz Appears as 19.1 kHz")
ax1.set_xlim(0, 0.5); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.axvline(x=f_nyq_al/1e3, color="red", linestyle="--", linewidth=2, label=f"Nyquist = {f_nyq_al/1e3:.2f} kHz")
ax2.annotate("", xy=(f_al/1e3, 0.6), xytext=(f_sig/1e3, 0.6),
             arrowprops=dict(arrowstyle="<->", color="purple", lw=2))
ax2.annotate("Folds back", xy=((f_al+f_sig)/2/1e3, 0.65), fontsize=10, ha="center", color="purple")
ax2.bar(f_sig/1e3, 0.5, width=0.3, color="red", alpha=0.7, label="25 kHz (original)")
ax2.bar(f_al/1e3, 0.5, width=0.3, color="blue", alpha=0.7, label="19.1 kHz (alias)")
ax2.axvspan(0, 20, alpha=0.08, color="green", label="Audible band (0–20 kHz)")
ax2.set_xlabel("Frequency (kHz)"); ax2.set_ylabel("Magnitude")
ax2.set_title("Spectral Folding Around the Nyquist Frequency")
ax2.set_xlim(0, 30); ax2.set_ylim(0, 1); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_aliasing.png")

# --- 8.2.6 Hilbert Transform ---
fs_ht=2048; t_ht=np.arange(0, 1.0, 1/fs_ht)
fc_ht=200; fm_ht=8
env_true=1+0.5*np.cos(2*np.pi*fm_ht*t_ht)
x_am=env_true*np.cos(2*np.pi*fc_ht*t_ht)
X_fft=np.fft.fft(x_am); N_ht=len(X_fft)
h_ht=np.zeros(N_ht); h_ht[0]=1; h_ht[1:N_ht//2]=2; h_ht[N_ht//2]=1
z_ht=np.fft.ifft(X_fft*h_ht); env_ext=np.abs(z_ht)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(t_ht, x_am, "b-", linewidth=0.5, alpha=0.6, label="AM signal x(t)")
ax1.plot(t_ht, env_true, "r-", linewidth=2, label="True envelope")
ax1.plot(t_ht, -env_true, "r-", linewidth=2, alpha=0.5)
ax1.set_ylabel("Amplitude"); ax1.set_title("AM Signal: 200 Hz Carrier Modulated at 8 Hz (m = 50%)")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax2.plot(t_ht, env_ext, "b-", linewidth=2, label="Extracted envelope |z(t)|")
ax2.plot(t_ht, env_true, "r--", linewidth=1.5, alpha=0.7, label="True envelope")
ax2.axhline(y=1.5, color="gray", linestyle=":", alpha=0.5, label="Max = 1.5")
ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Min = 0.5")
ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Envelope Amplitude")
ax2.set_title("Hilbert Transform Envelope Extraction"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_hilbert.png")

# --- 8.4.4 Bilinear Transform ---
fs_bt=10000; T_bt=1/fs_bt; fc_bt=2000
wc_dig=2*np.pi*fc_bt/fs_bt
Wc_an=(2/T_bt)*np.tan(wc_dig/2)
f_an=np.logspace(1, 4.5, 1000)
Ha=Wc_an/(1j*2*np.pi*f_an+Wc_an); Ha_dB=20*np.log10(np.abs(Ha))
b0_bt=0.4208; a1_bt=-0.1584
f_dig=np.linspace(1, fs_bt/2, 1000); w_dig=2*np.pi*f_dig/fs_bt
z_bt=np.exp(1j*w_dig)
Hd=b0_bt*(1+z_bt**(-1))/(1+a1_bt*z_bt**(-1))
Hd_dB=20*np.log10(np.abs(Hd)+1e-12)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(f_an, Ha_dB, "b-", linewidth=2, label="Analog prototype H_a(s)")
ax.semilogx(f_dig, Hd_dB, "r-", linewidth=2, label="Digital filter H(z) via bilinear")
ax.axhline(y=-3, color="gray", linestyle=":", alpha=0.6, label="−3 dB")
ax.axvline(x=fc_bt, color="green", linestyle="--", alpha=0.7, label=f"f_c = {fc_bt} Hz")
ax.axvline(x=fs_bt/2, color="orange", linestyle="--", alpha=0.5, label=f"Nyquist = {fs_bt//2} Hz")
idx_3db=np.argmin(np.abs(Hd_dB-(-3)))
ax.plot(f_dig[idx_3db], Hd_dB[idx_3db], "ro", markersize=8, zorder=5)
ax.annotate(f"−3 dB at {f_dig[idx_3db]:.0f} Hz", xy=(f_dig[idx_3db], -3),
            xytext=(f_dig[idx_3db]*1.5, -8), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")
ax.set_title("Bilinear Transform: Analog vs Digital (f_s = 10 kHz)")
ax.set_xlim(10, 10000); ax.set_ylim(-40, 5); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
save(fig, "ch08_bilinear.png")

# --- 8.5.7 Allpass Filter ---
a1_ap=-1.38; a2_ap=0.69
w_ap=np.linspace(0.001, np.pi, 1000); z_ap=np.exp(1j*w_ap)
H_ap=(a2_ap+a1_ap*z_ap**(-1)+z_ap**(-2))/(1+a1_ap*z_ap**(-1)+a2_ap*z_ap**(-2))
H_ap_mag=np.abs(H_ap); H_ap_phase=np.unwrap(np.angle(H_ap))
dw_ap=w_ap[1]-w_ap[0]; grp_delay=-np.gradient(H_ap_phase, dw_ap)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.plot(w_ap/np.pi, 20*np.log10(H_ap_mag+1e-12), "b-", linewidth=2)
ax1.set_ylabel("Magnitude (dB)"); ax1.set_title("Second-Order Allpass Filter (a₁ = −1.38, a₂ = 0.69)")
ax1.set_ylim(-1, 1); ax1.axhline(y=0, color="gray", linestyle=":", alpha=0.5, label="0 dB (unity)")
ax1.legend(); ax1.grid(True, alpha=0.3)
pk=np.argmax(grp_delay)
ax2.plot(w_ap/np.pi, grp_delay, "r-", linewidth=2, label="Group delay τ(ω)")
ax2.plot(w_ap[pk]/np.pi, grp_delay[pk], "go", markersize=8, zorder=5)
ax2.annotate(f"Peak: {grp_delay[pk]:.1f} samples\nat ω = {w_ap[pk]/np.pi:.2f}π",
             xy=(w_ap[pk]/np.pi, grp_delay[pk]),
             xytext=(w_ap[pk]/np.pi+0.15, grp_delay[pk]-1),
             fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax2.set_xlabel("Normalized Frequency (×π rad/sample)"); ax2.set_ylabel("Group Delay (samples)")
ax2.set_title("Group Delay — Used to Equalize IIR Filter Phase Distortion")
ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_allpass.png")

# --- 8.6.6 Wavelet Transform ---
fs_wt=8000; N_wt=1024; t_wt=np.arange(N_wt)/fs_wt
f0_wt=100; f1_wt=3000
chirp_wt=np.sin(2*np.pi*(f0_wt*t_wt+(f1_wt-f0_wt)/(2*t_wt[-1])*t_wt**2))
pulse_wt=np.zeros(N_wt); pc=int(0.06*fs_wt); pulse_wt[pc-4:pc+4]=2.0
sig_wt=chirp_wt+pulse_wt

def haar_dwt(x):
    n=len(x)//2*2; x=x[:n]
    return (x[0::2]+x[1::2])/np.sqrt(2), (x[0::2]-x[1::2])/np.sqrt(2)

a1_wt, d1_wt=haar_dwt(sig_wt)
a2_wt, d2_wt=haar_dwt(a1_wt)
a3_wt, d3_wt=haar_dwt(a2_wt)

fig, axes = plt.subplots(5, 1, figsize=(10, 10))
axes[0].plot(t_wt*1e3, sig_wt, "b-", linewidth=0.8)
axes[0].set_ylabel("Amplitude"); axes[0].set_title("Original Signal (chirp 100–3000 Hz + transient)")
axes[0].set_xlim(0, t_wt[-1]*1e3); axes[0].grid(True, alpha=0.3)
t_d1=np.linspace(0, t_wt[-1]*1e3, len(d1_wt))
axes[1].plot(t_d1, d1_wt, "r-", linewidth=0.8)
axes[1].set_ylabel("d₁"); axes[1].set_title("Level 1 Detail: 2–4 kHz (512 coeff)"); axes[1].grid(True, alpha=0.3)
t_d2=np.linspace(0, t_wt[-1]*1e3, len(d2_wt))
axes[2].plot(t_d2, d2_wt, "g-", linewidth=0.8)
axes[2].set_ylabel("d₂"); axes[2].set_title("Level 2 Detail: 1–2 kHz (256 coeff)"); axes[2].grid(True, alpha=0.3)
t_d3=np.linspace(0, t_wt[-1]*1e3, len(d3_wt))
axes[3].plot(t_d3, d3_wt, "m-", linewidth=0.8)
axes[3].set_ylabel("d₃"); axes[3].set_title("Level 3 Detail: 0.5–1 kHz (128 coeff)"); axes[3].grid(True, alpha=0.3)
t_a3=np.linspace(0, t_wt[-1]*1e3, len(a3_wt))
axes[4].plot(t_a3, a3_wt, "k-", linewidth=0.8)
axes[4].set_ylabel("a₃"); axes[4].set_title("Level 3 Approximation: 0–0.5 kHz (128 coeff)")
axes[4].set_xlabel("Time (ms)"); axes[4].grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch08_wavelet.png")


# ============================================================
# Chapter 9: Electromagnetics
# ============================================================
print("Chapter 9: Electromagnetics")

# --- 9.2.5 B-H Hysteresis ---
B_sat_h=1.8; H_c_h=50; B_r_h=1.2
k_h=np.tan(B_r_h/B_sat_h*np.pi/2)/H_c_h; H_max_h=500
H_up=np.linspace(-H_max_h, H_max_h, 1000)
H_dn=np.linspace(H_max_h, -H_max_h, 1000)
B_up=B_sat_h*(2/np.pi)*np.arctan(k_h*(H_up+H_c_h))
B_dn=B_sat_h*(2/np.pi)*np.arctan(k_h*(H_dn-H_c_h))

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(H_up, B_up, "b-", linewidth=2, label="Magnetizing (H increasing)")
ax.plot(H_dn, B_dn, "r-", linewidth=2, label="Demagnetizing (H decreasing)")
idx_rem=np.argmin(np.abs(H_dn)); idx_coer=np.argmin(np.abs(B_dn))
ax.plot(0, B_dn[idx_rem], "go", markersize=10, zorder=5)
ax.annotate(f"B_r = {B_dn[idx_rem]:.2f} T\n(Remanence)", xy=(0, B_dn[idx_rem]),
            xytext=(80, B_dn[idx_rem]+0.1), fontsize=10, color="green",
            arrowprops=dict(arrowstyle="->", color="green"))
ax.plot(H_dn[idx_coer], 0, "mo", markersize=10, zorder=5)
ax.annotate(f"H_c = {abs(H_dn[idx_coer]):.0f} A/m\n(Coercivity)", xy=(H_dn[idx_coer], 0),
            xytext=(H_dn[idx_coer]-180, -0.5), fontsize=10, color="purple",
            arrowprops=dict(arrowstyle="->", color="purple"))
ax.axhline(y=0, color="gray", linewidth=0.5); ax.axvline(x=0, color="gray", linewidth=0.5)
ax.set_xlabel("Magnetic Field Intensity H (A/m)"); ax.set_ylabel("Flux Density B (T)")
ax.set_title("B-H Hysteresis Loop — Soft Magnetic Material (Silicon Steel)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_xlim(-H_max_h, H_max_h); ax.set_ylim(-2.0, 2.0)
fig.tight_layout()
save(fig, "ch09_hysteresis.png")

# --- 9.4.4 Skin Effect ---
mu0_s=4*np.pi*1e-7; sig_cu=5.8e7; sig_al_s=3.5e7; sig_st=1.0e7; mur_st=200
f_sk=np.logspace(0, 11, 1000)
d_cu=1/np.sqrt(np.pi*f_sk*mu0_s*sig_cu)
d_al=1/np.sqrt(np.pi*f_sk*mu0_s*sig_al_s)
d_st=1/np.sqrt(np.pi*f_sk*mu0_s*mur_st*sig_st)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(f_sk, d_cu*1e3, "b-", linewidth=2, label="Copper (σ = 5.8×10⁷ S/m)")
ax.loglog(f_sk, d_al*1e3, "r-", linewidth=2, label="Aluminum (σ = 3.5×10⁷ S/m)")
ax.loglog(f_sk, d_st*1e3, "g-", linewidth=2, label="Mild Steel (σ = 10⁷, μᵣ = 200)")
for fp, lab in [(60, "60 Hz"), (1e6, "1 MHz"), (10e9, "10 GHz")]:
    dp=1/np.sqrt(np.pi*fp*mu0_s*sig_cu)
    ax.plot(fp, dp*1e3, "ko", markersize=7, zorder=5)
    txt=f"{dp*1e3:.1f} mm" if dp>1e-3 else f"{dp*1e6:.1f} μm"
    ax.annotate(f"{lab}\nδ = {txt}", xy=(fp, dp*1e3), xytext=(fp*3, dp*1e3*2),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black"))
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Skin Depth (mm)")
ax.set_title("Skin Depth vs Frequency for Common Conductors")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(1, 1e11); ax.set_ylim(1e-4, 100)
fig.tight_layout()
save(fig, "ch09_skin_depth.png")

# --- 9.5.6 TDR Transients ---
Vs_t=1.0; Z0_t=50; Zs_t=50; ZL_t=150; v_p=2e8; ln_t=3
td_t=ln_t/v_p; V_inc=Vs_t*Z0_t/(Zs_t+Z0_t)
GL=(ZL_t-Z0_t)/(ZL_t+Z0_t); V_ref=GL*V_inc
t_max_t=80e-9; t_a=np.linspace(0, t_max_t, 1000)
V_src=np.zeros_like(t_a); V_ld=np.zeros_like(t_a)
for i, t in enumerate(t_a):
    vs=0; vl=0
    if t>=0: vs+=V_inc
    if t>=2*td_t: vs+=V_ref
    if t>=td_t: vl+=V_inc+V_ref
    V_src[i]=vs; V_ld[i]=vl

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
# Lattice diagram
ax1.set_xlim(0, 3); ax1.set_ylim(0, 70); ax1.invert_yaxis()
ax1.axvline(x=0, color="black", linewidth=2); ax1.axvline(x=3, color="black", linewidth=2)
ax1.text(0, -3, "Source\n(50 Ω)", ha="center", fontsize=10, fontweight="bold")
ax1.text(3, -3, "Load\n(150 Ω)", ha="center", fontsize=10, fontweight="bold")
ax1.annotate("", xy=(3, 15), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
ax1.text(1.5, 4, f"V⁺ = {V_inc:.2f} V", fontsize=11, color="blue", ha="center", rotation=-15, fontweight="bold")
ax1.annotate("", xy=(0, 30), xytext=(3, 15), arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
ax1.text(1.5, 19, f"V⁻ = Γ_L × V⁺ = {V_ref:.2f} V", fontsize=11, color="red", ha="center", rotation=15, fontweight="bold")
ax1.text(0.1, 33, "Γ_S = 0 (absorbed)", fontsize=9, color="gray", style="italic")
for tn in [0, 15, 30]:
    ax1.axhline(y=tn, color="gray", linestyle=":", alpha=0.3)
    ax1.text(-0.2, tn, f"{tn} ns", fontsize=9, ha="right", va="center")
ax1.set_xlabel("Position along line (m)"); ax1.set_ylabel("Time (ns)")
ax1.set_title("Lattice (Bounce) Diagram: 50 Ω Source → 50 Ω Line → 150 Ω Load"); ax1.grid(False)

# Voltage vs time
ax2.step(t_a*1e9, V_src, "b-", linewidth=2, where="post", label="V at source end")
ax2.step(t_a*1e9, V_ld, "r-", linewidth=2, where="post", label="V at load end")
ax2.axhline(y=0.75, color="green", linestyle="--", alpha=0.6, label="Steady state = 0.75 V")
ax2.annotate(f"V⁺ = {V_inc:.2f} V", xy=(1, V_inc), xytext=(8, V_inc+0.1),
             fontsize=10, color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
ax2.annotate(f"V_L = {V_inc+V_ref:.2f} V", xy=(td_t*1e9+1, V_inc+V_ref),
             xytext=(td_t*1e9+10, V_inc+V_ref+0.1),
             fontsize=10, color="red", arrowprops=dict(arrowstyle="->", color="red"))
ax2.set_xlabel("Time (ns)"); ax2.set_ylabel("Voltage (V)")
ax2.set_title("Voltage vs Time at Source and Load Ends")
ax2.set_xlim(0, t_max_t*1e9); ax2.set_ylim(-0.1, 1.0); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch09_tdr.png")

# --- 9.7.1 Shielding Effectiveness ---
sig_al_se=3.5e7; sig_cu_ref=5.8e7; sig_r_se=sig_al_se/sig_cu_ref
mu_r_se=1; t_se=1.5e-3; mu0_se=4*np.pi*1e-7
f_se=np.logspace(3, 10, 1000)
d_al_se=1/np.sqrt(np.pi*f_se*mu0_se*sig_al_se)
A_loss=8.686*(t_se/d_al_se)
R_loss=168-10*np.log10(f_se*mu_r_se/sig_r_se)
SE_tot=A_loss+R_loss

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(f_se, A_loss, "b--", linewidth=1.5, label="Absorption loss A")
ax.semilogx(f_se, R_loss, "r--", linewidth=1.5, label="Reflection loss R")
ax.semilogx(f_se, SE_tot, "k-", linewidth=2.5, label="Total SE = A + R")
f_m=1e8; d_m=1/np.sqrt(np.pi*f_m*mu0_se*sig_al_se)
A_m=8.686*(t_se/d_m); R_m=168-10*np.log10(f_m*mu_r_se/sig_r_se)
ax.plot(f_m, A_m+R_m, "go", markersize=10, zorder=5)
ax.annotate(f"100 MHz: SE = {A_m+R_m:.0f} dB\n(A = {A_m:.0f}, R = {R_m:.0f})",
            xy=(f_m, A_m+R_m), xytext=(f_m/10, A_m+R_m+100), fontsize=10, color="green",
            arrowprops=dict(arrowstyle="->", color="green"))
ax.axhline(y=60, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Target: 60 dB")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Shielding Effectiveness (dB)")
ax.set_title("Shielding Effectiveness — 1.5 mm Aluminum Enclosure")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(1e3, 1e10); ax.set_ylim(0, 2000)
fig.tight_layout()
save(fig, "ch09_shielding.png")

# --- 9.5.5 Microstrip Characteristic Impedance vs W/h ---
def _ms_Z0_g(wh_ratio, er):
    """Hammerstad approximate microstrip Z0 (Ω) and effective εᵣ."""
    wh = wh_ratio
    eta0 = 377.0
    if wh < 1:
        F = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / wh) ** 0.7528)
        Z = (eta0 / (2 * np.pi)) * np.log(F / wh + np.sqrt(1 + (2 / wh) ** 2))
        e = (er + 1) / 2 + (er - 1) / 2 * (1 / np.sqrt(1 + 12 / wh) + 0.04 * (1 - wh) ** 2)
    else:
        e = (er + 1) / 2 + (er - 1) / 2 / np.sqrt(1 + 12 / wh)
        Z = (eta0 / (2 * np.pi * np.sqrt(e))) / (wh + 1.393 + 0.667 * np.log(wh + 1.444))
    return Z, e

_wh_arr_g = np.linspace(0.05, 6.0, 800)
_substrates_g = [
    ("FR-4 (εᵣ = 4.3)",           4.3,  "blue"),
    ("Rogers RO4003 (εᵣ = 3.55)", 3.55, "red"),
    ("PTFE (εᵣ = 2.2)",           2.2,  "green"),
]

fig, (_ax5a_g, _ax5b_g) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
for _lbl_g, _er_g, _c_g in _substrates_g:
    _Z0v_g  = np.array([_ms_Z0_g(_w, _er_g)[0] for _w in _wh_arr_g])
    _eefv_g = np.array([_ms_Z0_g(_w, _er_g)[1] for _w in _wh_arr_g])
    _ax5a_g.plot(_wh_arr_g, _Z0v_g,  color=_c_g, linewidth=2, label=_lbl_g)
    _ax5b_g.plot(_wh_arr_g, _eefv_g, color=_c_g, linewidth=2, label=_lbl_g)

_ax5a_g.axhline(y=50, color="gray", linestyle="--", alpha=0.7, linewidth=1.5, label="Z₀ = 50 Ω target")
_ax5a_g.axhline(y=75, color="gray", linestyle=":",  alpha=0.5, linewidth=1.5, label="Z₀ = 75 Ω target")
_wh_ex_g = 1.81
_Z0_ex_g, _eeff_ex_g = _ms_Z0_g(_wh_ex_g, 4.3)
_ax5a_g.plot(_wh_ex_g, _Z0_ex_g,   "ko", markersize=9, zorder=6)
_ax5a_g.annotate(
    f"FR-4: W/h = {_wh_ex_g}, Z₀ = {_Z0_ex_g:.1f} Ω",
    xy=(_wh_ex_g, _Z0_ex_g), xytext=(_wh_ex_g + 0.8, _Z0_ex_g + 20),
    fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
)
_ax5b_g.plot(_wh_ex_g, _eeff_ex_g, "ko", markersize=9, zorder=6)
_ax5b_g.annotate(
    f"εeff = {_eeff_ex_g:.2f}",
    xy=(_wh_ex_g, _eeff_ex_g), xytext=(_wh_ex_g + 0.8, _eeff_ex_g - 0.25),
    fontsize=9, arrowprops=dict(arrowstyle="->", color="black"),
)
_ax5a_g.set_ylabel("Characteristic Impedance Z₀ (Ω)", fontsize=10)
_ax5a_g.set_title("Microstrip Characteristic Impedance vs Trace Width Ratio (W/h)", fontsize=11)
_ax5a_g.legend(fontsize=9); _ax5a_g.grid(True, alpha=0.3); _ax5a_g.set_ylim(0, 200)
_ax5b_g.set_xlabel("W/h  (Trace Width / Substrate Height)", fontsize=10)
_ax5b_g.set_ylabel("Effective Dielectric Constant εeff", fontsize=10)
_ax5b_g.set_title("Effective Dielectric Constant vs W/h", fontsize=11)
_ax5b_g.legend(fontsize=9); _ax5b_g.grid(True, alpha=0.3); _ax5b_g.set_xlim(0.05, 6)
fig.tight_layout()
save(fig, "ch09_microstrip_Z0.png")


# ============================================================
# Chapter 10: Power Electronics
# ============================================================
print("Chapter 10: Power Electronics")

# --- 10.2.1 Rectifier ---
f_ac=60; Vrms=120; Vpeak=Vrms*np.sqrt(2); Vd=0.7; Vpeak_rect=Vpeak-2*Vd
t_rect=np.linspace(0, 3/f_ac, 2000)
v_ac=Vpeak*np.sin(2*np.pi*f_ac*t_rect)
v_rectified=np.maximum(np.abs(Vpeak*np.sin(2*np.pi*f_ac*t_rect))-2*Vd, 0)
C_f=1000e-6; R_l=100; tau_f=R_l*C_f
v_filt=np.zeros_like(t_rect); v_filt[0]=Vpeak_rect; dt=t_rect[1]-t_rect[0]
for i in range(1, len(t_rect)):
    if v_rectified[i]>v_filt[i-1]: v_filt[i]=v_rectified[i]
    else: v_filt[i]=v_filt[i-1]*np.exp(-dt/tau_f)
ripple=np.max(v_filt[len(t_rect)//3:])-np.min(v_filt[len(t_rect)//3:])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_rect*1e3, v_ac, "b-", linewidth=1, alpha=0.5, label="AC Input")
ax1.plot(t_rect*1e3, v_rectified, "r-", linewidth=1.5, label="Full-Wave Rectified")
ax1.set_ylabel("Voltage (V)"); ax1.set_title("Full-Wave Bridge Rectifier"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(t_rect*1e3, v_rectified, "r-", linewidth=1, alpha=0.3, label="Rectified (no filter)")
ax2.plot(t_rect*1e3, v_filt, "b-", linewidth=2, label="Filtered (C = 1000 μF)")
ax2.axhline(y=np.mean(v_filt[len(t_rect)//3:]), color="green", linestyle="--", alpha=0.6,
            label=f"Avg DC ≈ {np.mean(v_filt[len(t_rect)//3:]):.1f} V")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Voltage (V)")
ax2.set_title("Capacitor-Filtered Output (R_load = 100 Ω)"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_rectifier.png")

# --- 10.3.1 Buck ---
Vin_b=48; Vout_b=12; Iout_b=2.5; L_b=100e-6; fsw_b=100e3
D_b=Vout_b/Vin_b; T_b=1/fsw_b; dI_b=(Vin_b-Vout_b)*D_b*T_b/L_b
t_b=np.linspace(0, 3*T_b, 3000); i_b=np.zeros_like(t_b); pwm_b=np.zeros_like(t_b)
for i, tv in enumerate(t_b):
    tc=tv%T_b
    if tc<D_b*T_b: i_b[i]=Iout_b-dI_b/2+(Vin_b-Vout_b)/L_b*tc; pwm_b[i]=1
    else: i_b[i]=Iout_b+dI_b/2-Vout_b/L_b*(tc-D_b*T_b); pwm_b[i]=0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 3]})
ax1.fill_between(t_b*1e6, pwm_b*Vin_b, step="post", alpha=0.3, color="orange")
ax1.step(t_b*1e6, pwm_b*Vin_b, "orange", linewidth=1.5, where="post", label="Switch Node")
ax1.set_ylabel("V_sw (V)"); ax1.set_title(f"Buck Converter (Vin={Vin_b}V → Vout={Vout_b}V, D={D_b:.2f})"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(t_b*1e6, i_b, "b-", linewidth=2, label=f"I_L(t), ΔI = {dI_b:.2f} A ({dI_b/Iout_b*100:.0f}% ripple)")
ax2.axhline(y=Iout_b, color="red", linestyle="--", alpha=0.6, label=f"I_avg = {Iout_b} A")
ax2.set_xlabel("Time (μs)"); ax2.set_ylabel("Inductor Current (A)"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_buck.png")

# --- 10.3.2 Boost ---
Vin_bo=12; Vout_bo=48; Iout_bo=0.5; L_bo=220e-6; fsw_bo=150e3
D_bo=1-Vin_bo/Vout_bo; T_bo=1/fsw_bo; Iin_bo=Iout_bo/(1-D_bo)
dI_bo=Vin_bo*D_bo*T_bo/L_bo
t_bo=np.linspace(0, 3*T_bo, 3000); i_bo=np.zeros_like(t_bo)
for i, tv in enumerate(t_bo):
    tc=tv%T_bo
    if tc<D_bo*T_bo: i_bo[i]=Iin_bo-dI_bo/2+Vin_bo/L_bo*tc
    else: i_bo[i]=Iin_bo+dI_bo/2-(Vout_bo-Vin_bo)/L_bo*(tc-D_bo*T_bo)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_bo*1e6, i_bo, "b-", linewidth=2, label=f"I_L(t), ΔI = {dI_bo:.2f} A")
ax.axhline(y=Iin_bo, color="red", linestyle="--", alpha=0.6, label=f"I_avg = {Iin_bo:.1f} A")
ax.set_xlabel("Time (μs)"); ax.set_ylabel("Inductor Current (A)")
ax.set_title(f"Boost Converter (Vin={Vin_bo}V → Vout={Vout_bo}V, D={D_bo:.2f})"); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_boost.png")

# --- 10.4.1 SPWM ---
f_ref=60; f_car=5000; ma=0.8; Vdc=400
t_sp=np.linspace(0, 1/f_ref, 10000)
v_ref=ma*np.sin(2*np.pi*f_ref*t_sp)
v_car=2*np.abs(2*(t_sp*f_car-np.floor(t_sp*f_car+0.5)))-1
v_pwm=np.where(v_ref>v_car, Vdc/2, -Vdc/2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_sp*1e3, v_ref, "b-", linewidth=2, label=f"Reference (60 Hz, m_a={ma})")
ax1.plot(t_sp*1e3, v_car, "gray", linewidth=0.5, alpha=0.7, label="Carrier (5 kHz)")
ax1.set_ylabel("Normalized Amplitude"); ax1.set_title("SPWM: Reference vs Carrier"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.fill_between(t_sp*1e3, v_pwm, step="mid", alpha=0.3, color="orange")
ax2.step(t_sp*1e3, v_pwm, "orange", linewidth=0.5, where="mid")
V1p=ma*Vdc/2
ax2.plot(t_sp*1e3, V1p*np.sin(2*np.pi*f_ref*t_sp), "b--", linewidth=2, label=f"Fundamental: V₁ = {V1p:.0f} V peak")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Output Voltage (V)")
ax2.set_title("Inverter Output"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_spwm.png")

# --- 10.6.1 Power Losses ---
Rds=0.05; Id=10; P_c=Id**2*Rds; Vds=48; tr=30e-9; tf=50e-9
fsw_r=np.linspace(10e3, 500e3, 500)
P_sw=0.5*Vds*Id*(tr+tf)*fsw_r; P_tot=P_c+P_sw
ci=np.argmin(np.abs(P_sw-P_c))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(fsw_r/1e3, np.full_like(fsw_r, P_c), "b--", linewidth=2, label=f"Conduction: {P_c:.1f} W")
ax.plot(fsw_r/1e3, P_sw, "r--", linewidth=2, label="Switching: P_sw ∝ f_sw")
ax.plot(fsw_r/1e3, P_tot, "k-", linewidth=2.5, label="Total Losses")
ax.plot(fsw_r[ci]/1e3, P_tot[ci], "go", markersize=10, label=f"Equal at {fsw_r[ci]/1e3:.0f} kHz")
ax.set_xlabel("Switching Frequency (kHz)"); ax.set_ylabel("Power Loss (W)")
ax.set_title(f"MOSFET Losses (R_DS(on)={Rds*1e3:.0f} mΩ, I_D={Id} A)"); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_power_losses.png")

# --- 10.2.3 Rectifier Harmonics (6-pulse vs 12-pulse) ---
harmonics_6 = [5, 7, 11, 13, 17, 19, 23, 25]
I1 = 100  # fundamental current per bridge
I_6pulse = [I1/h for h in harmonics_6]
# 12-pulse: 5th & 7th cancel, 11th & 13th add (from 2 bridges), etc.
I_12pulse = []
for h in harmonics_6:
    if h in (5, 7, 17, 19):  # cancelled by 30° phase shift
        I_12pulse.append(0)
    else:  # 11, 13, 23, 25 add in phase from 2 bridges, normalize to I1_total=200A
        I_12pulse.append(2 * I1/h / (2 * I1) * 100)  # percent of combined fundamental
I_6pulse_pct = [I1/h / I1 * 100 for h in harmonics_6]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(harmonics_6))
w = 0.35
ax.bar(x - w/2, I_6pulse_pct, w, color="tab:red", alpha=0.8, label="6-pulse")
ax.bar(x + w/2, I_12pulse, w, color="tab:blue", alpha=0.8, label="12-pulse")
ax.set_xticks(x)
ax.set_xticklabels([f"{h}th" for h in harmonics_6])
ax.set_xlabel("Harmonic Order"); ax.set_ylabel("Current (% of fundamental)")
ax.set_title("Rectifier Harmonic Spectrum: 6-Pulse vs 12-Pulse")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(I_6pulse_pct):
    if v > 1:
        ax.text(i - w/2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=8)
fig.tight_layout()
save(fig, "ch10_harmonics.png")

# --- 10.4.3 Multilevel Inverter (2-level vs 3-level) ---
Vdc_ml = 800; f_fund = 60; f_car_ml = 5000; ma_ml = 0.85
t_ml = np.linspace(0, 1/f_fund, 8000)
v_ref_ml = ma_ml * np.sin(2 * np.pi * f_fund * t_ml)
# Carrier: triangle wave
v_car_ml = 2 * np.abs(2 * (t_ml * f_car_ml - np.floor(t_ml * f_car_ml + 0.5))) - 1

# 2-level: +Vdc/2 or -Vdc/2
v_2level = np.where(v_ref_ml > v_car_ml, Vdc_ml/2, -Vdc_ml/2)

# 3-level NPC: two carriers (upper 0..1, lower -1..0)
v_3level = np.zeros_like(t_ml)
for i in range(len(t_ml)):
    ref = v_ref_ml[i]
    car = v_car_ml[i]
    if ref >= 0:
        v_3level[i] = Vdc_ml/2 if ref > car else 0
    else:
        v_3level[i] = -Vdc_ml/2 if ref < car else 0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.step(t_ml*1e3, v_2level, "tab:red", linewidth=0.5, where="mid", alpha=0.7)
ax1.plot(t_ml*1e3, ma_ml*Vdc_ml/2*np.sin(2*np.pi*f_fund*t_ml), "b--", linewidth=2, label=f"Fundamental ({ma_ml*Vdc_ml/2:.0f} V peak)")
ax1.set_ylabel("Voltage (V)"); ax1.set_title("Two-Level Inverter Output (±400 V steps)")
ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.step(t_ml*1e3, v_3level, "tab:green", linewidth=0.5, where="mid", alpha=0.7)
ax2.plot(t_ml*1e3, ma_ml*Vdc_ml/2*np.sin(2*np.pi*f_fund*t_ml), "b--", linewidth=2, label=f"Fundamental ({ma_ml*Vdc_ml/2:.0f} V peak)")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Voltage (V)")
ax2.set_title("Three-Level NPC Inverter Output (0/±400 V steps)"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_multilevel.png")

# --- 10.7.1 Active PFC Current Shaping ---
f_line = 50; Vrms_pfc = 230; Vpk = Vrms_pfc * np.sqrt(2)
Vout_pfc = 400; P_pfc = 500
t_pfc = np.linspace(0, 2/f_line, 2000)
v_in = Vpk * np.sin(2 * np.pi * f_line * t_pfc)
# Without PFC: capacitor-input current (narrow pulses near peak)
theta_cond = np.radians(30)  # approx conduction angle
i_nopfc = np.zeros_like(t_pfc)
for k in range(4):  # 4 half-cycles in 2 line periods
    t_peak = (k * 0.5 + 0.25) / f_line
    mask = np.abs(t_pfc - t_peak) < theta_cond / (2 * np.pi * f_line)
    window = np.cos(2 * np.pi * f_line * (t_pfc[mask] - t_peak)) - np.cos(theta_cond)
    i_nopfc[mask] = window / np.max(window) * 12  # peak ~12 A

# With PFC: sinusoidal current in phase with voltage
Ipk = P_pfc / Vrms_pfc * np.sqrt(2)
i_pfc = Ipk * np.abs(np.sin(2 * np.pi * f_line * t_pfc))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_pfc*1e3, np.abs(v_in)/Vpk*5, "gray", linewidth=1, alpha=0.5, label="Voltage (scaled)")
ax1.plot(t_pfc*1e3, i_nopfc, "r-", linewidth=1.5, label="Input current (no PFC)")
ax1.set_ylabel("Current (A)"); ax1.set_title("Without PFC: Peaky Capacitor-Input Current")
ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(t_pfc*1e3, np.abs(v_in)/Vpk*Ipk, "gray", linewidth=1, alpha=0.5, label="Voltage (scaled)")
ax2.plot(t_pfc*1e3, i_pfc, "g-", linewidth=1.5, label=f"Input current (PFC), I_peak = {Ipk:.1f} A")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("Current (A)")
ax2.set_title("With Active PFC: Sinusoidal Input Current (PF ≈ 1.0)"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_pfc.png")

# --- 10.8.1 Battery Pack Voltage vs SOC ---
soc_pct = np.linspace(0, 100, 200)
soc_frac = soc_pct / 100.0
ocv_cell = 3.0 + 1.2 * soc_frac - 0.35 * (1 - soc_frac) * np.exp(-20 * soc_frac) \
           + 0.05 * np.log(soc_frac + 0.001) + 0.15 * soc_frac**2
n_cells = 96; R_pack = 0.192; I_load = 150
ocv_pack = ocv_cell * n_cells
v_loaded = ocv_pack - I_load * R_pack

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(soc_pct, ocv_pack, "b-", linewidth=2, label="Open-Circuit Voltage (no load)")
ax.plot(soc_pct, v_loaded, "r--", linewidth=2, label=f"Terminal Voltage (150 A load, IR = {I_load * R_pack:.1f} V)")
idx_50 = 100
ax.annotate(f"Nominal: {ocv_pack[idx_50]:.0f} V",
            xy=(50, ocv_pack[idx_50]), xytext=(60, ocv_pack[idx_50] + 15),
            fontsize=10, color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
ax.annotate(f"Loaded: {v_loaded[idx_50]:.0f} V",
            xy=(50, v_loaded[idx_50]), xytext=(60, v_loaded[idx_50] - 15),
            fontsize=10, color="red", arrowprops=dict(arrowstyle="->", color="red"))
ax.fill_between(soc_pct, v_loaded, ocv_pack, alpha=0.1, color="orange", label="IR voltage drop")
ax.set_xlabel("State of Charge (%)"); ax.set_ylabel("Pack Voltage (V)")
ax.set_title("96s NMC Battery Pack: OCV and Loaded Voltage vs SOC")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 100)
fig.tight_layout()
save(fig, "ch10_bms_discharge.png")

# --- 10.8.2 Cell Balancing Convergence ---
# Visualization uses compressed time scale to show convergence behavior
R_bleed = 33
v_init = np.array([3.95, 3.97, 3.99, 4.01, 4.03, 4.05, 4.08, 4.10, 4.12, 4.14, 4.16, 4.18])
v_target = v_init.min()
t_min = np.linspace(0, 30, 500)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.coolwarm(np.linspace(0, 1, len(v_init)))
for idx, v0 in enumerate(v_init):
    if v0 > v_target:
        # Use a visualization-friendly time constant (~10 min for largest delta)
        tau = 10.0 * (v0 - v_target) / (v_init.max() - v_target)
        tau = max(tau, 3)
        v_t = v_target + (v0 - v_target) * np.exp(-t_min / tau)
    else:
        v_t = np.full_like(t_min, v0)
    ax.plot(t_min, v_t, linewidth=1.5, color=colors[idx],
            label=f"Cell {idx+1}: {v0:.2f} V" if idx % 3 == 0 or idx == 11 else None)
ax.axhline(y=v_target, color="green", linestyle="--", alpha=0.6, linewidth=1.5, label=f"Target: {v_target:.2f} V")
ax.set_xlabel("Balancing Time (minutes)"); ax.set_ylabel("Cell Voltage (V)")
ax.set_title(f"Passive Cell Balancing: 12 Cells with {R_bleed} \u03a9 Bleed Resistors")
ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3); ax.set_ylim(3.93, 4.20)
fig.tight_layout()
save(fig, "ch10_bms_balancing.png")

# --- 10.8.3 SOC Estimation with Drift ---
# Exaggerate drift for visualization (real ±0.5% is too subtle at full scale)
Q_full = 100; soc_init_bms = 85; sensor_error = 0.005
q_discharged = np.linspace(0, 80, 300)
soc_true = soc_init_bms - (q_discharged / Q_full) * 100
drift_ah = sensor_error * q_discharged
drift_soc = (drift_ah / Q_full) * 100
# Exaggerate by 10× for visual clarity; label shows true values
drift_viz = drift_soc * 10
soc_estimated = soc_true + drift_viz * 0.7  # biased high

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(q_discharged, soc_true, "b-", linewidth=2, label="True SOC")
ax.plot(q_discharged, soc_estimated, "r--", linewidth=2, label="Estimated SOC (drift exaggerated 10\u00d7)")
ax.fill_between(q_discharged, soc_true - drift_viz, soc_true + drift_viz,
                alpha=0.15, color="red", label="Uncertainty band (exaggerated 10\u00d7)")
idx_60 = np.argmin(np.abs(q_discharged - 60))
ax.annotate(f"At 60 Ah: true error = \u00b1{drift_soc[idx_60]:.2f}%\n(shown 10\u00d7 for visibility)",
            xy=(60, soc_true[idx_60]), xytext=(30, soc_true[idx_60] + 12),
            fontsize=10, color="darkred", arrowprops=dict(arrowstyle="->", color="darkred"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.set_xlabel("Charge Discharged (Ah)"); ax.set_ylabel("State of Charge (%)")
ax.set_title("Coulomb Counting SOC Estimation with \u00b10.5% Current Sensor Drift")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 80); ax.set_ylim(0, 90)
fig.tight_layout()
save(fig, "ch10_bms_soc.png")

# --- 10.9.3 Frequency Regulation Droop Response ---
P_rated_bess = 20  # MW
R_droop = 0.04
f_nom = 60.0
deadband = 0.036  # Hz

freq = np.linspace(59.5, 60.5, 500)
delta_f = freq - f_nom
delta_p = np.zeros_like(delta_f)
for i, df in enumerate(delta_f):
    if abs(df) > deadband:
        df_eff = abs(df) - deadband
        power = P_rated_bess * df_eff / (f_nom * R_droop)
        power = min(power, P_rated_bess)
        delta_p[i] = -np.sign(df) * power  # negative df -> positive power (discharge)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freq, delta_p, "b-", linewidth=2)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.axvline(x=f_nom, color="gray", linestyle="--", alpha=0.5, label="Nominal 60 Hz")
ax.axvspan(f_nom - deadband, f_nom + deadband, alpha=0.1, color="green", label=f"Deadband ±{deadband} Hz")
# Mark example points from 10.9.3
ax.plot(59.85, 0.95, "ro", markersize=10, zorder=5)
ax.annotate("59.85 Hz → 0.95 MW\n(discharge)", xy=(59.85, 0.95), xytext=(59.6, 4),
            fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.plot(60.10, -0.533, "bs", markersize=10, zorder=5)
ax.annotate("60.10 Hz → −0.533 MW\n(charge)", xy=(60.10, -0.533), xytext=(60.2, -4),
            fontsize=9, color="blue", arrowprops=dict(arrowstyle="->", color="blue"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.fill_between(freq, 0, delta_p, where=(delta_p > 0), alpha=0.15, color="red", label="Discharge (support)")
ax.fill_between(freq, 0, delta_p, where=(delta_p < 0), alpha=0.15, color="blue", label="Charge (absorb)")
ax.set_xlabel("Grid Frequency (Hz)"); ax.set_ylabel("BESS Power Output (MW)")
ax.set_title(f"BESS Frequency Regulation: {P_rated_bess} MW, {R_droop*100:.0f}% Droop, ±{deadband} Hz Deadband")
ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)
ax.set_xlim(59.5, 60.5); ax.set_ylim(-12, 12)
fig.tight_layout()
save(fig, "ch10_bess_droop.png")

# --- 10.9.4 Peak Shaving Load Profile ---
hours = np.linspace(0, 24, 288)
# Synthetic commercial load profile
base_load = 2.0 + 0.5 * np.sin(2 * np.pi * (hours - 6) / 24)
peak_bump = 2.5 * np.exp(-0.5 * ((hours - 14) / 1.5)**2)
load_profile = base_load + peak_bump
load_profile = np.clip(load_profile, 1.5, 5.0)

threshold = 3.5
shaved_load = np.minimum(load_profile, threshold)
bess_discharge = load_profile - shaved_load

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hours, load_profile, "b-", linewidth=2, label="Original Load Profile")
ax.plot(hours, shaved_load, "g-", linewidth=2, label=f"Load with BESS (capped at {threshold} MW)")
ax.fill_between(hours, shaved_load, load_profile, where=(load_profile > threshold),
                alpha=0.3, color="red", label="BESS Discharge (peak shaved)")
ax.axhline(y=threshold, color="green", linestyle="--", alpha=0.6, linewidth=1.5)
ax.axhline(y=5.0, color="red", linestyle=":", alpha=0.5)
ax.annotate("Peak: 5.0 MW", xy=(14, 5.0), xytext=(16, 4.7),
            fontsize=10, color="red", arrowprops=dict(arrowstyle="->", color="red"))
ax.annotate(f"Target: {threshold} MW", xy=(8, threshold), xytext=(3, 4.0),
            fontsize=10, color="green", arrowprops=dict(arrowstyle="->", color="green"))
# Annotate savings
ax.annotate("Demand charge\nsavings: $27k/mo", xy=(14, 4.2), xytext=(18, 4.5),
            fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.set_xlabel("Hour of Day"); ax.set_ylabel("Demand (MW)")
ax.set_title("Commercial Peak Shaving: 1.5 MW / 3.0 MWh BESS")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 24); ax.set_ylim(0, 5.5)
fig.tight_layout()
save(fig, "ch10_bess_peak_shaving.png")

# --- 10.9.5 Capacity Degradation and LCOS ---
years_bess = np.arange(0, 16)
degradation_rate = 0.025
capacity = 100 * (1 - degradation_rate * years_bess)
annual_discharge = capacity * 0.80 * 365 / 1000  # GWh

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Capacity fade
ax1.bar(years_bess, capacity, color="steelblue", alpha=0.7, edgecolor="navy", linewidth=0.5)
ax1.axhline(y=81.25, color="orange", linestyle="--", linewidth=1.5, label="15-yr average: 81.25 MWh")
ax1.axhline(y=62.5, color="red", linestyle=":", linewidth=1.5, label="End-of-life: 62.5 MWh (37.5% fade)")
ax1.set_xlabel("Year"); ax1.set_ylabel("Usable Capacity (MWh)")
ax1.set_title("100 MWh LFP BESS: 2.5%/yr Capacity Fade")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3, axis="y")
ax1.set_xlim(-0.5, 15.5); ax1.set_ylim(0, 110)

# Right: Cumulative energy and LCOS components
capital = 28_000_000
om_annual = np.array([cap * 0.80 * 365 * 6 for cap in capacity[1:]])  # $/year
cum_energy = np.cumsum([cap * 0.80 * 365 for cap in capacity[1:]])  # MWh
lcos_over_time = np.zeros(15)
pw_factor_8 = np.array([(1 - (1.08)**(-n)) / 0.08 if n > 0 else 0 for n in range(1, 16)])
for n in range(15):
    pv_om = sum(om_annual[:n+1] / (1.08)**(np.arange(1, n+2)))
    lcos_over_time[n] = (capital + pv_om) / cum_energy[n] if cum_energy[n] > 0 else 0

ax2.plot(years_bess[1:], lcos_over_time, "r-", linewidth=2, marker="o", markersize=5)
ax2.axhline(y=82.1, color="blue", linestyle="--", alpha=0.6, label="15-yr LCOS: $82.1/MWh")
ax2.axhline(y=140, color="green", linestyle=":", alpha=0.6, label="Revenue: $140/MWh")
ax2.fill_between(years_bess[1:], lcos_over_time, 140, where=(lcos_over_time < 140),
                 alpha=0.1, color="green", label="Profit margin")
ax2.set_xlabel("Project Year"); ax2.set_ylabel("LCOS ($/MWh)")
ax2.set_title("Levelized Cost of Storage Over Project Life")
ax2.legend(fontsize=9, loc="upper right"); ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.5, 15.5); ax2.set_ylim(0, 300)

fig.tight_layout()
save(fig, "ch10_bess_degradation.png")


# --- 10.10.1 CC/CV Charging Profile ---

dt_h = 0.001
# CC phase
t_cc_h = 1.6; I_cc_val = 2.5; V_start = 3.0; V_cutoff = 4.20
t_cc_arr = np.arange(0, t_cc_h, dt_h)
v_cc = V_start + (V_cutoff - V_start) * (t_cc_arr / t_cc_h) ** 0.7
i_cc_arr = np.full_like(t_cc_arr, I_cc_val)

# CV phase
tau_cv = 0.4; I_term = 0.25
t_cv_end = -tau_cv * np.log(I_term / I_cc_val)
t_cv_arr = np.arange(0, t_cv_end, dt_h)
i_cv_arr = I_cc_val * np.exp(-t_cv_arr / tau_cv)
v_cv = np.full_like(t_cv_arr, V_cutoff)

t_total = np.concatenate([t_cc_arr, t_cc_h + t_cv_arr])
v_total = np.concatenate([v_cc, v_cv])
i_total = np.concatenate([i_cc_arr, i_cv_arr])
q_total = np.cumsum(i_total * dt_h)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax1.plot(t_total, v_total, "b-", linewidth=2)
ax1.axhline(y=V_cutoff, color="red", linestyle="--", alpha=0.5, label=f"Cutoff {V_cutoff} V")
ax1.axvline(x=t_cc_h, color="gray", linestyle=":", alpha=0.7)
ax1.set_ylabel("Cell Voltage (V)"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
ax1.set_ylim(2.8, 4.4)
ax1.set_title("CC/CV Charging Profile — 5.0 Ah NMC 21700 Cell at 0.5C")
ax1.text(t_cc_h / 2, 3.1, "CC Phase", ha="center", fontsize=11, fontweight="bold", color="green")
ax1.text(t_cc_h + t_cv_end / 2, 3.1, "CV Phase", ha="center", fontsize=11, fontweight="bold", color="purple")

ax2.plot(t_total, i_total, "r-", linewidth=2)
ax2.axhline(y=I_term, color="orange", linestyle="--", alpha=0.5, label=f"Termination {I_term} A (C/20)")
ax2.axvline(x=t_cc_h, color="gray", linestyle=":", alpha=0.7)
ax2.set_ylabel("Charge Current (A)"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 3.0)

ax3.plot(t_total, q_total, "g-", linewidth=2)
ax3.axhline(y=5.0, color="gray", linestyle="--", alpha=0.5, label="Rated 5.0 Ah")
ax3.axvline(x=t_cc_h, color="gray", linestyle=":", alpha=0.7)
ax3.set_xlabel("Time (hours)"); ax3.set_ylabel("Charge Delivered (Ah)")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

fig.tight_layout()
save(fig, "ch10_cccv_charging.png")

# --- 10.10.4 Fast Charging Thermal ---

I_chg = 280; R_25C = 0.6e-3; R_5C = 1.5e-3
m_cell = 5.5; cp_cell = 1100; T_max_safe = 45
Q_25C = I_chg**2 * R_25C; Q_5C = I_chg**2 * R_5C
t_m = np.linspace(0, 60, 500); t_s = t_m * 60
T_from25 = 25 + Q_25C * t_s / (m_cell * cp_cell)
T_from5 = 5 + Q_5C * t_s / (m_cell * cp_cell)
t_lim = m_cell * cp_cell * (T_max_safe - 5) / Q_5C / 60

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(t_m, T_from25, "b-", linewidth=2, label="Start at 25°C (R = 0.6 mΩ)")
ax1.plot(t_m, T_from5, "r-", linewidth=2, label="Start at 5°C (R = 1.5 mΩ)")
ax1.axhline(y=T_max_safe, color="red", linestyle="--", alpha=0.6, label=f"Max safe {T_max_safe}°C")
ax1.axvline(x=t_lim, color="red", linestyle=":", alpha=0.5)
ax1.annotate(f"Limit at {t_lim:.1f} min", xy=(t_lim, T_max_safe), xytext=(t_lim + 5, 50),
             fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax1.fill_between(t_m, T_max_safe, 70, alpha=0.08, color="red", label="Plating risk zone")
ax1.set_xlabel("Time (minutes)"); ax1.set_ylabel("Cell Temperature (°C)")
ax1.set_title("Adiabatic Temperature Rise — 280 Ah LFP at 1C")
ax1.legend(fontsize=9, loc="upper left"); ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 60); ax1.set_ylim(0, 70)

bars = ax2.bar(["25°C\n(0.6 mΩ)", "5°C\n(1.5 mΩ)"], [Q_25C, Q_5C],
               color=["#6699CC", "#CC4444"], edgecolor="black", linewidth=0.8, width=0.5)
for bar, q in zip(bars, [Q_25C, Q_5C]):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
             f"{q:.1f} W", ha="center", fontsize=12, fontweight="bold")
ax2.set_ylabel("I²R Heat Generation (W)")
ax2.set_title("Heat Generation: I²R at 280 A (1C)")
ax2.grid(True, alpha=0.3, axis="y"); ax2.set_ylim(0, 140)
ax2.annotate("2.5× increase", xy=(1, Q_5C / 2), fontsize=11, fontweight="bold", color="darkred", ha="center")

fig.tight_layout()
save(fig, "ch10_fast_charging_thermal.png")

# --- 10.10.5 Wireless Charging Efficiency ---

k_arr = np.linspace(0.01, 0.8, 500)
def _eta_max(k, Q1, Q2):
    kQQ = k**2 * Q1 * Q2
    return kQQ / (1 + np.sqrt(1 + kQQ))**2

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_arr, _eta_max(k_arr, 712, 890) * 100, "b-", linewidth=2, label="EV (Q₁=712, Q₂=890)")
ax.plot(k_arr, _eta_max(k_arr, 200, 250) * 100, "g-", linewidth=2, label="Mid-range (Q₁=200, Q₂=250)")
ax.plot(k_arr, _eta_max(k_arr, 31, 11) * 100, "r-", linewidth=2, label="Qi consumer (Q₁=31, Q₂=11)")
ax.axvspan(0.1, 0.3, alpha=0.1, color="blue", label="EV range (k = 0.1–0.3)")
ax.axvspan(0.5, 0.8, alpha=0.1, color="red", label="Qi range (k = 0.5–0.8)")
ax.plot(0.20, _eta_max(0.20, 712, 890) * 100, "bo", markersize=10, zorder=5)
ax.annotate(f"Example 10.10.5\nk = 0.20, η = {_eta_max(0.20, 712, 890)*100:.1f}%",
            xy=(0.20, _eta_max(0.20, 712, 890) * 100), xytext=(0.35, 96),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="blue"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.set_xlabel("Coupling Coefficient (k)"); ax.set_ylabel("Maximum Theoretical Efficiency (%)")
ax.set_title("Wireless Charging Efficiency vs Coupling Coefficient")
ax.legend(fontsize=9, loc="lower right"); ax.grid(True, alpha=0.3)
ax.set_xlim(0, 0.8); ax.set_ylim(0, 100)

fig.tight_layout()
save(fig, "ch10_wireless_charging_efficiency.png")


# --- 10.11.2 Ragone Plot ---
import matplotlib.patches as _mp10_rag
_technologies_g = [
    (1e-3,  0.05,   1e6,   1e8,   "Conventional\nCapacitor",  "#9B59B6"),
    (1,     15,     500,   1e4,   "EDLC\n(Supercapacitor)",   "#2980B9"),
    (5,     30,     200,   5e3,   "Pseudocapacitor /\nHybrid", "#27AE60"),
    (10,    30,     100,   2e3,   "Lithium-Ion\nCapacitor",    "#1ABC9C"),
    (20,    40,     50,    300,   "Lead-Acid\nBattery",        "#E67E22"),
    (100,   250,    100,   1000,  "Li-Ion\nBattery",           "#E74C3C"),
    (150,   400,    50,    500,   "Li-S / Li-Air\n(emerging)", "#C0392B"),
]
fig, ax = plt.subplots(figsize=(10, 7))
for _emin, _emax, _pmin, _pmax, _lbl, _col in _technologies_g:
    _rect_g = _mp10_rag.FancyBboxPatch(
        (_emin, _pmin), _emax - _emin, _pmax - _pmin,
        boxstyle="round,pad=0", linewidth=1.5,
        edgecolor=_col, facecolor=_col, alpha=0.20,
    )
    ax.add_patch(_rect_g)
    _xe = (_emin * _emax) ** 0.5
    _xp = (_pmin * _pmax) ** 0.5
    ax.text(_xe, _xp, _lbl, ha="center", va="center",
            fontsize=8.5, color=_col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))
ax.plot(5.9, 8000, "b^", markersize=11, zorder=6, label="Example 10.11.2 EDLC")
ax.annotate("§10.11.2\nEDLC example", xy=(5.9, 8000), xytext=(2, 12000),
            fontsize=8, color="#2980B9", arrowprops=dict(arrowstyle="->", color="#2980B9"))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Specific Energy (Wh/kg)", fontsize=11)
ax.set_ylabel("Specific Power (W/kg)", fontsize=11)
ax.set_title("Ragone Plot — Energy Storage Technology Comparison\n"
             "(Approximate performance envelopes; actual values vary by product and condition)", fontsize=11)
ax.set_xlim(5e-4, 600); ax.set_ylim(30, 2e8)
ax.grid(True, which="both", alpha=0.25)
fig.tight_layout()
save(fig, "ch10_supercap_ragone.png")


# --- 10.11.3 Supercapacitor Discharge Voltage Profile ---
_C_g = 10.0; _ESR_g = 0.050; _V0_g = 16.2; _I_g = 20.0
_Vmin_g = _V0_g / 2
_t_end_g = (_V0_g - _Vmin_g) / (_I_g / _C_g)
_t_g = np.linspace(0, _t_end_g + 0.5, 500)
_Vc_g   = np.where(_t_g <= _t_end_g, _V0_g - (_I_g / _C_g) * _t_g, _Vmin_g)
_Vt_g   = np.where(_t_g <= _t_end_g, _Vc_g - _I_g * _ESR_g,        _Vmin_g)
_Vb_g   = np.full_like(_t_g, np.mean(_Vc_g[_t_g <= _t_end_g]))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(_t_g, _Vc_g,  "b-",  linewidth=2.5, label="V_C (capacitor voltage, ideal)")
ax.plot(_t_g, _Vt_g,  "r--", linewidth=2.0, label=f"V_terminal (ESR = {_ESR_g*1000:.0f} mΩ)")
ax.plot(_t_g, _Vb_g,  "g:",  linewidth=1.5, alpha=0.7,
        label=f"Battery reference (flat at {_Vb_g[0]:.1f} V)")
ax.annotate(f"ESR drop = {_I_g*_ESR_g:.1f} V",
            xy=(0.1, _V0_g - _I_g * _ESR_g), xytext=(0.8, _V0_g - _I_g * _ESR_g - 2.5),
            fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"))
ax.axvline(x=_t_end_g, color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
ax.axhline(y=_Vmin_g,  color="gray", linestyle="--", alpha=0.6, linewidth=1.2)
ax.annotate(f"t = {_t_end_g:.2f} s, V₀/2 = {_Vmin_g:.1f} V\n(75% energy extracted)",
            xy=(_t_end_g, _Vmin_g), xytext=(_t_end_g - 1.8, _Vmin_g + 2.5),
            fontsize=9, color="gray", arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_xlabel("Time (s)", fontsize=11); ax.set_ylabel("Voltage (V)", fontsize=11)
ax.set_title(f"Supercapacitor Constant-Current Discharge (C={_C_g} F, ESR={_ESR_g*1000:.0f} mΩ, "
             f"V₀={_V0_g} V, I={_I_g:.0f} A)", fontsize=11)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim(0, _t_end_g + 0.5); ax.set_ylim(0, _V0_g + 2)
fig.tight_layout()
save(fig, "ch10_supercap_discharge.png")


# --- 10.11.1 EDLC Fundamentals: Energy vs Voltage ---
_C_edlc_g = 750.0
_V_org_g  = 2.7
_V_aq_g   = 1.2
_V_std_g  = 2.5
_v_sw_g   = np.linspace(0, _V_org_g, 500)
_E_sw_g   = 0.5 * _C_edlc_g * _v_sw_g**2
_E_org_g  = 0.5 * _C_edlc_g * _V_org_g**2
_E_aq_g   = 0.5 * _C_edlc_g * _V_aq_g**2
_E_std_g  = 0.5 * _C_edlc_g * _V_std_g**2
fig, (ax_ec_g, ax_eb_g) = plt.subplots(1, 2, figsize=(12, 5))
ax_ec_g.plot(_v_sw_g, _E_sw_g, "b-", linewidth=2.5, label="E = ½CV²  (C = 750 F)")
_v_use_g    = np.linspace(_V_org_g / 2, _V_org_g, 200)
_E_use_g    = 0.5 * _C_edlc_g * _v_use_g**2
ax_ec_g.fill_between(_v_use_g, _E_use_g, alpha=0.25, color="green",
                     label="Usable 75%  (V_max/2 → V_max)")
_v_str_g    = np.linspace(0, _V_org_g / 2, 200)
_E_str_g    = 0.5 * _C_edlc_g * _v_str_g**2
ax_ec_g.fill_between(_v_str_g, _E_str_g, alpha=0.12, color="red",
                     label="Stranded 25%  (0 → V_max/2)")
ax_ec_g.axvline(x=_V_org_g / 2, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
ax_ec_g.plot(_V_org_g, _E_org_g, "bo", markersize=10, zorder=5)
ax_ec_g.plot(_V_aq_g,  _E_aq_g,  "rs", markersize=10, zorder=5)
ax_ec_g.annotate(f"Organic max\n{_E_org_g:.0f} J  ({_E_org_g/3600:.3f} Wh)",
                 xy=(_V_org_g, _E_org_g), xytext=(1.85, _E_org_g - 280),
                 fontsize=9, color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
ax_ec_g.annotate(f"Aqueous max\n{_E_aq_g:.0f} J  ({_E_aq_g/3600:.4f} Wh)",
                 xy=(_V_aq_g, _E_aq_g), xytext=(0.2, _E_aq_g + 350),
                 fontsize=9, color="red", arrowprops=dict(arrowstyle="->", color="red"))
ax_ec_g.text(1.6, 1600, f"Ratio: {_E_org_g/_E_aq_g:.2f}×", fontsize=11,
             fontweight="bold", color="purple")
ax_ec_g.set_xlabel("Cell Voltage (V)"); ax_ec_g.set_ylabel("Energy Stored (J)")
ax_ec_g.set_title(f"EDLC Energy vs Voltage  (C = {_C_edlc_g:.0f} F)")
ax_ec_g.legend(fontsize=9); ax_ec_g.grid(True, alpha=0.3)
ax_ec_g.set_xlim(0, _V_org_g * 1.05); ax_ec_g.set_ylim(0, _E_org_g * 1.18)
_v_lab_g = ["Aqueous\n1.2 V", "Organic\n2.5 V", "Organic\n2.7 V"]
_en_g    = [_E_aq_g, _E_std_g, _E_org_g]
_bc_g    = ["#E74C3C", "#F39C12", "#2980B9"]
_bars_g  = ax_eb_g.bar(_v_lab_g, _en_g, color=_bc_g, edgecolor="black", linewidth=0.8)
for _b_g, _ej_g in zip(_bars_g, _en_g):
    ax_eb_g.text(_b_g.get_x() + _b_g.get_width() / 2, _ej_g + 50,
                 f"{_ej_g:.0f} J\n({_ej_g/3600*1000:.0f} mWh)",
                 ha="center", fontsize=10, fontweight="bold")
ax_eb_g.set_ylabel("Energy Stored (J)")
ax_eb_g.set_title(f"Energy at Different V_max  (C = {_C_edlc_g:.0f} F)")
ax_eb_g.grid(True, alpha=0.3, axis="y"); ax_eb_g.set_ylim(0, _E_org_g * 1.22)
fig.tight_layout()
save(fig, "ch10_supercap_edlc_energy.png")


# --- 10.11.4 Series-Parallel Bank Design ---
_C_cb_g  = 3000.0; _V_cb_g = 2.7
_E_cb_g  = 0.5 * _C_cb_g * _V_cb_g**2
_ns_g    = np.arange(1, 28)
def _bank_g(ns, np_):
    V = ns * _V_cb_g; C = np_ * _C_cb_g / ns; E = 0.5 * C * V**2
    return V, C, E
_V1_g, _C1_g, _E1_g = _bank_g(_ns_g, 1)
_V2_g, _C2_g, _E2_g = _bank_g(_ns_g, 2)
fig, (ax_bv_g, ax_bc_g, ax_be_g) = plt.subplots(1, 3, figsize=(14, 5))
ax_bv_g.plot(_ns_g, _V1_g, "b-o", linewidth=2, markersize=5, label="n_p = 1")
ax_bv_g.plot(_ns_g, _V2_g, "r--s", linewidth=2, markersize=5, label="n_p = 2 (same V)")
ax_bv_g.set_xlabel("Series Cells (n_s)"); ax_bv_g.set_ylabel("Bank Voltage (V)")
ax_bv_g.set_title("Bank Voltage vs n_s"); ax_bv_g.legend(fontsize=9); ax_bv_g.grid(True, alpha=0.3)
ax_bc_g.plot(_ns_g, _C1_g, "b-o", linewidth=2, markersize=5, label="n_p = 1")
ax_bc_g.plot(_ns_g, _C2_g, "r--s", linewidth=2, markersize=5, label="n_p = 2")
ax_bc_g.set_xlabel("Series Cells (n_s)"); ax_bc_g.set_ylabel("Bank Capacitance (F)")
ax_bc_g.set_title("Bank Capacitance vs n_s"); ax_bc_g.legend(fontsize=9); ax_bc_g.grid(True, alpha=0.3)
ax_be_g.plot(_ns_g, _E1_g / 3600, "b-o", linewidth=2, markersize=5, label="n_p = 1")
ax_be_g.plot(_ns_g, _E2_g / 3600, "r--s", linewidth=2, markersize=5, label="n_p = 2")
ax_be_g.axhline(y=_E_cb_g / 3600, color="gray", linestyle=":", alpha=0.5,
                label=f"1 cell = {_E_cb_g/3600:.2f} Wh")
_ns_ex_g, _np_ex_g = 18, 2
_, _, _E_ex_g = _bank_g(_ns_ex_g, _np_ex_g)
ax_be_g.plot(_ns_ex_g, _E_ex_g / 3600, "g*", markersize=15, zorder=6,
             label=f"Ex 10.11.4: n_s={_ns_ex_g}, n_p={_np_ex_g}\n{_E_ex_g/3600:.1f} Wh @ 48.6 V")
ax_be_g.set_xlabel("Series Cells (n_s)"); ax_be_g.set_ylabel("Bank Energy (Wh)")
ax_be_g.set_title("Bank Energy vs n_s\n(E = n_s × n_p × E_cell)")
ax_be_g.legend(fontsize=8, loc="upper left"); ax_be_g.grid(True, alpha=0.3)
fig.suptitle(f"Series-Parallel Bank Design  (C_cell = {_C_cb_g:.0f} F, V_cell = {_V_cb_g} V)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, "ch10_supercap_bank_design.png")


# --- 10.11.5 Applications: Regenerative Braking Cycle ---
_C_rg_g = 2.67; _Vmn_g = 100.0; _Vmx_g = 200.0
_E_rg_g = 40_000; _tb_g = 5.0; _th_g = 1.0; _ta_g = 5.0
_Ir_g   = _E_rg_g / (_tb_g * (_Vmn_g + _Vmx_g) / 2)
_dt_g   = 0.02
_t_b_g  = np.arange(0, _tb_g, _dt_g)
_t_h_g  = np.arange(0, _th_g, _dt_g)
_t_a_g  = np.arange(0, _ta_g, _dt_g)
_Vc_b_g = _Vmn_g + (_Ir_g / _C_rg_g) * _t_b_g
_Vc_h_g = np.full_like(_t_h_g, _Vmx_g)
_Vc_a_g = _Vmx_g - (_Ir_g / _C_rg_g) * _t_a_g
_I_b_g  = np.full_like(_t_b_g,  _Ir_g)
_I_h_g  = np.zeros_like(_t_h_g)
_I_a_g  = np.full_like(_t_a_g, -_Ir_g)
_P_b_g  = (_Vc_b_g * _Ir_g)  / 1000
_P_h_g  = np.zeros_like(_t_h_g)
_P_a_g  = -(_Vc_a_g * _Ir_g) / 1000
_tall_g = np.concatenate([_t_b_g, _tb_g + _t_h_g, _tb_g + _th_g + _t_a_g])
_Vca_g  = np.concatenate([_Vc_b_g, _Vc_h_g, _Vc_a_g])
_Ia_g   = np.concatenate([_I_b_g,  _I_h_g,  _I_a_g])
_Pa_g   = np.concatenate([_P_b_g,  _P_h_g,  _P_a_g])
fig, (ax_rv_g, ax_ri_g, ax_rp_g) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for _ax_rg in (ax_rv_g, ax_ri_g, ax_rp_g):
    _ax_rg.axvspan(0, _tb_g, alpha=0.06, color="green")
    _ax_rg.axvspan(_tb_g, _tb_g + _th_g, alpha=0.06, color="gray")
    _ax_rg.axvspan(_tb_g + _th_g, _tb_g + _th_g + _ta_g, alpha=0.06, color="orange")
ax_rv_g.plot(_tall_g, _Vca_g, "b-", linewidth=2.5)
ax_rv_g.axhline(y=_Vmx_g, color="red",  linestyle="--", alpha=0.4, label=f"V_max = {_Vmx_g:.0f} V")
ax_rv_g.axhline(y=_Vmn_g, color="gray", linestyle="--", alpha=0.4, label=f"V_min = {_Vmn_g:.0f} V")
ax_rv_g.set_ylabel("V_C (V)")
ax_rv_g.set_title(f"Regenerative Braking Cycle — Light Rail Tram\n"
                  f"C = {_C_rg_g} F,  V range = {_Vmn_g:.0f}–{_Vmx_g:.0f} V,  I_avg = {_Ir_g:.1f} A")
ax_rv_g.legend(fontsize=9); ax_rv_g.grid(True, alpha=0.3)
ax_rv_g.set_ylim(_Vmn_g * 0.82, _Vmx_g * 1.12)
ax_rv_g.text(2.5, _Vmn_g + 5, "Braking\n(charging)", ha="center",
             fontsize=10, color="darkgreen", fontweight="bold")
ax_rv_g.text(_tb_g + _th_g / 2, _Vmx_g - 10, "Hold", ha="center", fontsize=9, color="gray")
ax_rv_g.text(_tb_g + _th_g + 2.5, _Vmn_g + 5, "Acceleration\n(discharging)",
             ha="center", fontsize=10, color="darkorange", fontweight="bold")
ax_ri_g.plot(_tall_g, _Ia_g, "r-", linewidth=2)
ax_ri_g.axhline(y=0, color="black", linewidth=0.8)
ax_ri_g.set_ylabel("Current (A)"); ax_ri_g.set_ylim(-_Ir_g * 1.4, _Ir_g * 1.4)
ax_ri_g.grid(True, alpha=0.3)
ax_ri_g.text(2.5,  _Ir_g * 0.45, f"+{_Ir_g:.1f} A", ha="center", fontsize=10, color="darkgreen")
ax_ri_g.text(_tb_g + _th_g + 2.5, -_Ir_g * 0.45, f"−{_Ir_g:.1f} A",
             ha="center", fontsize=10, color="darkorange")
ax_rp_g.fill_between(_tall_g, _Pa_g, 0, where=(_Pa_g >= 0),
                     alpha=0.3, color="green", label=f"Regen in ≈ {_E_rg_g/1000:.0f} kJ")
ax_rp_g.fill_between(_tall_g, _Pa_g, 0, where=(_Pa_g <= 0),
                     alpha=0.3, color="orange", label="Traction assist")
ax_rp_g.plot(_tall_g, _Pa_g, "k-", linewidth=2)
ax_rp_g.axhline(y=0, color="black", linewidth=0.8)
ax_rp_g.set_xlabel("Time (s)"); ax_rp_g.set_ylabel("Power (kW)")
ax_rp_g.legend(fontsize=9, loc="lower right"); ax_rp_g.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch10_supercap_regen.png")


# --- §10.12.1 PV Module I-V and P-V Curves — Effect of Irradiance ---
fig, (ax_iv, ax_pv) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
fig.suptitle("PV Module I-V and P-V Curves — Effect of Irradiance at 25 °C\n"
             "(60-cell mono-Si: V_oc = 45 V, I_sc = 9.0 A at STC)",
             fontsize=12, fontweight="bold")

# Single-diode model parameters at STC (G=1000 W/m²)
_Isc_stc = 9.0       # A
_Voc_stc = 45.0      # V
_n_id    = 1.3       # ideality factor
_Ns      = 60        # cells in series
_Vt      = 0.02585 * _Ns          # module thermal voltage (V)
_a       = _n_id * _Vt            # n·V_T for module
_I0      = _Isc_stc / np.exp(_Voc_stc / _a)   # saturation current (A)

_irradiances = [1000, 750, 500, 250]
_colors      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
_labels      = ["1000 W/m² (STC)", "750 W/m²", "500 W/m²", "250 W/m²"]

for _G, _col, _lbl in zip(_irradiances, _colors, _labels):
    _Isc_g = _Isc_stc * (_G / 1000)
    _Voc_g = _Voc_stc + _a * np.log(_G / 1000) if _G < 1000 else _Voc_stc
    _I0_g  = _Isc_g / np.exp(_Voc_g / _a)

    _Vv = np.linspace(0, _Voc_g * 1.01, 500)
    _Ii = _Isc_g - _I0_g * (np.exp(_Vv / _a) - 1)
    _Ii = np.clip(_Ii, 0, None)
    _Pp = _Vv * _Ii

    # Find MPP
    _idx_mpp = np.argmax(_Pp)
    _Vmp_g   = _Vv[_idx_mpp]
    _Imp_g   = _Ii[_idx_mpp]
    _Pmp_g   = _Pp[_idx_mpp]

    ax_iv.plot(_Vv, _Ii, color=_col, linewidth=2, label=_lbl)
    ax_iv.plot(_Vmp_g, _Imp_g, "o", color=_col, markersize=7)

    ax_pv.plot(_Vv, _Pp, color=_col, linewidth=2, label=f"{_lbl}  (P_max = {_Pmp_g:.0f} W)")
    ax_pv.plot(_Vmp_g, _Pmp_g, "o", color=_col, markersize=7)

ax_iv.set_ylabel("Current (A)", fontsize=11)
ax_iv.set_title("I-V Curves (dots = MPP)", fontsize=11)
ax_iv.legend(fontsize=9, loc="upper right")
ax_iv.set_ylim(0, 10)
ax_iv.grid(True, alpha=0.3)

ax_pv.set_xlabel("Voltage (V)", fontsize=11)
ax_pv.set_ylabel("Power (W)", fontsize=11)
ax_pv.set_title("P-V Curves (dots = MPP)", fontsize=11)
ax_pv.legend(fontsize=9, loc="upper left")
ax_pv.set_ylim(0, 360)
ax_pv.grid(True, alpha=0.3)

fig.tight_layout()
save(fig, "ch10_pv_iv_curves.png")


# ============================================================
# Chapter 12: Electric Motors
# ============================================================
print("Chapter 12: Electric Motors")

# --- 12.3.4 Stepper Motor Torque Curves ---
T_hold = 1.9; f_res = 175
steps_s = np.linspace(0, 2000, 1000)
T_pullout = T_hold * np.exp(-steps_s / 800)
res_dip = 0.3 * np.exp(-((steps_s - f_res) / 30)**2)
T_pullout = np.maximum(T_pullout - res_dip, 0)
T_pullin = np.maximum(0.7 * T_hold * np.exp(-steps_s / 500) - res_dip * 0.8, 0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps_s, T_pullout, "b-", linewidth=2, label="Pull-out torque")
ax.plot(steps_s, T_pullin, "r--", linewidth=2, label="Pull-in torque")
ax.fill_between(steps_s, T_pullin, T_pullout, alpha=0.1, color="blue", label="Slew range (run only)")
ax.fill_between(steps_s, 0, T_pullin, alpha=0.1, color="green", label="Start/stop range")
ax.axvline(x=f_res, color="orange", linestyle=":", alpha=0.7)
ax.annotate(f"Mid-band\nresonance\n({f_res} steps/s)",
            xy=(f_res, T_pullout[np.argmin(np.abs(steps_s - f_res))]),
            xytext=(f_res + 150, 1.4), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="orange"), color="orange")
idx_1k = np.argmin(np.abs(steps_s - 1000))
ax.plot(1000, T_pullout[idx_1k], "go", markersize=8, zorder=5)
ax.annotate(f"1000 steps/s\nT ≈ {T_pullout[idx_1k]:.2f} N·m",
            xy=(1000, T_pullout[idx_1k]), xytext=(1100, T_pullout[idx_1k] + 0.3),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.set_xlabel("Step Rate (steps/s)"); ax.set_ylabel("Torque (N·m)")
ax.set_title("Stepper Motor Torque vs Speed (NEMA 23, 1.9 N·m holding)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 2000); ax.set_ylim(0, 2.2)
fig.tight_layout()
save(fig, "ch12_stepper_torque.png")

# --- 12.4.5 FOC Vector Diagram ---
Id_foc = 0; Iq_foc = 8.0; Rs_foc = 0.5; Ld_foc = 8e-3; Lq_foc = 12e-3
lam_m = 0.25; omega_e_foc = 314.2
Vd_foc = Rs_foc * Id_foc - omega_e_foc * Lq_foc * Iq_foc
Vq_foc = Rs_foc * Iq_foc + omega_e_foc * Ld_foc * Id_foc + omega_e_foc * lam_m
Vs_foc = np.sqrt(Vd_foc**2 + Vq_foc**2)
Is_foc = np.sqrt(Id_foc**2 + Iq_foc**2)

fig, ax = plt.subplots(figsize=(8, 8))
ax.axhline(y=0, color="gray", linewidth=0.5); ax.axvline(x=0, color="gray", linewidth=0.5)
sc_I = 5; sc_V = 0.5
ax.annotate("", xy=(Id_foc*sc_I, Iq_foc*sc_I), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
ax.text(Id_foc*sc_I+1, Iq_foc*sc_I, f"Is = {Is_foc:.1f} A\n(Iq = {Iq_foc:.1f} A)", fontsize=11, color="blue")
ax.annotate("", xy=(Vd_foc*sc_V, Vq_foc*sc_V), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
ax.text(Vd_foc*sc_V-8, Vq_foc*sc_V+1,
        f"Vs = {Vs_foc:.1f} V\n(Vd={Vd_foc:.1f}, Vq={Vq_foc:.1f})", fontsize=11, color="red")
emf_q = omega_e_foc * lam_m
ax.annotate("", xy=(0, emf_q*sc_V), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="green", lw=2, linestyle="--"))
ax.text(1, emf_q*sc_V-2, f"Back-EMF\n= {emf_q:.1f} V", fontsize=10, color="green")
ax.annotate("", xy=(lam_m*150, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="purple", lw=2, linestyle="--"))
ax.text(lam_m*150+1, -3, f"λm = {lam_m} Wb", fontsize=10, color="purple")
ax.set_xlim(-25, 50); ax.set_ylim(-10, 55); ax.set_aspect("equal")
ax.set_xlabel("d-axis"); ax.set_ylabel("q-axis")
ax.set_title("FOC Vector Diagram: PMSM with Id = 0 at 1500 RPM"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch12_foc_vectors.png")

# --- 12.5.6 Motor Insulation Life vs Temperature ---
T_rated_ins = 155; L_rated_ins = 20000
T_range_ins = np.linspace(130, 200, 200)
L_life = L_rated_ins * 2**((T_rated_ins - T_range_ins) / 10)
T_ex_ins = 172.1; L_ex_ins = L_rated_ins * 2**((T_rated_ins - T_ex_ins) / 10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(T_range_ins, L_life, "b-", linewidth=2, label="Insulation life (Arrhenius)")
ax.plot(T_rated_ins, L_rated_ins, "go", markersize=10, zorder=5)
ax.annotate(f"Rated: {T_rated_ins}°C, {L_rated_ins:,} hrs",
            xy=(T_rated_ins, L_rated_ins), xytext=(T_rated_ins-15, L_rated_ins*2),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.plot(T_ex_ins, L_ex_ins, "ro", markersize=10, zorder=5)
ax.annotate(f"Example: {T_ex_ins}°C\n{L_ex_ins:,.0f} hrs (69% reduction)",
            xy=(T_ex_ins, L_ex_ins), xytext=(T_ex_ins+3, L_ex_ins*3),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")
for temp, cls in [(130, "Class B"), (155, "Class F"), (180, "Class H")]:
    ax.axvline(x=temp, color="gray", linestyle=":", alpha=0.5)
    ax.text(temp+0.5, 800, cls, fontsize=8, color="gray", rotation=90)
ax.set_xlabel("Hot-Spot Temperature (°C)"); ax.set_ylabel("Insulation Life (hours)")
ax.set_title("Motor Insulation Life vs Temperature (Arrhenius / 10°C Rule)")
ax.legend(); ax.grid(True, alpha=0.3, which="both"); ax.set_xlim(130, 200)
fig.tight_layout()
save(fig, "ch12_insulation_life.png")


# ============================================================
# Chapter 13: Op-Amps
# ============================================================
print("Chapter 13: Op-Amps")

# --- 13.2.3 Integrator ---
R_int=10e3; C_int=0.1e-6; Vin_int=-2; t_int=np.linspace(0, 8e-3, 1000)
Vout_int=np.minimum(-(Vin_int/(R_int*C_int))*t_int, 12)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
ax1.plot(t_int*1e3, np.full_like(t_int, Vin_int), "b-", linewidth=2)
ax1.set_ylabel("V_in (V)"); ax1.set_title("Integrator: Input Step"); ax1.set_ylim(-3, 1); ax1.grid(True, alpha=0.3)
ax2.plot(t_int*1e3, Vout_int, "r-", linewidth=2, label="V_out(t)")
ax2.axhline(y=12, color="gray", linestyle="--", alpha=0.5, label="V_out = 12 V")
ax2.plot(6, 12, "go", markersize=10, zorder=5)
ax2.annotate("t = 6 ms, V_out = 12 V", xy=(6, 12), xytext=(6.3, 9), fontsize=10,
             arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("V_out (V)")
ax2.set_title("Integrator Output Ramp (R = 10 kΩ, C = 0.1 μF)"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch13_integrator.png")

# --- 13.2.4 Differentiator ---
R_d=100e3; C_d=0.01e-6; f_d=1000; Vpp=2
t_d=np.linspace(0, 3e-3, 3000); per=1/f_d
v_tri=(2*Vpp/per)*(per/2-np.abs(t_d%per-per/2))-Vpp/2
dv=np.gradient(v_tri, t_d); v_out_d=-R_d*C_d*dv

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_d*1e3, v_tri, "b-", linewidth=2, label="V_in (triangular)")
ax1.set_ylabel("V_in (V)"); ax1.set_title("Differentiator Input: 1 kHz Triangular Wave"); ax1.legend(); ax1.grid(True, alpha=0.3)
exp_lev=R_d*C_d*2*Vpp*f_d
ax2.plot(t_d*1e3, v_out_d, "r-", linewidth=2, label="V_out = −RC × dV_in/dt")
ax2.axhline(y=exp_lev, color="gray", linestyle=":", alpha=0.5, label=f"±{exp_lev:.1f} V")
ax2.axhline(y=-exp_lev, color="gray", linestyle=":", alpha=0.5)
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("V_out (V)")
ax2.set_title("Differentiator Output: Square Wave"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch13_differentiator.png")

# --- 13.5.2 Sallen-Key ---
fc_sk=1000; Q_sk=0.707; f_sk=np.logspace(1, 5, 1000)
s_n=1j*f_sk/fc_sk; H_sk=1/(1+s_n/Q_sk+s_n**2)
H_mag_sk=20*np.log10(np.abs(H_sk)); H_ph_sk=np.degrees(np.angle(H_sk))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
ax1.semilogx(f_sk, H_mag_sk, "b-", linewidth=2)
ax1.axhline(y=-3, color="red", linestyle="--", alpha=0.6, label="−3 dB")
ax1.axvline(x=fc_sk, color="red", linestyle=":", alpha=0.6, label=f"f_c = {fc_sk} Hz")
f_sl=np.array([5000, 50000]); sl=-40*np.log10(f_sl/fc_sk)-3
ax1.plot(f_sl, sl, "g--", linewidth=1.5, alpha=0.7, label="−40 dB/decade")
ax1.set_ylabel("Magnitude (dB)"); ax1.set_title("Sallen-Key 2nd-Order Butterworth (f_c = 1 kHz)")
ax1.set_ylim(-80, 5); ax1.legend(); ax1.grid(True, alpha=0.3, which="both")
ax2.semilogx(f_sk, H_ph_sk, "r-", linewidth=2)
ax2.axhline(y=-90, color="gray", linestyle=":", alpha=0.5); ax2.axvline(x=fc_sk, color="red", linestyle=":", alpha=0.6)
ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Phase (°)"); ax2.set_ylim(-180, 0)
ax2.legend(); ax2.grid(True, alpha=0.3, which="both")
fig.tight_layout()
save(fig, "ch13_sallen_key.png")

# --- 13.6.2 Schmitt Trigger ---
VTH=3.0; VTL=2.0; Vsp=13.5; Vsn=-13.5
t_sm=np.linspace(0, 4e-3, 4000); Vin_sm=2.5+1.5*np.sin(2*np.pi*500*t_sm)
Vout_sm=np.zeros_like(Vin_sm); state=-1
for i in range(len(Vin_sm)):
    if state==-1 and Vin_sm[i]>VTH: state=1
    elif state==1 and Vin_sm[i]<VTL: state=-1
    Vout_sm[i]=Vsp if state==1 else Vsn

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot([0, VTL, VTL], [Vsn, Vsn, Vsp], "b-", linewidth=2, label="Rising")
ax1.plot([VTH, VTH, 5], [Vsp, Vsn, Vsn], "r-", linewidth=2, label="Falling")
ax1.plot([VTL, VTH], [Vsp, Vsp], "b-", linewidth=2); ax1.plot([0, VTL], [Vsn, Vsn], "r-", linewidth=2)
ax1.annotate("", xy=(VTL, Vsp-1), xytext=(VTL, Vsn+1), arrowprops=dict(arrowstyle="->", color="blue", lw=2))
ax1.annotate("", xy=(VTH, Vsn+1), xytext=(VTH, Vsp-1), arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax1.annotate(f"V_TL = {VTL} V", xy=(VTL, -16), fontsize=10, ha="center", color="blue")
ax1.annotate(f"V_TH = {VTH} V", xy=(VTH, -16), fontsize=10, ha="center", color="red")
ax1.set_xlabel("V_in (V)"); ax1.set_ylabel("V_out (V)"); ax1.set_title("Hysteresis Transfer Characteristic")
ax1.set_xlim(-0.5, 5.5); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(t_sm*1e3, Vin_sm, "b-", linewidth=1.5, label="V_in")
ax2.axhline(y=VTH, color="red", linestyle="--", alpha=0.5, label=f"V_TH = {VTH} V")
ax2.axhline(y=VTL, color="blue", linestyle="--", alpha=0.5, label=f"V_TL = {VTL} V")
ax2t=ax2.twinx(); ax2t.plot(t_sm*1e3, Vout_sm, "r-", linewidth=1.5, alpha=0.7, label="V_out")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("V_in (V)", color="blue"); ax2t.set_ylabel("V_out (V)", color="red")
ax2.set_title("Schmitt Trigger Time Response")
l1, lb1 = ax2.get_legend_handles_labels(); l2, lb2 = ax2t.get_legend_handles_labels()
ax2.legend(l1+l2, lb1+lb2, fontsize=8); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch13_schmitt_trigger.png")

# --- 13.7.2 Slew Rate ---
SR=13e6; Vpp_sr=20; f_sr=200e3; t_sr=np.linspace(0, 10e-6, 5000); T_sr=1/f_sr
v_ideal=Vpp_sr/2*np.sign(np.sin(2*np.pi*f_sr*t_sr))
v_slew=np.zeros_like(t_sr); v_slew[0]=v_ideal[0]; dt_sr=t_sr[1]-t_sr[0]
for i in range(1, len(t_sr)):
    d=v_ideal[i]-v_slew[i-1]; mx=SR*dt_sr
    v_slew[i]=v_slew[i-1]+np.sign(d)*min(abs(d), mx)
t_rise_sr=Vpp_sr/SR

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_sr*1e6, v_ideal, "b--", linewidth=1, alpha=0.5, label="Ideal square wave")
ax.plot(t_sr*1e6, v_slew, "r-", linewidth=2, label=f"Slew-limited (SR = 13 V/μs)")
ax.annotate(f"Rise time = {t_rise_sr*1e6:.2f} μs", xy=(T_sr/2*1e6+t_rise_sr/2*1e6, 0), xytext=(T_sr/2*1e6+2, -4),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.set_xlabel("Time (μs)"); ax.set_ylabel("Voltage (V)")
ax.set_title(f"Slew Rate Limiting: 200 kHz, SR = 13 V/μs"); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch13_slew_rate.png")

# --- 13.2.5 Log Amplifier Transfer Curve ---
kT_q = 0.02585; I_S_log = 1e-14; R_in_log = 10e3
V_in_log = np.logspace(-2, 1, 500)
V_out_log = -kT_q * np.log(V_in_log / R_in_log / I_S_log)
V_pts = [0.01, 0.1, 1.0, 10.0]
Vo_pts = [-kT_q * np.log(v / R_in_log / I_S_log) for v in V_pts]

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(V_in_log, V_out_log * 1000, "b-", linewidth=2,
            label="V_out = -(kT/q) ln(V_in/(I_S R))")
for v, vo in zip(V_pts, Vo_pts):
    ax.plot(v, vo * 1000, "ro", markersize=8, zorder=5)
    ax.annotate(f"{vo*1000:.0f} mV", xy=(v, vo*1000),
                xytext=(v * 1.5, vo*1000 + 8), fontsize=9, color="red")
ax.set_xlabel("V_in (V)"); ax.set_ylabel("V_out (mV)")
ax.set_title("Log Amplifier Transfer Curve (I_S = 10⁻¹⁴ A, R_in = 10 kΩ, T = 25°C)")
ax.annotate("−59.5 mV/decade", xy=(0.1, -536), fontsize=11, color="green",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.legend(); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
save(fig, "ch13_log_amp.png")

# --- 13.5.4 MFB Bandpass Filter ---
f0_mfb = 1000.0; Q_mfb = 10.0; A0_mfb = 5.0
f_mfb = np.logspace(1.5, 4.5, 2000)
u_mfb = f_mfb / f0_mfb
H_mag_mfb = A0_mfb / np.sqrt(1 + Q_mfb**2 * (u_mfb - 1.0 / u_mfb)**2)
H_dB_mfb = 20 * np.log10(H_mag_mfb)
BW_mfb = f0_mfb / Q_mfb
f_low_mfb = f0_mfb * (np.sqrt(1 + 1 / (4 * Q_mfb**2)) - 1 / (2 * Q_mfb))
f_high_mfb = f0_mfb * (np.sqrt(1 + 1 / (4 * Q_mfb**2)) + 1 / (2 * Q_mfb))
A0_dB_mfb = 20 * np.log10(A0_mfb)
A3dB_mfb = A0_dB_mfb - 3

fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogx(f_mfb, H_dB_mfb, "b-", linewidth=2, label=f"MFB Bandpass (Q = {Q_mfb:.0f})")
ax.axvline(x=f0_mfb, color="red", linestyle="--", alpha=0.7, label=f"f₀ = {f0_mfb:.0f} Hz")
ax.plot(f0_mfb, A0_dB_mfb, "ro", markersize=8, zorder=5)
ax.annotate(f"f₀ = {f0_mfb:.0f} Hz\n|A₀| = {A0_dB_mfb:.1f} dB",
            xy=(f0_mfb, A0_dB_mfb), xytext=(f0_mfb * 2.5, A0_dB_mfb - 4),
            fontsize=10, color="red", arrowprops=dict(arrowstyle="->", color="red"))
ax.axhline(y=A3dB_mfb, color="green", linestyle=":", alpha=0.7,
           label=f"−3 dB level ({A3dB_mfb:.1f} dB)")
ax.axvline(x=f_low_mfb, color="green", linestyle=":", alpha=0.5)
ax.axvline(x=f_high_mfb, color="green", linestyle=":", alpha=0.5)
ax.annotate("", xy=(f_high_mfb, A3dB_mfb - 1.5), xytext=(f_low_mfb, A3dB_mfb - 1.5),
            arrowprops=dict(arrowstyle="<->", color="green", lw=1.5))
ax.annotate(f"BW = {BW_mfb:.0f} Hz",
            xy=((f_low_mfb + f_high_mfb) / 2, A3dB_mfb - 2.5), ha="center", fontsize=10, color="green")
ax.annotate(f"Q = f₀/BW = {Q_mfb:.0f}",
            xy=(150, A0_dB_mfb - 8), fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")
ax.set_title(f"MFB Bandpass Filter (f₀ = {f0_mfb:.0f} Hz, Q = {Q_mfb:.0f}, |A₀| = {A0_mfb:.0f})")
ax.set_ylim(A0_dB_mfb - 35, A0_dB_mfb + 5); ax.legend(); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
save(fig, "ch13_mfb_bandpass.png")

# --- 13.5.5 Active Twin-T Notch Filter ---
f0_notch = 60; k_notch = 0.96; Q_notch = 1 / (4 * (1 - k_notch))
f_notch = np.linspace(1, 200, 5000)
s_n = 1j * f_notch / f0_notch
H_notch = (s_n**2 + 1) / (s_n**2 + s_n / Q_notch + 1)
H_mag_notch = 20 * np.log10(np.abs(H_notch))

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(f_notch, H_mag_notch, "b-", linewidth=2)
ax.axvline(x=60, color="red", linestyle="--", alpha=0.6, label="f₀ = 60 Hz")
BW_notch = f0_notch / Q_notch
ax.axvspan(60 - BW_notch/2, 60 + BW_notch/2, alpha=0.15, color="orange",
           label=f"BW = {BW_notch:.1f} Hz (Q = {Q_notch:.2f})")
ax.axhline(y=-3, color="gray", linestyle=":", alpha=0.5, label="−3 dB")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude (dB)")
ax.set_title("Active Twin-T Notch Filter (f₀ = 60 Hz, k = 0.96)")
ax.set_ylim(-50, 5); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch13_notch_filter.png")

# --- 13.6.4 Relaxation Oscillator ---
R_rx = 22e3; C_rx = 10e-9; beta_rx = 0.5; Vsat_rx = 12
tau_rx = R_rx * C_rx; VTH_rx = beta_rx * Vsat_rx; VTL_rx = -VTH_rx
f_rx = 1 / (2 * tau_rx * np.log((1 + beta_rx) / (1 - beta_rx)))
dt_rx = 0.5e-6; t_rx = np.arange(0, 4 / f_rx, dt_rx)
v_cap_rx = np.zeros_like(t_rx); v_out_rx = np.zeros_like(t_rx)
v_cap_rx[0] = VTL_rx; v_out_rx[0] = Vsat_rx
for i in range(1, len(t_rx)):
    tgt = v_out_rx[i-1]
    v_cap_rx[i] = tgt + (v_cap_rx[i-1] - tgt) * np.exp(-dt_rx / tau_rx)
    if v_out_rx[i-1] > 0 and v_cap_rx[i] >= VTH_rx: v_out_rx[i] = -Vsat_rx
    elif v_out_rx[i-1] < 0 and v_cap_rx[i] <= VTL_rx: v_out_rx[i] = Vsat_rx
    else: v_out_rx[i] = v_out_rx[i-1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(t_rx*1e3, v_out_rx, "r-", linewidth=1.5, label="V_out (square wave)")
ax1.set_ylabel("V_out (V)"); ax1.set_title(f"Relaxation Oscillator (f = {f_rx:.0f} Hz)")
ax1.set_ylim(-15, 15); ax1.legend(); ax1.grid(True, alpha=0.3)
ax2.plot(t_rx*1e3, v_cap_rx, "b-", linewidth=2, label="V_cap (exponential)")
ax2.axhline(y=VTH_rx, color="red", linestyle="--", alpha=0.6, label=f"V_TH = +{VTH_rx:.0f} V")
ax2.axhline(y=VTL_rx, color="blue", linestyle="--", alpha=0.6, label=f"V_TL = {VTL_rx:.0f} V")
ax2.set_xlabel("Time (ms)"); ax2.set_ylabel("V_cap (V)")
ax2.set_title(f"Capacitor Voltage (R = 22 kΩ, C = 10 nF, β = {beta_rx})"); ax2.legend(); ax2.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch13_relaxation_osc.png")


# ============================================================
# Chapter 16: Antenna Design
# ============================================================
print("Chapter 16: Antenna Design")

# --- 16.5.3 Phased Array Pattern ---
N_elem = 32; d_lam = 0.55; theta_scan = 45
theta_arr = np.linspace(-90, 90, 4000); theta_arr_rad = np.radians(theta_arr)
beta_arr = -2 * np.pi * d_lam * np.sin(np.radians(theta_scan))
psi_arr = 2 * np.pi * d_lam * np.sin(theta_arr_rad) + beta_arr
af_num = np.sin(N_elem * psi_arr / 2)
af_den = N_elem * np.sin(psi_arr / 2)
with np.errstate(divide="ignore", invalid="ignore"):
    AF = np.where(np.abs(af_den) < 1e-10, 1.0, af_num / af_den)
AF_dB = np.clip(20 * np.log10(np.abs(AF) + 1e-12), -40, 0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta_arr, AF_dB, "b-", linewidth=1.5)
ax.axvline(x=theta_scan, color="red", linestyle="--", alpha=0.6, label=f"Main beam: θ₀ = {theta_scan}°")
ax.axhline(y=-3, color="gray", linestyle=":", alpha=0.5, label="−3 dB")
ax.annotate("Grating lobe (sin θ = −1.11)\noutside visible space",
            xy=(-90, -5), fontsize=9, color="orange",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.annotate("Max GL-free scan: ±54.9°", xy=(54.9, -15), fontsize=9, color="green",
            arrowprops=dict(arrowstyle="->", color="green"))
ax.set_xlabel("Angle from Broadside (°)"); ax.set_ylabel("Array Factor (dB)")
ax.set_title(f"Phased Array Pattern: {N_elem} elements, d = {d_lam}λ, scanned to {theta_scan}°")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(-90, 90); ax.set_ylim(-40, 2)
fig.tight_layout()
save(fig, "ch16_phased_array.png")


# ============================================================
# Chapter 17: Radar Systems
# ============================================================
print("Chapter 17: Radar Systems")

# --- 17.2.2 FMCW Beat Frequency vs Range ---
c_r = 3e8; B_fmcw = 4e9; T_sw = 40e-6
R_rng = np.linspace(0, 100, 500)
fb_rng = 2 * R_rng * B_fmcw / (c_r * T_sw)
f_nyq = 20e6; R_max_adc = f_nyq * c_r * T_sw / (2 * B_fmcw)
R_ex_r = 50; fb_ex_r = 2 * R_ex_r * B_fmcw / (c_r * T_sw)
dR_fmcw = c_r / (2 * B_fmcw)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(R_rng, fb_rng / 1e6, "b-", linewidth=2, label="Beat frequency")
ax.axhline(y=f_nyq / 1e6, color="red", linestyle="--", alpha=0.6,
           label=f"ADC Nyquist limit = {f_nyq/1e6:.0f} MHz")
ax.axvline(x=R_max_adc, color="red", linestyle=":", alpha=0.4)
ax.plot(R_ex_r, fb_ex_r / 1e6, "go", markersize=10, zorder=5)
ax.annotate(f"R = {R_ex_r} m\nf_b = {fb_ex_r/1e6:.1f} MHz\n(exceeds ADC limit)",
            xy=(R_ex_r, fb_ex_r/1e6), xytext=(R_ex_r+8, fb_ex_r/1e6-5),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.annotate(f"R_max = {R_max_adc:.0f} m\n(at 40 MHz ADC)",
            xy=(R_max_adc, f_nyq/1e6), xytext=(R_max_adc+8, f_nyq/1e6+5),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.annotate(f"ΔR = {dR_fmcw*100:.2f} cm", xy=(5, 2), fontsize=11, color="purple",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.set_xlabel("Range (m)"); ax.set_ylabel("Beat Frequency (MHz)")
ax.set_title("FMCW Radar: Beat Frequency vs Range (77 GHz, B = 4 GHz, T = 40 μs)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 100); ax.set_ylim(0, 70)
fig.tight_layout()
save(fig, "ch17_fmcw_beat.png")


# ============================================================
# Chapter 18: Optics
# ============================================================
print("Chapter 18: Optics")

# --- 18.2.4 Dielectric Mirror Reflectivity ---
n_H = 2.30; n_L = 1.46; ratio_nl = n_H / n_L
N_pairs = np.arange(1, 21)
R_mirror = ((ratio_nl**(2*N_pairs) - 1) / (ratio_nl**(2*N_pairs) + 1))**2
R8 = ((ratio_nl**16 - 1) / (ratio_nl**16 + 1))**2
R15 = ((ratio_nl**30 - 1) / (ratio_nl**30 + 1))**2

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(N_pairs, R_mirror * 100, "b-o", linewidth=2, markersize=6)
ax.plot(8, R8*100, "ro", markersize=12, zorder=5)
ax.annotate(f"N = 8: R = {R8*100:.2f}%", xy=(8, R8*100), xytext=(10, R8*100-3),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="red"), color="red")
ax.plot(15, R15*100, "go", markersize=12, zorder=5)
ax.annotate(f"N = 15: R = {R15*100:.4f}%", xy=(15, R15*100), xytext=(16, 97),
            fontsize=10, arrowprops=dict(arrowstyle="->", color="green"), color="green")
ax.axhline(y=99, color="gray", linestyle=":", alpha=0.5, label="99%")
ax.axhline(y=99.9, color="gray", linestyle="--", alpha=0.5, label="99.9%")
ax.set_xlabel("Number of Layer Pairs (N)"); ax.set_ylabel("Reflectivity (%)")
ax.set_title(f"Dielectric HR Mirror Reflectivity (TiO₂/SiO₂, n_H/n_L = {ratio_nl:.3f})")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 21); ax.set_ylim(0, 101)
fig.tight_layout()
save(fig, "ch18_mirror_reflectivity.png")


# ============================================================
# Chapter 19: Engineering Economics
# ============================================================
print("Chapter 19: Engineering Economics")

# --- 19.1.2 Compound Interest Growth ---
P_ci = 100_000; r_ci = 0.05
years_ci = np.arange(0, 21)
f_simple = P_ci * (1 + r_ci * years_ci)
f_compound = P_ci * (1 + r_ci) ** years_ci
f_continuous = P_ci * np.exp(r_ci * years_ci)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(years_ci, f_simple / 1000, "r--", linewidth=2, label="Simple Interest")
ax.plot(years_ci, f_compound / 1000, "b-", linewidth=2, label="Compound Interest (annual)")
ax.plot(years_ci, f_continuous / 1000, "g:", linewidth=2, label="Continuous Compounding")
gap_ci = f_compound[-1] - f_simple[-1]
ax.annotate(f"Gap: ${gap_ci/1000:.1f}k",
            xy=(20, f_compound[-1] / 1000), xytext=(16, f_compound[-1] / 1000 + 8),
            fontsize=10, color="blue", arrowprops=dict(arrowstyle="->", color="blue"))
ax.plot(20, f_simple[-1] / 1000, "rs", markersize=8, zorder=5)
ax.plot(20, f_compound[-1] / 1000, "bo", markersize=8, zorder=5)
ax.plot(20, f_continuous[-1] / 1000, "g^", markersize=8, zorder=5)
ax.text(20.3, f_simple[-1] / 1000, f"${f_simple[-1]/1000:.0f}k", fontsize=9, color="red", va="center")
ax.text(20.3, f_compound[-1] / 1000, f"${f_compound[-1]/1000:.0f}k", fontsize=9, color="blue", va="center")
ax.text(20.3, f_continuous[-1] / 1000 + 2, f"${f_continuous[-1]/1000:.0f}k", fontsize=9, color="green", va="center")
ax.set_xlabel("Years"); ax.set_ylabel("Future Value ($k)")
ax.set_title("Growth of $100,000 at 5% Annual Rate: Simple vs Compound vs Continuous")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, 22)
fig.tight_layout()
save(fig, "ch19_compound_interest.png")

# --- 19.8 Depreciation Methods Comparison ---
cost_dep = 120_000; salvage_dep = 10_000; life_dep = 10
years_dep = np.arange(0, life_dep + 1)
d_sl = (cost_dep - salvage_dep) / life_dep
bv_sl = np.maximum(cost_dep - d_sl * years_dep, salvage_dep)

d_rate = 2.0 / life_dep
bv_ddb = np.zeros(life_dep + 1); bv_ddb[0] = cost_dep
for yr in range(1, life_dep + 1):
    dep = d_rate * bv_ddb[yr - 1]
    rem = life_dep - yr + 1
    sl_dep = (bv_ddb[yr - 1] - salvage_dep) / rem if rem > 0 else 0
    dep = max(dep, sl_dep)
    bv_ddb[yr] = max(bv_ddb[yr - 1] - dep, salvage_dep)

macrs_pct = [10.00, 18.00, 14.40, 11.52, 9.22, 7.37, 6.55, 6.55, 6.56, 6.55, 3.28]
bv_macrs = np.zeros(life_dep + 1); bv_macrs[0] = cost_dep
for yr in range(1, life_dep + 1):
    dep = cost_dep * macrs_pct[yr - 1] / 100 if yr - 1 < len(macrs_pct) else 0
    bv_macrs[yr] = max(bv_macrs[yr - 1] - dep, 0)

total_units = 100_000
annual_units = [8000, 12000, 14000, 11000, 10000, 9000, 8000, 10000, 9000, 9000]
d_unit = (cost_dep - salvage_dep) / total_units
bv_uop = np.zeros(life_dep + 1); bv_uop[0] = cost_dep
for yr in range(1, life_dep + 1):
    dep = d_unit * annual_units[yr - 1]
    bv_uop[yr] = max(bv_uop[yr - 1] - dep, salvage_dep)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(years_dep, bv_sl / 1000, "b-", linewidth=2, marker="o", markersize=5, label="Straight-Line")
ax.plot(years_dep, bv_ddb / 1000, "r--", linewidth=2, marker="s", markersize=5, label="200% DDB")
ax.plot(years_dep, bv_macrs / 1000, color="orange", linestyle="-.", linewidth=2, marker="^", markersize=5, label="MACRS 10-year")
ax.plot(years_dep, bv_uop / 1000, "g:", linewidth=2, marker="d", markersize=5, label="Units-of-Production")
ax.axhline(y=salvage_dep / 1000, color="gray", linestyle=":", alpha=0.5, label=f"Salvage = ${salvage_dep/1000:.0f}k")
ax.set_xlabel("Year"); ax.set_ylabel("Book Value ($k)")
ax.set_title(f"Depreciation Methods: $120,000 Asset over {life_dep}-Year Life")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xlim(0, life_dep); ax.set_ylim(0, 130)
fig.tight_layout()
save(fig, "ch19_depreciation.png")

# --- 19.13.2 NPV Sensitivity ---
initial_cost_npv = 1_000_000; PA_factor = 8.5595
savings_range = np.linspace(80_000, 240_000, 300)
npv_range = -initial_cost_npv + savings_range * PA_factor
breakeven_s = initial_cost_npv / PA_factor

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(savings_range / 1000, npv_range / 1000, "b-", linewidth=2)
ax.fill_between(savings_range / 1000, npv_range / 1000, 0, where=(npv_range >= 0),
                alpha=0.1, color="green", label="NPV > 0 (viable)")
ax.fill_between(savings_range / 1000, npv_range / 1000, 0, where=(npv_range < 0),
                alpha=0.1, color="red", label="NPV < 0 (not viable)")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.plot(breakeven_s / 1000, 0, "ko", markersize=10, zorder=5)
ax.annotate(f"Breakeven: ${breakeven_s/1000:.1f}k/yr",
            xy=(breakeven_s / 1000, 0), xytext=(breakeven_s / 1000 - 15, -120),
            fontsize=10, color="black", arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
for s, label, color in [(128_000, "\u221220%", "red"), (160_000, "Base", "blue"), (192_000, "+20%", "green")]:
    n = -initial_cost_npv + s * PA_factor
    ax.axvline(x=s / 1000, color=color, linestyle="--", alpha=0.5)
    ax.plot(s / 1000, n / 1000, "o", color=color, markersize=8, zorder=5)
    ax.annotate(f"{label}\nNPV = ${n/1000:.0f}k", xy=(s / 1000, n / 1000),
                xytext=(s / 1000 + 3, n / 1000 + 30), fontsize=9, color=color)
ax.set_xlabel("Annual Savings ($k)"); ax.set_ylabel("NPV ($k)")
ax.set_title("NPV Sensitivity: $1M Solar Carport Project (MARR = 8%, 15-year life)")
ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "ch19_sensitivity.png")


# ============================================================
# Appendix A: Phasors
# ============================================================
print("Appendix A: Phasors")

# --- A.3.1 Polar Form ---
Z_p = -3+4j
fig, ax = plt.subplots(figsize=(7, 7))
ax.axhline(y=0, color="k", linewidth=0.5); ax.axvline(x=0, color="k", linewidth=0.5)
ax.annotate("", xy=(Z_p.real, Z_p.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
ax.plot(Z_p.real, Z_p.imag, "bo", markersize=10, zorder=5)
ax.annotate(f"Z = {Z_p.real:.0f} + j{Z_p.imag:.0f}\n|Z| = {abs(Z_p):.0f}, θ = {np.degrees(np.angle(Z_p)):.1f}°",
            xy=(Z_p.real, Z_p.imag), xytext=(Z_p.real-2.5, Z_p.imag+0.5), fontsize=11, color="blue")
ax.plot([Z_p.real, Z_p.real], [0, Z_p.imag], "r--", linewidth=1, alpha=0.6)
ax.plot([0, Z_p.real], [Z_p.imag, Z_p.imag], "r--", linewidth=1, alpha=0.6)
theta_a=np.linspace(0, np.angle(Z_p), 50)
ax.plot(1.5*np.cos(theta_a), 1.5*np.sin(theta_a), "g-", linewidth=1.5)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_xlabel("Real"); ax.set_ylabel("Imaginary")
ax.set_title("Complex Plane: Z = −3 + j4"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "app_a_polar_form.png")

# --- A.4.3 Phasor Addition ---
V1=100+0j; V2=0+60j; Vt=V1+V2
fig, ax = plt.subplots(figsize=(8, 7))
ax.axhline(y=0, color="k", linewidth=0.5); ax.axvline(x=0, color="k", linewidth=0.5)
ax.annotate("", xy=(V1.real, V1.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
ax.annotate("V₁ = 100∠0° V", xy=(50, -8), fontsize=11, color="blue")
ax.annotate("", xy=(Vt.real, Vt.imag), xytext=(V1.real, V1.imag), arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
ax.annotate("V₂ = 60∠90° V", xy=(V1.real+3, V1.imag+25), fontsize=11, color="red")
ax.annotate("", xy=(Vt.real, Vt.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
ax.annotate(f"V_total = {abs(Vt):.1f}∠{np.degrees(np.angle(Vt)):.1f}° V",
            xy=(Vt.real/2-15, Vt.imag/2+5), fontsize=11, color="green", fontweight="bold")
ax.set_xlim(-20, 130); ax.set_ylim(-20, 80); ax.set_xlabel("Real (V)"); ax.set_ylabel("Imaginary (V)")
ax.set_title("Phasor Addition: V₁ + V₂ = V_total"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "app_a_phasor_addition.png")

# --- A.4.4 Phasor Diagram ---
I_ph=5+0j; R_pd=30; XL_pd=40
VR=I_ph*R_pd; VL=I_ph*1j*XL_pd; Vtot=VR+VL
fig, ax = plt.subplots(figsize=(8, 7))
ax.axhline(y=0, color="k", linewidth=0.5); ax.axvline(x=0, color="k", linewidth=0.5)
ax.annotate("", xy=(I_ph.real*20, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="orange", lw=2))
ax.annotate("I = 5∠0° A", xy=(I_ph.real*20/2, -18), fontsize=10, color="orange")
ax.annotate("", xy=(VR.real, VR.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
ax.annotate("V_R = 150∠0° V", xy=(VR.real/2+15, 8), fontsize=10, color="blue", ha="center")
ax.annotate("", xy=(Vtot.real, Vtot.imag), xytext=(VR.real, VR.imag), arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
ax.annotate("V_L = 200∠90° V", xy=(VR.real+5, VR.imag+VL.imag/2), fontsize=10, color="red")
ax.annotate("", xy=(Vtot.real, Vtot.imag), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
ax.annotate(f"V_total = {abs(Vtot):.0f}∠{np.degrees(np.angle(Vtot)):.1f}° V",
            xy=(Vtot.real/2-30, Vtot.imag/2+10), fontsize=11, color="green", fontweight="bold")
ang=np.degrees(np.angle(Vtot)); theta_a=np.linspace(0, np.radians(ang), 50)
ax.plot(40*np.cos(theta_a), 40*np.sin(theta_a), "g-", linewidth=1.5)
ax.annotate(f"θ = {ang:.1f}°", xy=(45, 15), fontsize=10, color="green")
ax.set_xlim(-30, 200); ax.set_ylim(-30, 230); ax.set_xlabel("Real (V)"); ax.set_ylabel("Imaginary (V)")
ax.set_title("Phasor Diagram: Series RL Circuit"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "app_a_phasor_diagram.png")

# --- A.5.3 Power Triangle ---
S_m=2400; S_a=25; P=S_m*np.cos(np.radians(S_a)); Q=S_m*np.sin(np.radians(S_a))
fig, ax = plt.subplots(figsize=(8, 6))
ax.annotate("", xy=(P, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
ax.annotate(f"P = {P:.0f} W", xy=(P/2, -80), fontsize=12, color="blue", ha="center")
ax.annotate("", xy=(P, Q), xytext=(P, 0), arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
ax.annotate(f"Q = {Q:.0f} VAR", xy=(P+50, Q/2), fontsize=12, color="red")
ax.annotate("", xy=(P, Q), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
ax.annotate(f"S = {S_m:.0f} VA", xy=(P/2-200, Q/2+50), fontsize=12, color="green", fontweight="bold")
theta_a=np.linspace(0, np.radians(S_a), 50)
ax.plot(400*np.cos(theta_a), 400*np.sin(theta_a), "g-", linewidth=1.5)
ax.annotate(f"φ = {S_a}°\npf = {np.cos(np.radians(S_a)):.3f}", xy=(450, 80), fontsize=10, color="green")
ax.plot([P-60, P-60, P], [0, 60, 60], "k-", linewidth=1)
ax.set_xlim(-200, 2700); ax.set_ylim(-200, 1400); ax.set_xlabel("Real Power (W)")
ax.set_ylabel("Reactive Power (VAR)"); ax.set_title("Power Triangle: S = P + jQ"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "app_a_power_triangle.png")


# ============================================================
# Appendix B: atan vs atan2
# ============================================================
print("Appendix B: atan vs atan2")

# --- B.1.2 Quadrant Ambiguity ---
Z1q=3+4j; Z2q=-3-4j
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
for ax in (ax1, ax2):
    ax.axhline(y=0, color="k", linewidth=0.5); ax.axvline(x=0, color="k", linewidth=0.5)
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

ax1.set_title("arctan(b/a) — WRONG for Z₂")
ax1.annotate("", xy=(3, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2))
ax1.plot(3, 4, "bo", markersize=8); ax1.annotate("Z₁ = 3+j4\narctan = 53.1° ✓", xy=(3.5, 5), fontsize=9, color="blue")
ax1.annotate("", xy=(-3, -4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
ax1.plot(-3, -4, "ro", markersize=8); ax1.annotate("Z₂ = −3−j4\narctan = 53.1° ✗", xy=(-5.5, -5.5), fontsize=9, color="red")
theta_w=np.linspace(0, np.radians(53.13), 30)
ax1.plot(1.5*np.cos(theta_w), 1.5*np.sin(theta_w), "r--", linewidth=1.5)
ax1.annotate("Both → 53.1°!", xy=(1.2, 1.2), fontsize=10, color="red", fontweight="bold")
ax1.set_xlabel("Real"); ax1.set_ylabel("Imaginary")

ax2.set_title("atan2(b, a) — CORRECT for both")
ax2.annotate("", xy=(3, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="blue", lw=2))
ax2.plot(3, 4, "bo", markersize=8); ax2.annotate("Z₁: atan2 = 53.1° ✓", xy=(3.5, 5), fontsize=9, color="blue")
ax2.annotate("", xy=(-3, -4), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
ax2.plot(-3, -4, "ro", markersize=8); ax2.annotate("Z₂: atan2 = −126.9° ✓", xy=(-5.5, -5.5), fontsize=9, color="red")
theta_z1=np.linspace(0, np.arctan2(4, 3), 30)
ax2.plot(1.5*np.cos(theta_z1), 1.5*np.sin(theta_z1), "b-", linewidth=1.5)
theta_z2=np.linspace(0, np.arctan2(-4, -3), 30)
ax2.plot(1.5*np.cos(theta_z2), 1.5*np.sin(theta_z2), "r-", linewidth=1.5)
ax2.set_xlabel("Real"); ax2.set_ylabel("Imaginary")
fig.tight_layout()
save(fig, "app_b_quadrant_ambiguity.png")

# --- B.2.1 Four quadrants ---
pts=[(3,4,"Q-I","blue"),(-3,4,"Q-II","green"),(-3,-4,"Q-III","red"),(3,-4,"Q-IV","orange")]
fig, ax = plt.subplots(figsize=(7, 7))
ax.axhline(y=0, color="k", linewidth=0.5); ax.axvline(x=0, color="k", linewidth=0.5)
for a, b, lab, col in pts:
    ang=np.degrees(np.arctan2(b, a)); at_only=np.degrees(np.arctan(b/a))
    ax.annotate("", xy=(a, b), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax.plot(a, b, "o", color=col, markersize=10, zorder=5)
    ox=0.5 if a>0 else -3.5; oy=0.5 if b>0 else -1.0
    ax.annotate(f"{lab}: ({a},{b})\natan2={ang:.1f}°\narctan={at_only:.1f}°",
                xy=(a+ox, b+oy), fontsize=9, color=col)
    ta=np.linspace(0, np.radians(ang), 50)
    ax.plot(1.5*np.cos(ta), 1.5*np.sin(ta), color=col, linewidth=1.5, alpha=0.7)
ax.set_xlim(-7, 7); ax.set_ylim(-7, 7); ax.set_xlabel("Real"); ax.set_ylabel("Imaginary")
ax.set_title("atan2 Returns Correct Angles in All Four Quadrants"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "app_b_four_quadrants.png")

# --- B.2.3 Axis points ---
aps=[(1,0,"Pure R: 0°","blue"),(0,1,"Pure L: +90°","red"),(-1,0,"−R: ±180°","green"),(0,-1,"Pure C: −90°","orange")]
fig, ax = plt.subplots(figsize=(7, 7))
ax.axhline(y=0, color="k", linewidth=0.5); ax.axvline(x=0, color="k", linewidth=0.5)
tc=np.linspace(0, 2*np.pi, 200); ax.plot(np.cos(tc), np.sin(tc), "k-", alpha=0.15, linewidth=1)
for a, b, lab, col in aps:
    ang=np.degrees(np.arctan2(b, a))
    ax.annotate("", xy=(a, b), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=3))
    ax.plot(a, b, "o", color=col, markersize=12, zorder=5)
    if a==1: ax.annotate(f"{lab}\natan2={ang:.0f}°", xy=(a+0.1, b+0.1), fontsize=10, color=col)
    elif a==-1: ax.annotate(f"{lab}\natan2={ang:.0f}°", xy=(a-0.1, b+0.15), fontsize=10, color=col, ha="right")
    elif b==1: ax.annotate(f"{lab}\natan2={ang:.0f}°", xy=(a+0.1, b+0.1), fontsize=10, color=col)
    else: ax.annotate(f"{lab}\natan2={ang:.0f}°", xy=(a+0.1, b-0.3), fontsize=10, color=col)
ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_xlabel("Real"); ax.set_ylabel("Imaginary")
ax.set_title("atan2 Special Cases: Points on the Axes"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
fig.tight_layout()
save(fig, "app_b_axis_points.png")


# ============================================================
# Appendix C: Decibels
# ============================================================
print("Appendix C: Decibels")

# --- C.1.3 dB Scale ---
dBr=np.linspace(-20, 40, 500); pr=10**(dBr/10); vr=10**(dBr/20)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.semilogy(dBr, pr, "b-", linewidth=2)
for db, r in [(-20,0.01),(-10,0.1),(-3,0.5),(0,1),(3,2),(10,10),(20,100),(30,1000)]:
    ax1.plot(db, r, "ro", markersize=6, zorder=5)
    ax1.annotate(f"{db}dB→{r}×", xy=(db, r), xytext=(db+1.5, r*1.5), fontsize=8, color="red",
                 arrowprops=dict(arrowstyle="->", color="red", lw=0.5))
ax1.set_xlabel("Decibels (dB)"); ax1.set_ylabel("Power Ratio"); ax1.set_title("Power: 10^(dB/10)")
ax1.grid(True, alpha=0.3, which="both"); ax1.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
ax2.semilogy(dBr, vr, "r-", linewidth=2)
for db, r in [(-20,0.1),(-6,0.5),(0,1),(6,2),(20,10),(40,100)]:
    ax2.plot(db, r, "bo", markersize=6, zorder=5)
    ax2.annotate(f"{db}dB→{r}×", xy=(db, r), xytext=(db+1.5, r*1.3), fontsize=8, color="blue",
                 arrowprops=dict(arrowstyle="->", color="blue", lw=0.5))
ax2.set_xlabel("Decibels (dB)"); ax2.set_ylabel("Voltage Ratio"); ax2.set_title("Voltage: 10^(dB/20)")
ax2.grid(True, alpha=0.3, which="both"); ax2.axhline(y=1, color="gray", linestyle=":", alpha=0.5)
fig.tight_layout()
save(fig, "app_c_db_scale.png")

# --- C.3.1 Link Budget ---
stages=["Laser\nSource","Conn\n1","Fiber\n(20 km)","Conn\n2","Received"]
gains=[3,-0.5,-6,-0.5,0]; cum=[3]
for g in gains[1:-1]: cum.append(cum[-1]+g)
cum.append(cum[-1]); sens=-28

fig, ax = plt.subplots(figsize=(10, 6))
colors_lb=["green","orange","red","orange","blue"]
ax.bar(range(len(stages)), cum, color=colors_lb, alpha=0.7, edgecolor="black", linewidth=1.2)
for i, v in enumerate(cum):
    off=1 if v>=0 else -1.5
    ax.text(i, v+off, f"{v:.1f} dBm", ha="center", fontsize=11, fontweight="bold")
ax.axhline(y=sens, color="red", linestyle="--", linewidth=2, label=f"Sensitivity: {sens} dBm")
ax.annotate("", xy=(4.4, cum[-1]), xytext=(4.4, sens), arrowprops=dict(arrowstyle="<->", color="green", lw=2))
ax.text(4.55, (cum[-1]+sens)/2, f"Margin\n{cum[-1]-sens:.0f} dB", fontsize=11, color="green", fontweight="bold")
ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages)
ax.set_ylabel("Power Level (dBm)"); ax.set_title("Fiber Optic Link Budget"); ax.legend(loc="lower left")
ax.grid(True, alpha=0.3, axis="y"); ax.set_ylim(-35, 10)
fig.tight_layout()
save(fig, "app_c_link_budget.png")

# --- C.4.1 Bode plot ---
A_OL_dB=100; GBW=10e6; A_OL=10**(A_OL_dB/20); fp=GBW/A_OL
A_CL_dB=40; A_CL=10**(A_CL_dB/20); BW_CL=GBW/A_CL
fb=np.logspace(0, 8, 1000)
H_OL=A_OL/(1+1j*fb/fp); g_OL=20*np.log10(np.abs(H_OL)); g_CL=np.minimum(A_CL_dB, g_OL)

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogx(fb, g_OL, "b-", linewidth=2, label="Open-loop gain")
ax.semilogx(fb, g_CL, "r-", linewidth=2, label=f"Closed-loop (A_CL = {A_CL_dB} dB)")
ax.plot(fp, A_OL_dB-3, "go", markersize=8, label=f"Dominant pole: {fp:.0f} Hz")
ax.plot(GBW, 0, "mo", markersize=8, label=f"Unity gain: {GBW/1e6:.0f} MHz")
ax.plot(BW_CL, A_CL_dB-3, "rs", markersize=8, label=f"CL bandwidth: {BW_CL/1e3:.0f} kHz")
ax.annotate("−20 dB/decade", xy=(1e4, 60), fontsize=10, color="blue", rotation=-20)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Gain (dB)")
ax.set_title("Op-Amp Bode Plot"); ax.set_xlim(1, 1e8); ax.set_ylim(-10, 110)
ax.legend(loc="upper right", fontsize=9); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
save(fig, "app_c_bode_plot.png")

# ============================================================
# Chapter 4: Control Systems
# ============================================================
print("Chapter 4: Control Systems")

# --- 4.5 Bode Plot: Second-Order System ---
wn = 10.0  # natural frequency (rad/s)
zeta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

w = np.logspace(-1, 2.5, 1000)

fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for zeta, color in zip(zeta_values, colors):
    H = wn**2 / (wn**2 - w**2 + 1j * 2 * zeta * wn * w)
    mag_db = 20 * np.log10(np.abs(H))
    phase_deg = np.degrees(np.angle(H))
    ax_mag.plot(w, mag_db, color=color, linewidth=2, label=f"ζ = {zeta}")
    ax_phase.plot(w, phase_deg, color=color, linewidth=2, label=f"ζ = {zeta}")

for zeta, color in zip([0.1, 0.3, 0.5], colors[:3]):
    if zeta < 1 / np.sqrt(2):
        w_peak = wn * np.sqrt(1 - 2 * zeta**2)
        H_peak = wn**2 / (wn**2 - w_peak**2 + 1j * 2 * zeta * wn * w_peak)
        peak_db = 20 * np.log10(np.abs(H_peak))
        ax_mag.plot(w_peak, peak_db, "o", color=color, markersize=6, zorder=5)

ax_mag.set_ylabel("Magnitude (dB)")
ax_mag.set_title(f"Bode Plot: Second-Order System (ωn = {wn:.0f} rad/s)")
ax_mag.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
ax_mag.axvline(x=wn, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
ax_mag.text(wn * 1.05, -35, f"ωn = {wn:.0f}", fontsize=9, color="gray", va="top")
ax_mag.legend(fontsize=9, loc="lower left")
ax_mag.grid(True, alpha=0.3, which="both")
ax_mag.set_ylim(-40, 25)

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

fig.tight_layout()
save(fig, "ch04_bode_plot.png")

# --- 4.6 Step Response: Second-Order System ---
wn_step = 10.0
zeta_step = [0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
colors_step = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b"]

t = np.linspace(0, 3, 2000)

fig, ax = plt.subplots(figsize=(10, 6))

for zeta, color in zip(zeta_step, colors_step):
    if zeta < 1.0:
        wd = wn_step * np.sqrt(1 - zeta**2)
        phi = np.arctan2(np.sqrt(1 - zeta**2), zeta)
        y = 1 - (np.exp(-zeta * wn_step * t) / np.sqrt(1 - zeta**2)) * np.sin(wd * t + phi)
    elif zeta == 1.0:
        y = 1 - (1 + wn_step * t) * np.exp(-wn_step * t)
    else:
        s1 = -zeta * wn_step + wn_step * np.sqrt(zeta**2 - 1)
        s2 = -zeta * wn_step - wn_step * np.sqrt(zeta**2 - 1)
        y = 1 + (s1 * np.exp(s2 * t) - s2 * np.exp(s1 * t)) / (s2 - s1)
    ax.plot(t, y, color=color, linewidth=2, label=f"ζ = {zeta}")

ax.axhline(y=1.02, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax.axhline(y=0.98, color="gray", linewidth=1, linestyle="--", alpha=0.6)
ax.fill_between(t, 0.98, 1.02, color="gray", alpha=0.08)
ax.text(2.85, 1.035, "±2% band", fontsize=9, color="gray", ha="right")
ax.axhline(y=1.0, color="black", linewidth=0.5, linestyle=":")

zeta_ann = 0.3
wd_ann = wn_step * np.sqrt(1 - zeta_ann**2)
t_peak = np.pi / wd_ann
Mp = np.exp(-zeta_ann * np.pi / np.sqrt(1 - zeta_ann**2))
y_peak = 1 + Mp

ax.plot(t_peak, y_peak, "o", color="#ff7f0e", markersize=8, zorder=5)
ax.annotate(f"Overshoot = {Mp*100:.1f}%\nt_p = {t_peak:.3f} s",
             xy=(t_peak, y_peak), xytext=(t_peak + 0.25, y_peak + 0.05),
             fontsize=9, color="#ff7f0e",
             arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

t_settle = 4 / (zeta_ann * wn_step)
ax.axvline(x=t_settle, color="#ff7f0e", linewidth=1, linestyle=":", alpha=0.6)
ax.annotate(f"t_s = {t_settle:.2f} s (ζ=0.3)",
             xy=(t_settle, 0.5), xytext=(t_settle + 0.15, 0.45),
             fontsize=9, color="#ff7f0e",
             arrowprops=dict(arrowstyle="->", color="#ff7f0e"),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.set_xlabel("Time (s)")
ax.set_ylabel("Response c(t)")
ax.set_title(f"Unit Step Response: Second-Order System (ωn = {wn_step:.0f} rad/s)")
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.85)
fig.tight_layout()
save(fig, "ch04_step_response.png")

# --- 4.7 Root Locus: G(s) = K / [s(s+2)(s+5)] ---
K_values = np.linspace(0, 200, 5000)
roots_all = np.array([np.roots([1, 7, 10, K]) for K in K_values])

fig, ax = plt.subplots(figsize=(10, 8))

ax.axvspan(-12, 0, color="green", alpha=0.04)
ax.text(-5.5, 7.5, "Stable Region\n(LHP)", fontsize=11, color="green",
         ha="center", alpha=0.6, fontstyle="italic")
ax.text(2.5, 7.5, "Unstable\n(RHP)", fontsize=11, color="red",
         ha="center", alpha=0.5, fontstyle="italic")

ax.axvline(x=0, color="black", linewidth=1.0)
ax.axhline(y=0, color="black", linewidth=0.5)

for i in range(3):
    real_parts = roots_all[:, i].real
    imag_parts = roots_all[:, i].imag
    ax.plot(real_parts, imag_parts, "b.", markersize=0.5)

poles = [0, -2, -5]
for p in poles:
    ax.plot(p, 0, "kx", markersize=12, markeredgewidth=2.5, zorder=10)
ax.text(0.3, -0.5, "0", fontsize=10, color="black")
ax.text(-1.7, -0.7, "−2", fontsize=10, color="black")
ax.text(-4.7, -0.7, "−5", fontsize=10, color="black")

K_crit = 70
roots_crit = np.roots([1, 7, 10, K_crit])
for r in roots_crit:
    if abs(r.real) < 0.1:
        ax.plot(r.real, r.imag, "r*", markersize=15, zorder=10)

w_cross = np.sqrt(10)
ax.annotate(f"jω crossing at K = {K_crit}\nω = ±√10 ≈ ±{w_cross:.2f}",
             xy=(0, w_cross), xytext=(2, w_cross + 1.5),
             fontsize=9, color="red",
             arrowprops=dict(arrowstyle="->", color="red"),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.annotate("",
             xy=(0, -w_cross), xytext=(2, -w_cross - 1.5),
             arrowprops=dict(arrowstyle="->", color="red"))

s_break = (-14 + np.sqrt(196 - 120)) / 6
K_break = -(s_break**3 + 7 * s_break**2 + 10 * s_break)
ax.plot(s_break, 0, "gD", markersize=8, zorder=10)
ax.annotate(f"Breakaway\nσ = {s_break:.2f}, K = {K_break:.1f}",
             xy=(s_break, 0), xytext=(s_break - 1.5, 2.5),
             fontsize=9, color="green",
             arrowprops=dict(arrowstyle="->", color="green"),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

ax.plot([], [], "kx", markersize=10, markeredgewidth=2.5, label="Open-loop poles")
ax.plot([], [], "r*", markersize=12, label=f"jω axis crossing (K = {K_crit})")
ax.plot([], [], "gD", markersize=8, label="Breakaway point")
ax.plot([], [], "b-", linewidth=2, label="Root locus")

ax.set_xlabel("Real Axis (σ)")
ax.set_ylabel("Imaginary Axis (jω)")
ax.set_title("Root Locus: G(s) = K / [s(s+2)(s+5)]")
ax.legend(fontsize=9, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(-12, 6)
ax.set_ylim(-9, 9)
ax.set_aspect("equal")
fig.tight_layout()
save(fig, "ch04_root_locus.png")


# ============================================================
# Chapter 5: Embedded Systems
# ============================================================
print("Chapter 5: Embedded Systems")

# --- 5.4.2 SPI Mode 0 Timing Diagram ---
_byte_val_g = 0xA5
_bits_mosi_g = [(_byte_val_g >> (7 - i)) & 1 for i in range(8)]
_n_bits_g = 8
_t_per_bit_g = 1.0
_t_pre_g = 0.75
_t_total_g = _t_pre_g + _n_bits_g * _t_per_bit_g + 0.75
_t_g = np.linspace(0, _t_total_g, 2000)

_sclk_g = np.zeros_like(_t_g)
for _i_g in range(_n_bits_g):
    _ts_g = _t_pre_g + _i_g * _t_per_bit_g
    _tm_g = _ts_g + 0.5 * _t_per_bit_g
    _sclk_g[(_t_g >= _tm_g) & (_t_g < _ts_g + _t_per_bit_g)] = 1.0

_cs_g = np.where((_t_g >= _t_pre_g) & (_t_g < _t_pre_g + _n_bits_g * _t_per_bit_g), 0.0, 1.0)

_mosi_g = np.zeros_like(_t_g)
for _i_g, _b_g in enumerate(_bits_mosi_g):
    _ts_g = _t_pre_g + _i_g * _t_per_bit_g
    _mosi_g[(_t_g >= _ts_g) & (_t_g < _ts_g + _t_per_bit_g)] = float(_b_g)

_bits_miso_g = [(0x3C >> (7 - i)) & 1 for i in range(8)]
_miso_g = np.zeros_like(_t_g)
for _i_g, _b_g in enumerate(_bits_miso_g):
    _ts_g = _t_pre_g + _i_g * _t_per_bit_g
    _miso_g[(_t_g >= _ts_g) & (_t_g < _ts_g + _t_per_bit_g)] = float(_b_g)

_sig_colors_g = ["#2266CC", "#AA2222", "#228822", "#996600"]
_sig_labels_g = ["CS (active-low)", "SCLK", "MOSI (0xA5)", "MISO (0x3C)"]
_sig_waves_g  = [_cs_g, _sclk_g, _mosi_g, _miso_g]

fig, _axes_g = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
for _ax_g, _lbl_g, _sig_g, _sc_g in zip(_axes_g, _sig_labels_g, _sig_waves_g, _sig_colors_g):
    _ax_g.plot(_t_g, _sig_g, color=_sc_g, linewidth=2)
    _ax_g.fill_between(_t_g, 0, _sig_g, alpha=0.12, color=_sc_g)
    _ax_g.set_ylabel(_lbl_g, fontsize=9, labelpad=4)
    _ax_g.set_ylim(-0.3, 1.4)
    _ax_g.set_yticks([0, 1])
    _ax_g.set_yticklabels(["Lo", "Hi"], fontsize=8)
    _ax_g.grid(True, alpha=0.2, axis="x")
    _ax_g.spines["top"].set_visible(False)
    _ax_g.spines["right"].set_visible(False)

_ax_mosi_g = _axes_g[2]
for _i_g, _b_g in enumerate(_bits_mosi_g):
    _xc_g = _t_pre_g + (_i_g + 0.5) * _t_per_bit_g
    _ax_mosi_g.text(_xc_g, 1.15, str(_b_g), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=_sig_colors_g[2])

for _i_g in range(_n_bits_g):
    _tr_g = _t_pre_g + _i_g * _t_per_bit_g + 0.5 * _t_per_bit_g
    _axes_g[1].axvline(x=_tr_g, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
    _axes_g[2].axvline(x=_tr_g, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

_axes_g[-1].set_xlabel("Time (bit periods, 1 unit = 1 / f_SCLK)", fontsize=10)
_axes_g[0].set_title(
    "SPI Mode 0 Timing Diagram — 8-bit Transfer\n"
    "(MOSI: 0xA5 = 1010 0101, MISO: 0x3C = 0011 1100)",
    fontsize=11,
)
_axes_g[0].annotate(
    "", xy=(_t_pre_g + _n_bits_g * _t_per_bit_g, 0.5), xytext=(_t_pre_g, 0.5),
    arrowprops=dict(arrowstyle="<->", color="navy", lw=1.5),
)
_axes_g[0].text(_t_pre_g + _n_bits_g * 0.5 * _t_per_bit_g, 0.65,
                "8 clock cycles (active)", ha="center", fontsize=9, color="navy")
fig.tight_layout(h_pad=0.3)
save(fig, "ch05_spi_timing.png")

# --- 5.7.1 Sleep Mode Power Budget ---
_I_active_g = 15.0      # mA
_I_sleep_g  = 20.0      # μA
_t_active_g = 50.0      # ms
_C_mAh_g    = 1000.0
_T_wake_s_g  = np.linspace(1, 600, 1000)
_T_wake_ms_g = _T_wake_s_g * 1000
_t_sleep_ms_g = np.maximum(_T_wake_ms_g - _t_active_g, 0)
_I_avg_g = (_I_active_g * _t_active_g + (_I_sleep_g / 1000) * _t_sleep_ms_g) / _T_wake_ms_g
_life_yr_g  = (_C_mAh_g / _I_avg_g) / 8760

fig, (_ax_cur_g, _ax_life_g) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
_ax_cur_g.semilogy(_T_wake_s_g, _I_avg_g, "b-", linewidth=2)
_ax_cur_g.axhline(y=_I_sleep_g / 1000, color="green", linestyle="--", alpha=0.6,
                  label=f"Sleep floor: {_I_sleep_g} μA")
for _Tex_g, _ccol_g, _clbl_g in [(60, "red", "60 s"), (5, "orange", "5 s")]:
    _tsl_g = _Tex_g * 1000 - _t_active_g
    _Iex_g = (_I_active_g * _t_active_g + (_I_sleep_g / 1000) * _tsl_g) / (_Tex_g * 1000)
    _ax_cur_g.plot(_Tex_g, _Iex_g, "o", color=_ccol_g, markersize=9, zorder=5)
    _ax_cur_g.annotate(
        f"T = {_Tex_g} s\nI_avg = {_Iex_g*1000:.0f} μA",
        xy=(_Tex_g, _Iex_g), xytext=(_Tex_g + 30, _Iex_g * 3),
        fontsize=9, color=_ccol_g,
        arrowprops=dict(arrowstyle="->", color=_ccol_g),
    )
_ax_cur_g.set_ylabel("Average Current (mA)", fontsize=10)
_ax_cur_g.set_title(
    "Battery-Powered Sensor: Average Current vs Wake Interval\n"
    f"(I_active = {_I_active_g} mA for {_t_active_g} ms, I_sleep = {_I_sleep_g} μA)",
    fontsize=11,
)
_ax_cur_g.legend(fontsize=9)
_ax_cur_g.grid(True, alpha=0.3, which="both")

_ax_life_g.plot(_T_wake_s_g, _life_yr_g, "r-", linewidth=2, label="Battery life")
for _Tex_g, _ccol_g, _clbl_g in [(60, "red", "60 s"), (5, "orange", "5 s")]:
    _tsl_g = _Tex_g * 1000 - _t_active_g
    _Iex_g = (_I_active_g * _t_active_g + (_I_sleep_g / 1000) * _tsl_g) / (_Tex_g * 1000)
    _lex_g = (_C_mAh_g / _Iex_g) / 8760
    _ax_life_g.plot(_Tex_g, _lex_g, "o", color=_ccol_g, markersize=9, zorder=5)
    _ax_life_g.annotate(
        f"T = {_Tex_g} s\n{_lex_g:.2f} yr",
        xy=(_Tex_g, _lex_g), xytext=(_Tex_g + 30, _lex_g * 1.5),
        fontsize=9, color=_ccol_g,
        arrowprops=dict(arrowstyle="->", color=_ccol_g),
    )
_ax_life_g.set_xlabel("Wake Interval (s)", fontsize=10)
_ax_life_g.set_ylabel("Battery Life (years)", fontsize=10)
_ax_life_g.set_title(f"Battery Life vs Wake Interval (C = {_C_mAh_g} mAh)", fontsize=11)
_ax_life_g.legend(fontsize=9)
_ax_life_g.grid(True, alpha=0.3)
_ax_life_g.set_xlim(0, 600)
_ax_life_g.set_ylim(0)
fig.tight_layout()
save(fig, "ch05_sleep_power.png")

# --- 5.7.2 Low-Power Frequency Scaling ---
_I_high_g = 30.0    # mA at 168 MHz
_I_mid_g  =  8.0    # mA at 48 MHz
_I_stop_g =  0.020  # mA Stop mode
_f_compute_g = 0.10
_idle_frac_g = np.linspace(0, 0.90, 500)
_comm_frac_g = 1.0 - _f_compute_g - _idle_frac_g
_I_unopt_g = np.full_like(_idle_frac_g, _I_high_g)
_I_opt_g   = (_f_compute_g * _I_high_g
              + _comm_frac_g  * _I_mid_g
              + _idle_frac_g  * _I_stop_g)
_saving_g  = (_I_unopt_g - _I_opt_g) / _I_unopt_g * 100

fig, (_axc_g, _axs_g) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
_axc_g.plot(_idle_frac_g * 100, _I_unopt_g, "r-", linewidth=2,
            label="Unoptimized (always 168 MHz)")
_axc_g.plot(_idle_frac_g * 100, _I_opt_g,   "b-", linewidth=2,
            label="Optimized (freq scaling + Stop mode)")
_idle_ex_g = 0.70
_comm_ex_g = 1.0 - _f_compute_g - _idle_ex_g
_I_ex_g    = _f_compute_g * _I_high_g + _comm_ex_g * _I_mid_g + _idle_ex_g * _I_stop_g
_axc_g.plot(_idle_ex_g * 100, _I_ex_g, "bo", markersize=10, zorder=5)
_axc_g.annotate(
    f"Example §5.7.2\n70% idle → {_I_ex_g:.2f} mA\n(vs {_I_high_g:.0f} mA unoptimized)",
    xy=(_idle_ex_g * 100, _I_ex_g),
    xytext=(_idle_ex_g * 100 - 35, _I_ex_g + 6),
    fontsize=9, color="blue",
    arrowprops=dict(arrowstyle="->", color="blue"),
)
_axc_g.set_ylabel("Average Current (mA)", fontsize=10)
_axc_g.set_title(
    "IoT Device Average Current: Optimized vs Unoptimized\n"
    "(I_compute = 30 mA @ 168 MHz, I_comm = 8 mA @ 48 MHz, I_idle = 20 μA Stop mode)",
    fontsize=11,
)
_axc_g.legend(fontsize=9)
_axc_g.grid(True, alpha=0.3)
_axc_g.set_ylim(0, 35)

_saving_ex_g = (_I_high_g - _I_ex_g) / _I_high_g * 100
_axs_g.plot(_idle_frac_g * 100, _saving_g, "g-", linewidth=2)
_axs_g.fill_between(_idle_frac_g * 100, _saving_g, alpha=0.15, color="green")
_axs_g.plot(_idle_ex_g * 100, _saving_ex_g, "go", markersize=10, zorder=5)
_axs_g.annotate(
    f"{_saving_ex_g:.1f}% saving at 70% idle",
    xy=(_idle_ex_g * 100, _saving_ex_g),
    xytext=(_idle_ex_g * 100 - 30, 68),
    fontsize=9, color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen"),
)
_axs_g.set_xlabel("Idle Fraction (%)", fontsize=10)
_axs_g.set_ylabel("Current Reduction (%)", fontsize=10)
_axs_g.set_title("Power Saving (%) from Frequency Scaling + Stop Mode", fontsize=11)
_axs_g.grid(True, alpha=0.3)
_axs_g.set_xlim(0, 90)
_axs_g.set_ylim(0, 100)
fig.tight_layout()
save(fig, "ch05_lowpower_opt.png")


# ============================================================
# Chapter 6: Digital Logic
# ============================================================
print("Chapter 6: Digital Logic")

# --- 6.1 Number System Conversions ---
fig, ax = plt.subplots(figsize=(10, 8))

n_values = 16
col_labels = ["Decimal", "Bit 3\n(8)", "Bit 2\n(4)", "Bit 1\n(2)", "Bit 0\n(1)", "Hex", "Octal"]
n_cols = len(col_labels)

color_grid = np.zeros((n_values, n_cols))
text_grid = []

for i in range(n_values):
    row_text = []
    row_text.append(str(i))
    color_grid[i, 0] = i / 15.0
    bits = [(i >> b) & 1 for b in [3, 2, 1, 0]]
    for j, b in enumerate(bits):
        row_text.append(str(b))
        color_grid[i, 1 + j] = b
    row_text.append(f"{i:X}")
    color_grid[i, 5] = i / 15.0
    row_text.append(f"{i:o}")
    color_grid[i, 6] = i / 15.0
    text_grid.append(row_text)

cmap = plt.cm.YlOrRd

ax.imshow(color_grid, cmap=cmap, aspect="auto", alpha=0.55)

for i in range(n_values):
    for j in range(n_cols):
        ax.text(j, i, text_grid[i][j], ha="center", va="center",
                 fontsize=11, fontweight="bold", color="black")

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")

ax.set_yticks(range(n_values))
ax.set_yticklabels([f"  {i}" for i in range(n_values)], fontsize=9)

for i in range(n_values + 1):
    ax.axhline(i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
for j in range(n_cols + 1):
    ax.axvline(j - 0.5, color="gray", linewidth=0.5, alpha=0.5)

ax.axvline(0.5, color="black", linewidth=1.5)
ax.axvline(4.5, color="black", linewidth=1.5)

ax.set_title("Number System Conversion Table: Decimal, Binary, Hexadecimal, Octal (0–15)",
              fontsize=12, pad=35)
fig.tight_layout()
save(fig, "ch06_number_systems.png")

# --- 6.2 Boolean Logic Gate Truth Tables ---
gates = {
    "AND":  lambda a, b: a & b,
    "OR":   lambda a, b: a | b,
    "NAND": lambda a, b: 1 - (a & b),
    "NOR":  lambda a, b: 1 - (a | b),
    "XOR":  lambda a, b: a ^ b,
    "XNOR": lambda a, b: 1 - (a ^ b),
}

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes_flat = axes.flatten()

inputs = [0, 1]

for idx, (name, func) in enumerate(gates.items()):
    ax = axes_flat[idx]
    grid = np.array([[func(a, b) for b in inputs] for a in inputs])
    colors_gate = np.zeros((*grid.shape, 3))
    for i in range(2):
        for j in range(2):
            if grid[i, j] == 1:
                colors_gate[i, j] = [0.2, 0.7, 0.3]
            else:
                colors_gate[i, j] = [0.85, 0.25, 0.25]
    ax.imshow(colors_gate, aspect="equal")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["B=0", "B=1"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["A=0", "A=1"], fontsize=10)
    ax.set_title(f"{name} Gate", fontsize=13, fontweight="bold", pad=8)
    ax.axhline(0.5, color="white", linewidth=2)
    ax.axvline(0.5, color="white", linewidth=2)

fig.suptitle("2-Input Logic Gate Truth Tables (Green = 1, Red = 0)",
              fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "ch06_logic_gates.png")

# --- 6.5.5 State Machine Timing — 3-Bit Binary Counter ---
n_cycles = 16
t_clk = []
clk = []
for i in range(n_cycles):
    t_clk.extend([i, i + 0.5, i + 0.5, i + 1.0])
    clk.extend([1, 1, 0, 0])
t_clk = np.array(t_clk)
clk = np.array(clk)

t_cnt = np.arange(0, n_cycles + 1)
counter_vals = [i % 8 for i in range(n_cycles + 1)]
q0_vals = [(v >> 0) & 1 for v in counter_vals]
q1_vals = [(v >> 1) & 1 for v in counter_vals]
q2_vals = [(v >> 2) & 1 for v in counter_vals]

signals = [
    ("CLK", t_clk, clk, "C0"),
    ("Q0 (LSB)", t_cnt, q0_vals, "C1"),
    ("Q1", t_cnt, q1_vals, "C2"),
    ("Q2 (MSB)", t_cnt, q2_vals, "C3"),
]

fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True)

for idx, (label, t_sig, sig, color) in enumerate(signals):
    ax = axes[idx]
    if label == "CLK":
        ax.plot(t_sig, np.array(sig), color=color, linewidth=2)
        ax.set_ylim(-0.2, 1.4)
    else:
        ax.step(t_sig, sig, where="post", color=color, linewidth=2)
        ax.set_ylim(-0.2, 1.4)
    ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0,
                   labelpad=60, va="center")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=9)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.fill_between(t_sig if label == "CLK" else t_cnt,
                     sig, 0, alpha=0.15, color=color, step=None if label == "CLK" else "post")

ax_bottom = axes[3]
for i in range(n_cycles):
    val = i % 8
    ax_bottom.text(i + 0.5, -0.1, f"{val:03b}", ha="center", va="top",
                    fontsize=7, color="gray", fontstyle="italic")

axes[3].set_xlabel("Clock Cycle", fontsize=11)
axes[3].set_xlim(0, n_cycles)
axes[3].set_xticks(range(n_cycles + 1))

axes[0].set_title("3-Bit Binary Counter Timing Diagram (16 Clock Cycles)",
                    fontsize=13, fontweight="bold")
fig.tight_layout()
save(fig, "ch06_counter_timing.png")

# ============================================================
# Chapter 11: Instrumentation and Measurement
# ============================================================
print("Chapter 11: Instrumentation and Measurement")

# --- 11.1.1 Accuracy and Precision ---
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

rng = np.random.default_rng(42)

scenarios = [
    ("High Accuracy\nHigh Precision", 0.0, 0.0, 0.08),
    ("High Accuracy\nLow Precision", 0.0, 0.0, 0.35),
    ("Low Accuracy\nHigh Precision", 0.5, 0.4, 0.08),
    ("Low Accuracy\nLow Precision", 0.45, -0.35, 0.35),
]

for idx, (title, cx, cy, spread) in enumerate(scenarios):
    ax = axes[idx // 2][idx % 2]
    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        circle = plt.Circle((0, 0), r, fill=False, color="gray",
                             linewidth=1, linestyle="-", alpha=0.5)
        ax.add_patch(circle)
    ring_colors = ["#FFD700", "#FFA500", "#FF6347", "#CD5C5C", "#DCDCDC"]
    for i, r in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
        ring = plt.Circle((0, 0), r, fill=True, color=ring_colors[i],
                           alpha=0.15, zorder=0)
        ax.add_patch(ring)
    ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2, zorder=3)
    n_pts = 20
    x_pts = rng.normal(cx, spread, n_pts)
    y_pts = rng.normal(cy, spread, n_pts)
    ax.scatter(x_pts, y_pts, c="blue", s=50, edgecolors="navy",
                linewidths=0.8, alpha=0.85, zorder=4)
    ax.plot(np.mean(x_pts), np.mean(y_pts), "r^", markersize=12,
             markeredgecolor="darkred", markeredgewidth=1.5, zorder=5,
             label="Mean")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)

fig.suptitle("Accuracy vs Precision: Target Analogy",
              fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, "ch11_accuracy_precision.png")

# --- 11.1.5 Wheatstone Bridge ---
R_nom = 350.0
V_ex = 10.0

dr_ratio = np.linspace(-0.10, 0.10, 500)

R_active = R_nom * (1 + dr_ratio)
v_exact = V_ex * (R_active / (R_active + R_nom) - 0.5)
v_linear = V_ex * dr_ratio / 4.0
v_error = (v_exact - v_linear) * 1000

fig, (ax_main, ax_err) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})

ax_main.plot(dr_ratio * 100, v_exact * 1000, "b-", linewidth=2.5,
              label="Exact (nonlinear)")
ax_main.plot(dr_ratio * 100, v_linear * 1000, "r--", linewidth=2,
              label="Linear approximation")
ax_main.fill_between(dr_ratio * 100, v_exact * 1000, v_linear * 1000,
                      alpha=0.2, color="orange", label="Nonlinearity error")

for pct in [-10, -5, 5, 10]:
    idx_pct = np.argmin(np.abs(dr_ratio * 100 - pct))
    err_mv = v_error[idx_pct]
    ax_main.annotate(f"{err_mv:+.2f} mV",
                      xy=(pct, v_exact[idx_pct] * 1000),
                      xytext=(pct + (2 if pct > 0 else -2), v_exact[idx_pct] * 1000 + 8),
                      fontsize=8, color="darkorange",
                      arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8))

ax_main.set_ylabel("Output Voltage (mV)", fontsize=11)
ax_main.set_title(f"Wheatstone Bridge: Exact vs Linear Response (R = {R_nom:.0f} Ω, "
                   f"V_ex = {V_ex:.0f} V)", fontsize=12, fontweight="bold")
ax_main.legend(fontsize=10, loc="upper left")
ax_main.grid(True, alpha=0.3)
ax_main.axhline(0, color="black", linewidth=0.5)
ax_main.axvline(0, color="black", linewidth=0.5)

ax_err.plot(dr_ratio * 100, v_error, "darkorange", linewidth=2)
ax_err.fill_between(dr_ratio * 100, v_error, 0, alpha=0.2, color="orange")
ax_err.set_xlabel("ΔR/R (%)", fontsize=11)
ax_err.set_ylabel("Error (mV)", fontsize=11)
ax_err.set_title("Nonlinearity Error (Exact − Linear)", fontsize=11)
ax_err.grid(True, alpha=0.3)
ax_err.axhline(0, color="black", linewidth=0.5)

fig.tight_layout()
save(fig, "ch11_wheatstone_bridge.png")

# --- 11.2.1 Thermocouple Response ---
tc_data = {
    "J": {
        "max_temp": 760,
        "color": "#1f77b4",
        "coeffs": [0.0, 5.0381187815e-02, 3.0475836930e-05,
                    -8.5681065720e-08, 1.3228195295e-10,
                    -1.7052958337e-13, 2.0948090697e-16,
                    -1.2538395336e-19, 1.5631725697e-23],
    },
    "K": {
        "max_temp": 1372,
        "color": "#ff7f0e",
        "coeffs": [-1.7600413686e-02, 3.8921204975e-02, 1.8558770032e-05,
                    -9.9457592874e-08, 3.1840945719e-10,
                    -5.6072844889e-13, 5.6075059059e-16,
                    -3.2020720003e-19, 9.7151147152e-23,
                    -1.2104721275e-26],
    },
    "T": {
        "max_temp": 400,
        "color": "#2ca02c",
        "coeffs": [0.0, 3.8748106364e-02, 3.3292227880e-05,
                    2.0618243404e-07, -2.1882256846e-09,
                    1.0996880928e-11, -3.0815758772e-14,
                    4.5479135290e-17, -2.7512901673e-20],
    },
    "E": {
        "max_temp": 1000,
        "color": "#d62728",
        "coeffs": [0.0, 5.8665508708e-02, 4.5410977124e-05,
                    -7.7998048686e-08, 2.5800160612e-10,
                    -5.9452583057e-13, 9.3214058667e-16,
                    -8.1819730750e-19, 3.8003286862e-22,
                    -7.2893246250e-26],
    },
    "S": {
        "max_temp": 1768,
        "color": "#9467bd",
        "coeffs": [0.0, 5.4030544256e-03, 1.2593428974e-05,
                    -2.3247937549e-08, 3.2203091293e-11,
                    -3.3145945973e-14, 2.5575883544e-17,
                    -1.2507891902e-20, 2.7144077078e-24],
    },
}

fig, ax = plt.subplots(figsize=(12, 7))

for tc_type, data in tc_data.items():
    t_max = data["max_temp"]
    coeffs = data["coeffs"]
    color = data["color"]
    temp = np.linspace(0, t_max, 500)
    voltage = np.zeros_like(temp)
    for i, c in enumerate(coeffs):
        voltage += c * temp ** i
    ax.plot(temp, voltage, color=color, linewidth=2.5, label=f"Type {tc_type}")
    ax.annotate(f"  {tc_type}", xy=(temp[-1], voltage[-1]),
                 fontsize=12, fontweight="bold", color=color,
                 va="center")

ax.axvline(500, color="gray", linestyle=":", alpha=0.4)
ax.text(510, -2, "500°C", fontsize=9, color="gray", va="top")

ax.set_xlabel("Temperature (°C)", fontsize=12)
ax.set_ylabel("Thermocouple Voltage (mV)", fontsize=12)
ax.set_title("Thermocouple Voltage vs Temperature (Reference Junction at 0°C)",
              fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1850)
ax.set_ylim(-2, 80)

ax.text(1400, 72, "Sensitivity (at 25°C):", fontsize=9, fontweight="bold",
         color="black", va="top")
sensitivities = [("E", "~68 μV/°C"), ("J", "~52 μV/°C"), ("K", "~41 μV/°C"),
                  ("T", "~43 μV/°C"), ("S", "~6 μV/°C")]
for i, (tc, sens) in enumerate(sensitivities):
    ax.text(1400, 68 - i * 4, f"  Type {tc}: {sens}", fontsize=8,
             color=tc_data[tc]["color"], va="top")

fig.tight_layout()
save(fig, "ch11_thermocouple.png")


# ============================================================
# Chapter 15: Networking
# ============================================================
print("Chapter 15: Networking")

# --- 15.5.1 Ethernet Frame Efficiency vs Payload Size ---
_overhead_g = 8 + 14 + 4 + 12   # 38 B fixed on-wire overhead
_payload_g  = np.arange(46, 9001, 1)
_eff_g      = _payload_g / (_payload_g + _overhead_g) * 100

fig, _ax_eth_g = plt.subplots(figsize=(10, 6))
_ax_eth_g.plot(_payload_g, _eff_g, "b-", linewidth=2)
_mask_std_g = _payload_g <= 1500
_ax_eth_g.fill_between(_payload_g[_mask_std_g],  _eff_g[_mask_std_g],  alpha=0.12, color="blue",
                        label="Standard Ethernet (46–1500 B)")
_ax_eth_g.fill_between(_payload_g[~_mask_std_g], _eff_g[~_mask_std_g], alpha=0.12, color="orange",
                        label="Jumbo frames (> 1500 B)")
for _p_g, _lbl_g, _c_g in [
    (46,   "Min frame\n46 B payload",     "red"),
    (1500, "Max standard\n1500 B payload", "green"),
    (9000, "Jumbo frame\n9000 B payload",  "darkorange"),
]:
    _e_g = _p_g / (_p_g + _overhead_g) * 100
    _ax_eth_g.plot(_p_g, _e_g, "o", color=_c_g, markersize=9, zorder=5)
    _dx_g = 200 if _p_g < 500 else (-1200 if _p_g == 9000 else 300)
    _dy_g = -4  if _p_g < 500 else 2
    _ax_eth_g.annotate(
        f"{_lbl_g}\n{_e_g:.1f}%",
        xy=(_p_g, _e_g), xytext=(_p_g + _dx_g, _e_g + _dy_g),
        fontsize=9, color=_c_g,
        arrowprops=dict(arrowstyle="->", color=_c_g),
    )
_ax_eth_g.axvline(x=1500, color="gray", linestyle="--", alpha=0.6, linewidth=1)
_ax_eth_g.set_xlabel("Ethernet Payload Size (bytes)", fontsize=10)
_ax_eth_g.set_ylabel("Protocol Efficiency (%)", fontsize=10)
_ax_eth_g.set_title(
    "Ethernet Frame Efficiency vs Payload Size\n"
    f"(Fixed overhead = {_overhead_g} B: 8 preamble + 14 header + 4 FCS + 12 IFG)",
    fontsize=11,
)
_ax_eth_g.legend(fontsize=9, loc="lower right")
_ax_eth_g.grid(True, alpha=0.3)
_ax_eth_g.set_xlim(0, 9000)
_ax_eth_g.set_ylim(40, 102)
fig.tight_layout()
save(fig, "ch15_ethernet_efficiency.png")

# --- 15.3.5 Fiber Optic Link Budget vs Distance ---
_P_tx_g      =  3.0     # dBm
_P_rx_g      = -28.0    # dBm
_budget_g    = _P_tx_g - _P_rx_g   # 31 dB
_conn_loss_g = 2 * 0.5  # 1.0 dB
_spl_loss_g  = 4 * 0.1  # 0.4 dB
_fixed_g     = _conn_loss_g + _spl_loss_g   # 1.4 dB
_sys_margin_g = 3.0
_avail_g     = _budget_g - _fixed_g - _sys_margin_g   # 26.6 dB
_dist_km_g   = np.linspace(0, 120, 600)

_scenarios_g = [
    ("0.20 dB/km (standard G.652)", 0.20, "blue"),
    ("0.25 dB/km (older SMF)",      0.25, "green"),
    ("0.35 dB/km (1310 nm)",        0.35, "red"),
]

fig, _ax_fib_g = plt.subplots(figsize=(10, 6))
for _slbl_g, _alpha_g, _sc_g in _scenarios_g:
    _margin_arr_g = _avail_g - _alpha_g * _dist_km_g
    _ax_fib_g.plot(_dist_km_g, _margin_arr_g, color=_sc_g, linewidth=2, label=_slbl_g)
    _d_max_g = _avail_g / _alpha_g
    if _d_max_g <= 120:
        _ax_fib_g.plot(_d_max_g, 0, "x", color=_sc_g, markersize=10, markeredgewidth=2, zorder=5)
        _ax_fib_g.annotate(f"{_d_max_g:.0f} km", xy=(_d_max_g, 0.5),
                           ha="center", fontsize=8, color=_sc_g)
_d_ex_g     = 40.0
_margin_ex_g = _avail_g - 0.25 * _d_ex_g
_ax_fib_g.plot(_d_ex_g, _margin_ex_g, "go", markersize=11, zorder=6)
_ax_fib_g.annotate(
    f"§15.3.5 example\n40 km @ 0.25 dB/km\nMargin = {_margin_ex_g:.1f} dB",
    xy=(_d_ex_g, _margin_ex_g), xytext=(_d_ex_g + 10, _margin_ex_g - 4),
    fontsize=9, color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen"),
)
_ax_fib_g.axhline(y=0, color="black", linewidth=1.2)
_ax_fib_g.fill_between(_dist_km_g, 0, -15, alpha=0.08, color="red", label="Link fails (margin < 0)")
_ax_fib_g.axhline(y=_sys_margin_g, color="gray", linestyle=":", alpha=0.6,
                  label=f"System margin = {_sys_margin_g} dB")
_ax_fib_g.set_xlabel("Fiber Length (km)", fontsize=10)
_ax_fib_g.set_ylabel("Available Power Margin (dB)", fontsize=10)
_ax_fib_g.set_title(
    f"SMF Link Budget: Margin vs Distance\n"
    f"(Budget = {_budget_g} dB; fixed losses = {_fixed_g} dB; system margin = {_sys_margin_g} dB)",
    fontsize=11,
)
_ax_fib_g.legend(fontsize=9)
_ax_fib_g.grid(True, alpha=0.3)
_ax_fib_g.set_xlim(0, 120)
_ax_fib_g.set_ylim(-15, 30)
fig.tight_layout()
save(fig, "ch15_fiber_link_budget.png")

# --- 15.10.5 QoS Priority Queuing ---
_link_Mbps_g = 100.0
_ef_Mbps_g   = 0.10 * _link_Mbps_g
_af21_Mbps_g = 0.50 * _link_Mbps_g
_be_Mbps_g   = 0.40 * _link_Mbps_g
_n_calls_g   = 50
_bw_call_g   = 87.2e-3   # Mbps
_voip_Mbps_g = _n_calls_g * _bw_call_g
_ef_headroom_g = _ef_Mbps_g - _voip_Mbps_g
_ser_us_g = 1500 * 8 / (_link_Mbps_g * 1e6) * 1e6
_delay_noqos_g = 3 * _ser_us_g
_delay_llq_g   = _ser_us_g

fig, (_ax_bw_g, _ax_dly_g) = plt.subplots(1, 2, figsize=(12, 6))
_cls_labels_g = ["EF (VoIP)\nDSCP 46", "AF21 (Business)\nDSCP 18", "BE (Default)\nDSCP 0"]
_cls_bws_g    = [_ef_Mbps_g, _af21_Mbps_g, _be_Mbps_g]
_cls_colors_g = ["#CC3333", "#3366CC", "#888888"]
_bars_bw_g = _ax_bw_g.bar(_cls_labels_g, _cls_bws_g, color=_cls_colors_g,
                           edgecolor="white", linewidth=0.8, width=0.55)
for _bar_g, _bw_g in zip(_bars_bw_g, _cls_bws_g):
    _ax_bw_g.text(_bar_g.get_x() + _bar_g.get_width() / 2,
                  _bar_g.get_height() + 0.8,
                  f"{_bw_g:.0f} Mbps\n({_bw_g/_link_Mbps_g*100:.0f}%)",
                  ha="center", fontsize=10, fontweight="bold")
_ax_bw_g.axhline(y=_voip_Mbps_g, color="darkred", linestyle="--", alpha=0.7,
                 linewidth=1.5, label=f"VoIP demand: {_voip_Mbps_g:.2f} Mbps")
_ax_bw_g.annotate(
    f"{_n_calls_g} calls × {_bw_call_g*1000:.1f} kbps\n= {_voip_Mbps_g:.2f} Mbps\n"
    f"({_ef_headroom_g:.2f} Mbps headroom)",
    xy=(0, _voip_Mbps_g), xytext=(0.55, _voip_Mbps_g + 4),
    fontsize=8.5, color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred"),
)
_ax_bw_g.set_ylabel("Allocated Bandwidth (Mbps)", fontsize=10)
_ax_bw_g.set_title(f"LLQ Bandwidth Allocation\n(100 Mbps WAN link)", fontsize=11)
_ax_bw_g.legend(fontsize=9)
_ax_bw_g.grid(True, alpha=0.3, axis="y")
_ax_bw_g.set_ylim(0, 65)

_dly_labels_g = ["Without QoS\n(3 × 1500 B ahead)", "With LLQ\n(strict priority)"]
_dly_vals_g   = [_delay_noqos_g, _delay_llq_g]
_dly_colors_g = ["#CC3333", "#228833"]
_bars_dly_g = _ax_dly_g.bar(_dly_labels_g, _dly_vals_g, color=_dly_colors_g,
                             edgecolor="white", linewidth=0.8, width=0.45)
for _bar_g, _d_g in zip(_bars_dly_g, _dly_vals_g):
    _ax_dly_g.text(_bar_g.get_x() + _bar_g.get_width() / 2,
                   _d_g + 5, f"{_d_g:.0f} μs",
                   ha="center", fontsize=13, fontweight="bold")
_ratio_g = _delay_noqos_g / _delay_llq_g
_ax_dly_g.annotate(
    f"{_ratio_g:.0f}× improvement with LLQ",
    xy=(1, _delay_llq_g), xytext=(0.78, _delay_noqos_g * 0.6),
    fontsize=10, color="darkgreen", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="darkgreen"),
)
_ax_dly_g.text(
    0.5, 385,
    "VoIP latency budget: 150 ms = 150,000 μs\n(both cases well within budget)",
    ha="center", fontsize=8.5, color="gray",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
)
_ax_dly_g.set_ylabel("Voice Packet Queuing Delay (μs)", fontsize=10)
_ax_dly_g.set_title("Voice Packet Queuing Delay\nWithout vs. With LLQ (100 Mbps link)", fontsize=11)
_ax_dly_g.grid(True, alpha=0.3, axis="y")
_ax_dly_g.set_ylim(0, 450)
fig.tight_layout()
save(fig, "ch15_qos_llq.png")


print("\nDone! All images saved to", IMG_DIR)
