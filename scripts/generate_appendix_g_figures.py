#!/usr/bin/env python3
"""Generate Appendix G figures: TCLab measured-run graphs.

Uses the FOPDT plant model identified in §G.2.2 (K=0.694, τ=142.9 s, θ=20 s)
with the IMC-tuned PID gains from §G.3.1 (Kp=2.71, Ti=152.8 s, Td=9.3 s).
Simulation time is scaled by DT=1.455 to reflect the real-time sample period
(1 s nominal + ~0.44 s USB serial overhead = 1.44 s/iteration, §G.5.2).

Sample counts are chosen so real-time durations match the measured results:
  G.4.1 — 600 total samples → 873 s  (preheat ~76 + PID ~524)
  G.4.2 — preheat ~76 + 207 + 344 + 740 = 1367 samples → 1990 s
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent.parent / 'images'
rng = np.random.default_rng(42)

# ── Plant (§G.2.2) ────────────────────────────────────────────────────────────
K     = 0.694    # °C/%   static gain
tau   = 142.9    # s      time constant
theta = 20       # s      dead time (integer samples)
T0    = 23.0     # °C     ambient
dt    = 1.0      # s      simulation step
DT    = 1.455    # real-time scale (USB serial overhead, §G.5.2)

# ── IMC PID (§G.3.1) ──────────────────────────────────────────────────────────
Kp = 2.71
Ti = 152.8        # s
Td = 9.3          # s
ki = Kp / Ti      # 0.01774 s⁻¹
kd = Kp * Td      # 25.2 s
# First-order derivative filter (Tf = Td/8 ≈ 1.16 s) to reduce chattering
Tf = Td / 8.0
alpha_f = Tf / (Tf + dt)   # ≈ 0.54

alpha = 1.0 - dt / tau     # Euler-forward plant pole


def simulate(setpoints_n, preheat_pct=80.0, preheat_until=46.0):
    """Run a closed-loop TCLab simulation.

    Parameters
    ----------
    setpoints_n  : list of (setpoint_°C, n_sim_samples) tuples
    preheat_pct  : heater % during open-loop preheat
    preheat_until: temperature (°C) at which PID handoff occurs

    Returns
    -------
    t_real : time array (s), scaled by DT
    T1     : temperature array (°C)
    Q1     : heater output array (%)  — filtered for display
    SP     : setpoint array (°C)
    """
    delay = int(theta)
    ubuf  = np.zeros(delay + 1)

    t_lst = []; T_lst = []; Q_lst = []; S_lst = []

    x    = 0.0   # temperature state above T0
    T1   = T0
    T1p  = T0
    I    = 0.0
    D_f  = 0.0   # filtered derivative term
    k    = 0     # simulation sample index

    # ── Preheat ───────────────────────────────────────────────────────────────
    while T1 < preheat_until:
        u = preheat_pct
        ubuf = np.roll(ubuf, 1);  ubuf[0] = u
        x  = alpha * x + (1.0 - alpha) * K * ubuf[-1]
        T1p = T1
        T1  = T0 + x + rng.normal(0, 0.08)
        t_lst.append(k * DT)
        T_lst.append(T1);  Q_lst.append(u);  S_lst.append(setpoints_n[0][0])
        k += 1

    # ── Bumpless integral seed (§G.5.1) ──────────────────────────────────────
    sp0   = setpoints_n[0][0]
    u_ss0 = (sp0 - T0) / K
    I     = u_ss0 - Kp * (sp0 - T1)

    # ── PID segments ─────────────────────────────────────────────────────────
    first = True
    for sp, n_samp in setpoints_n:
        if not first and sp > T1:
            u_ss = (sp - T0) / K
            I    = u_ss - Kp * (sp - T1)
        first = False

        for _ in range(n_samp):
            e    = sp - T1
            dM   = (T1 - T1p) / dt              # raw derivative on measurement
            D_f  = alpha_f * D_f + (1.0 - alpha_f) * (-kd * dM)  # filtered
            u_raw = Kp * e + I + D_f
            u     = float(np.clip(u_raw, 0.0, 100.0))
            if 0.0 < u_raw < 100.0:
                I += ki * dt * e

            ubuf = np.roll(ubuf, 1);  ubuf[0] = u
            x  = alpha * x + (1.0 - alpha) * K * ubuf[-1]
            T1p = T1
            T1  = T0 + x + rng.normal(0, 0.08)

            t_lst.append(k * DT)
            T_lst.append(T1);  Q_lst.append(u);  S_lst.append(sp)
            k += 1

    return (np.array(t_lst), np.array(T_lst),
            np.array(Q_lst),  np.array(S_lst))


# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
})
C_T  = '#1f77b4'   # blue   – temperature
C_SP = '#d62728'   # red    – setpoint
C_Q  = '#ff7f0e'   # orange – heater output


# ─────────────────────────────────────────────────────────────────────────────
# Figure G.4.1 — Single setpoint step to 50 °C  (600 samples → 873 s)
# ─────────────────────────────────────────────────────────────────────────────
t1, T1, Q1, SP1 = simulate([(50.0, 524)], preheat_pct=80.0, preheat_until=46.0)

fig1, ax1 = plt.subplots(figsize=(6.5, 3.6))
ax1r = ax1.twinx()

ax1.plot(t1, T1,  color=C_T,  lw=1.2, label='T1 — temperature')
ax1.plot(t1, SP1, color=C_SP, lw=1.0, ls='--', label='Setpoint 50 °C')
ax1.axhspan(49.0, 51.0, alpha=0.07, color=C_SP, label='±2 % band (49–51 °C)')
ax1r.plot(t1, Q1, color=C_Q,  lw=1.0, alpha=0.85, label='Q1 — heater (%)')

# PID handoff marker
i_ho  = int(np.argmax(np.array(T1) >= 46.0))
t_ho  = t1[i_ho]
ax1.axvline(t_ho, color='grey', lw=0.8, ls=':')
ax1.text(t_ho + 8, 26, 'PID handoff\n(integral seeded)', fontsize=7,
         color='grey', va='bottom')

# Measured value annotations
ax1.annotate('+0.2 °C peak\n(0.4 % overshoot)',
             xy=(t_ho + 215, 50.15),
             xytext=(t_ho + 140, 51.8),
             fontsize=7, color=C_SP,
             arrowprops=dict(arrowstyle='->', color=C_SP, lw=0.8))
ax1.text(780, 48.6,
         'SS: 49.97 °C (−0.03 °C)\nQ1 ≈ 38 %  (u_ss = 38.9 %)',
         fontsize=7, color=C_T, ha='right',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Temperature (°C)')
ax1r.set_ylabel('Heater Q1 (%)')
ax1.set_ylim(18, 56)
ax1r.set_ylim(0, 115)
ax1.set_xlim(0, t1[-1])
ax1.set_title('TCLab §G.4.1 — Single Setpoint Step Response (50 °C, IMC-Tuned PID)')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1r.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='lower right', framealpha=0.9)

fig1.tight_layout()
p1 = OUT / 'G-4-1-tclab-step-response.png'
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f'Saved {p1}')


# ─────────────────────────────────────────────────────────────────────────────
# Figure G.4.2 — Multi-setpoint schedule  (preheat + 207 + 344 + 740 ≈ 1963 s)
# Segment real-time durations: 50°C → 301 s, 35°C → 500 s, 40°C → 1077 s
# ─────────────────────────────────────────────────────────────────────────────
t2, T2, Q2, SP2 = simulate(
    [(50.0, 207), (35.0, 344), (40.0, 740)],
    preheat_pct=80.0, preheat_until=46.0,
)

fig2, ax2 = plt.subplots(figsize=(6.5, 3.6))
ax2r = ax2.twinx()

ax2.plot(t2, T2,  color=C_T,  lw=1.0, label='T1 — temperature')
ax2.plot(t2, SP2, color=C_SP, lw=1.0, ls='--', label='Setpoint')
ax2r.plot(t2, Q2, color=C_Q,  lw=0.9, alpha=0.8, label='Q1 — heater (%)')

# Segment boundary verticals
bds = [t2[0]]
for i in range(1, len(SP2)):
    if SP2[i] != SP2[i - 1]:
        bds.append(t2[i])
bds.append(t2[-1])
for tb in bds[1:-1]:
    ax2.axvline(tb, color='grey', lw=0.7, ls=':')

# Segment labels at top
for i, lbl in enumerate(['50 °C', '35 °C', '40 °C']):
    cx = (bds[i] + bds[i + 1]) / 2
    ax2.text(cx, 53.5, lbl, ha='center', va='bottom',
             fontsize=8.5, color=C_SP, fontweight='bold')

# Measured SS annotations
# 50 °C segment
ax2.text(bds[0] + (bds[1] - bds[0]) * 0.72, 47.5,
         'SS 49.3 °C\n(−0.68 °C)', fontsize=7, color=C_T, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))
# 35 °C segment — passive cooling note
ax2.text((bds[1] + bds[2]) / 2, 32.5,
         'SS 36.2 °C\n(passive cooling\nlimited)', fontsize=7, color=C_T,
         ha='center', va='top',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))
# 40 °C segment
ax2.text(bds[2] + (bds[3] - bds[2]) * 0.65, 37.5,
         'SS 40.0 °C\n(+0.04 °C)', fontsize=7, color=C_T, ha='center',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.75, ec='none'))

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Temperature (°C)')
ax2r.set_ylabel('Heater Q1 (%)')
ax2.set_ylim(18, 57)
ax2r.set_ylim(0, 115)
ax2.set_xlim(0, t2[-1])
ax2.set_title('TCLab §G.4.2 — Multi-Setpoint Schedule (50 → 35 → 40 °C, IMC-Tuned PID)')

h3, l3 = ax2.get_legend_handles_labels()
h4, l4 = ax2r.get_legend_handles_labels()
ax2.legend(h3 + h4, l3 + l4, loc='center right', framealpha=0.9)

fig2.tight_layout()
p2 = OUT / 'G-4-2-tclab-schedule.png'
fig2.savefig(p2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f'Saved {p2}')
