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
        # Chapter 15: Networking — Example Visualizations

        Interactive graphs for selected example problems from Chapter 15,
        covering Ethernet frame efficiency, fiber link budgets, and
        Quality of Service (QoS) priority queuing.
        """
    )
    return


# --- 15.5.1 Ethernet Frame Efficiency vs Payload Size ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 15.5.1 Ethernet Frame Efficiency vs Payload Size

        An Ethernet frame carries between 46 and 1500 bytes of payload, plus
        38 bytes of fixed overhead (preamble 8 B + header 14 B + FCS 4 B +
        IFG 12 B).  Protocol efficiency drops sharply for small frames because
        the overhead becomes dominant.

        The minimum-size 64-byte frame (46 bytes payload) has only 54.8%
        efficiency, while a jumbo frame (9000 bytes) exceeds 99%.
        """
    )
    return


@app.cell
def _(np, plt):
    # Fixed on-wire overhead per frame (bytes)
    _overhead = 8 + 14 + 4 + 12   # preamble/SFD, Ethernet header, FCS, IFG = 38 B

    _payload = np.arange(46, 9001, 1)
    _eff     = _payload / (_payload + _overhead) * 100

    fig_eth, _ax_eth = plt.subplots(figsize=(10, 6))
    _ax_eth.plot(_payload, _eff, "b-", linewidth=2)

    _mask_std = _payload <= 1500
    _ax_eth.fill_between(_payload[_mask_std],  _eff[_mask_std],  alpha=0.12, color="blue",
                         label="Standard Ethernet (46–1500 B)")
    _ax_eth.fill_between(_payload[~_mask_std], _eff[~_mask_std], alpha=0.12, color="orange",
                         label="Jumbo frames (> 1500 B)")

    for _p, _lbl, _c in [
        (46,   "Min frame\n46 B payload",     "red"),
        (1500, "Max standard\n1500 B payload", "green"),
        (9000, "Jumbo frame\n9000 B payload",  "darkorange"),
    ]:
        _e = _p / (_p + _overhead) * 100
        _ax_eth.plot(_p, _e, "o", color=_c, markersize=9, zorder=5)
        _dx = 200 if _p < 500 else (-1200 if _p == 9000 else 300)
        _dy = -4  if _p < 500 else 2
        _ax_eth.annotate(
            f"{_lbl}\n{_e:.1f}%",
            xy=(_p, _e), xytext=(_p + _dx, _e + _dy),
            fontsize=9, color=_c,
            arrowprops=dict(arrowstyle="->", color=_c),
        )

    _ax_eth.axvline(x=1500, color="gray", linestyle="--", alpha=0.6, linewidth=1)
    _ax_eth.set_xlabel("Ethernet Payload Size (bytes)", fontsize=10)
    _ax_eth.set_ylabel("Protocol Efficiency (%)", fontsize=10)
    _ax_eth.set_title(
        "Ethernet Frame Efficiency vs Payload Size\n"
        f"(Fixed overhead = {_overhead} B: 8 preamble + 14 header + 4 FCS + 12 IFG)",
        fontsize=11,
    )
    _ax_eth.legend(fontsize=9, loc="lower right")
    _ax_eth.grid(True, alpha=0.3)
    _ax_eth.set_xlim(0, 9000)
    _ax_eth.set_ylim(40, 102)

    fig_eth.tight_layout()
    fig_eth
    return fig_eth


# --- 15.3.5 Fiber Optic Link Budget vs Distance ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 15.3.5 Fiber Optic Link Budget — Margin vs Distance

        A single-mode fiber link at 1550 nm with a transmitter at +3 dBm and
        a receiver sensitivity of −28 dBm has a total power budget of 31 dB.
        Subtracting fixed losses (2 connectors at 0.5 dB each, 4 splices at
        0.1 dB each, 3 dB system margin) leaves 26.6 dB for fiber attenuation.

        The chart shows available power margin vs fiber length for three
        common SMF attenuation values, and marks the 40 km example from §15.3.5.
        """
    )
    return


@app.cell
def _(np, plt):
    _P_tx      =  3.0     # dBm
    _P_rx      = -28.0    # dBm
    _budget    = _P_tx - _P_rx   # 31 dB

    _conn_loss = 2 * 0.5  # 1.0 dB
    _spl_loss  = 4 * 0.1  # 0.4 dB
    _fixed     = _conn_loss + _spl_loss   # 1.4 dB
    _sys_margin = 3.0

    _avail     = _budget - _fixed - _sys_margin   # 26.6 dB

    _dist_km   = np.linspace(0, 120, 600)

    _scenarios = [
        ("0.20 dB/km (standard G.652)", 0.20, "blue"),
        ("0.25 dB/km (older SMF)",      0.25, "green"),
        ("0.35 dB/km (1310 nm)",        0.35, "red"),
    ]

    fig_fib, _ax_fib = plt.subplots(figsize=(10, 6))

    for _slbl, _alpha, _sc in _scenarios:
        _margin_arr = _avail - _alpha * _dist_km
        _ax_fib.plot(_dist_km, _margin_arr, color=_sc, linewidth=2, label=_slbl)
        _d_max = _avail / _alpha
        if _d_max <= 120:
            _ax_fib.plot(_d_max, 0, "x", color=_sc, markersize=10,
                         markeredgewidth=2, zorder=5)
            _ax_fib.annotate(f"{_d_max:.0f} km", xy=(_d_max, 0.5),
                             ha="center", fontsize=8, color=_sc)

    # Mark §15.3.5 example (40 km, 0.25 dB/km)
    _d_ex     = 40.0
    _margin_ex = _avail - 0.25 * _d_ex
    _ax_fib.plot(_d_ex, _margin_ex, "go", markersize=11, zorder=6)
    _ax_fib.annotate(
        f"§15.3.5 example\n40 km @ 0.25 dB/km\nMargin = {_margin_ex:.1f} dB",
        xy=(_d_ex, _margin_ex), xytext=(_d_ex + 10, _margin_ex - 4),
        fontsize=9, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
    )

    _ax_fib.axhline(y=0, color="black", linewidth=1.2)
    _ax_fib.fill_between(_dist_km, 0, -15, alpha=0.08, color="red",
                         label="Link fails (margin < 0)")
    _ax_fib.axhline(y=_sys_margin, color="gray", linestyle=":", alpha=0.6,
                    label=f"System margin = {_sys_margin} dB")

    _ax_fib.set_xlabel("Fiber Length (km)", fontsize=10)
    _ax_fib.set_ylabel("Available Power Margin (dB)", fontsize=10)
    _ax_fib.set_title(
        f"SMF Link Budget: Margin vs Distance\n"
        f"(Budget = {_budget} dB; fixed losses = {_fixed} dB; "
        f"system margin = {_sys_margin} dB)",
        fontsize=11,
    )
    _ax_fib.legend(fontsize=9)
    _ax_fib.grid(True, alpha=0.3)
    _ax_fib.set_xlim(0, 120)
    _ax_fib.set_ylim(-15, 30)

    fig_fib.tight_layout()
    fig_fib
    return fig_fib


# --- 15.10.5 QoS Priority Queuing ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 15.10.5 QoS — Low Latency Queuing Bandwidth Allocation

        An enterprise WAN link (100 Mbps) uses Low Latency Queuing (LLQ) to
        protect 50 concurrent G.711 VoIP calls (87.2 kbps each = 4.36 Mbps)
        with a strict-priority EF queue limited to 10 % of the link.
        AF21 business traffic receives 50 % WFQ bandwidth and best-effort
        gets the remaining 40 %.

        The left chart shows the bandwidth allocation per traffic class.
        The right chart compares queuing delay for a voice packet without QoS
        (competing with large best-effort frames) vs. with LLQ.
        """
    )
    return


@app.cell
def _(plt):
    _link_Mbps = 100.0

    _ef_Mbps   = 0.10 * _link_Mbps   # 10 Mbps
    _af21_Mbps = 0.50 * _link_Mbps   # 50 Mbps
    _be_Mbps   = 0.40 * _link_Mbps   # 40 Mbps

    _n_calls     = 50
    _bw_call     = 87.2e-3   # Mbps
    _voip_Mbps   = _n_calls * _bw_call   # 4.36 Mbps
    _ef_headroom = _ef_Mbps - _voip_Mbps

    # Serialization delay for a 1500-byte frame at 100 Mbps
    _ser_us = 1500 * 8 / (_link_Mbps * 1e6) * 1e6   # 120 μs

    # Without QoS: voice waits behind 3 best-effort frames
    _delay_noqos_us = 3 * _ser_us   # 360 μs
    # With LLQ: voice waits at most for one frame already being transmitted
    _delay_llq_us   = _ser_us       # 120 μs

    fig_qos, (_ax_bw, _ax_dly) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Bandwidth allocation ---
    _cls_labels = ["EF (VoIP)\nDSCP 46", "AF21 (Business)\nDSCP 18", "BE (Default)\nDSCP 0"]
    _cls_bws    = [_ef_Mbps, _af21_Mbps, _be_Mbps]
    _cls_colors = ["#CC3333", "#3366CC", "#888888"]

    _bars_bw = _ax_bw.bar(_cls_labels, _cls_bws, color=_cls_colors,
                           edgecolor="white", linewidth=0.8, width=0.55)
    for _bar, _bw in zip(_bars_bw, _cls_bws):
        _ax_bw.text(_bar.get_x() + _bar.get_width() / 2,
                    _bar.get_height() + 0.8,
                    f"{_bw:.0f} Mbps\n({_bw/_link_Mbps*100:.0f}%)",
                    ha="center", fontsize=10, fontweight="bold")

    _ax_bw.axhline(y=_voip_Mbps, color="darkred", linestyle="--", alpha=0.7,
                   linewidth=1.5, label=f"VoIP demand: {_voip_Mbps:.2f} Mbps")
    _ax_bw.annotate(
        f"{_n_calls} calls × {_bw_call*1000:.1f} kbps\n= {_voip_Mbps:.2f} Mbps\n"
        f"({_ef_headroom:.2f} Mbps headroom)",
        xy=(0, _voip_Mbps), xytext=(0.55, _voip_Mbps + 4),
        fontsize=8.5, color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred"),
    )
    _ax_bw.set_ylabel("Allocated Bandwidth (Mbps)", fontsize=10)
    _ax_bw.set_title(f"LLQ Bandwidth Allocation\n(100 Mbps WAN link)", fontsize=11)
    _ax_bw.legend(fontsize=9)
    _ax_bw.grid(True, alpha=0.3, axis="y")
    _ax_bw.set_ylim(0, 65)

    # --- Queuing delay comparison ---
    _dly_labels = ["Without QoS\n(3 × 1500 B ahead)", "With LLQ\n(strict priority)"]
    _dly_vals   = [_delay_noqos_us, _delay_llq_us]
    _dly_colors = ["#CC3333", "#228833"]

    _bars_dly = _ax_dly.bar(_dly_labels, _dly_vals, color=_dly_colors,
                             edgecolor="white", linewidth=0.8, width=0.45)
    for _bar, _d in zip(_bars_dly, _dly_vals):
        _ax_dly.text(_bar.get_x() + _bar.get_width() / 2,
                     _d + 5,
                     f"{_d:.0f} μs",
                     ha="center", fontsize=13, fontweight="bold")

    _ratio = _delay_noqos_us / _delay_llq_us
    _ax_dly.annotate(
        f"{_ratio:.0f}× improvement with LLQ",
        xy=(1, _delay_llq_us), xytext=(0.78, _delay_noqos_us * 0.6),
        fontsize=10, color="darkgreen", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
    )
    _ax_dly.text(
        0.5, 385,
        "VoIP latency budget: 150 ms = 150,000 μs\n(both cases well within budget)",
        ha="center", fontsize=8.5, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7),
    )
    _ax_dly.set_ylabel("Voice Packet Queuing Delay (μs)", fontsize=10)
    _ax_dly.set_title("Voice Packet Queuing Delay\nWithout vs. With LLQ (100 Mbps link)",
                      fontsize=11)
    _ax_dly.grid(True, alpha=0.3, axis="y")
    _ax_dly.set_ylim(0, 450)

    fig_qos.tight_layout()
    fig_qos
    return fig_qos


if __name__ == "__main__":
    app.run()
