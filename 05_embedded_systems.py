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
        # Chapter 5: Embedded Systems — Example Visualizations

        Interactive graphs for selected example problems from Chapter 5,
        covering SPI timing, sleep-mode power budgets, and low-power
        design trade-offs.
        """
    )
    return


# --- 5.4.2 SPI Timing Diagram ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 5.4.2 SPI — 8-Bit Transfer Timing Diagram (Mode 0)

        SPI uses four signals: SCLK (clock), CS (chip select, active-low),
        MOSI (master out slave in), and MISO (master in slave out).
        In Mode 0 (CPOL = 0, CPHA = 0) data is sampled on the rising edge
        and shifted on the falling edge.  This diagram shows the transmission
        of the byte 0xA5 (1010 0101) from master to slave.
        """
    )
    return


@app.cell
def _(np, plt):
    # Byte 0xA5 = 1010_0101 (MSB first)
    _byte_val = 0xA5
    _bits_mosi = [((_byte_val) >> (7 - _i)) & 1 for _i in range(8)]

    # Build time axis: 8 clock cycles + pre/post idle
    _n_bits = 8
    _t_per_bit = 1.0          # 1 unit per clock period
    _t_pre = 0.75             # idle before CS asserts
    _t_total = _t_pre + _n_bits * _t_per_bit + 0.75

    _t = np.linspace(0, _t_total, 2000)

    # SCLK: high for the second half of each bit period
    _sclk = np.zeros_like(_t)
    for _i in range(_n_bits):
        _t_start = _t_pre + _i * _t_per_bit
        _t_mid = _t_start + 0.5 * _t_per_bit
        _sclk[(_t >= _t_mid) & (_t < _t_start + _t_per_bit)] = 1.0

    # CS: active-low during transfer
    _cs = np.where((_t >= _t_pre) & (_t < _t_pre + _n_bits * _t_per_bit), 0.0, 1.0)

    # MOSI data
    _mosi = np.zeros_like(_t)
    for _i, _b in enumerate(_bits_mosi):
        _ts = _t_pre + _i * _t_per_bit
        _te = _ts + _t_per_bit
        _mosi[(_t >= _ts) & (_t < _te)] = float(_b)

    # MISO dummy reply byte 0x3C = 0011_1100
    _bits_miso = [(0x3C >> (7 - _i)) & 1 for _i in range(8)]
    _miso = np.zeros_like(_t)
    for _i, _b in enumerate(_bits_miso):
        _ts = _t_pre + _i * _t_per_bit
        _te = _ts + _t_per_bit
        _miso[(_t >= _ts) & (_t < _te)] = float(_b)

    _sig_colors = ["#2266CC", "#AA2222", "#228822", "#996600"]
    _sig_labels = ["CS (active-low)", "SCLK", "MOSI (0xA5)", "MISO (0x3C)"]
    _sig_waves  = [_cs, _sclk, _mosi, _miso]

    fig_spi, _axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)

    for _ax, _lbl, _sig, _sc in zip(_axes, _sig_labels, _sig_waves, _sig_colors):
        _ax.plot(_t, _sig, color=_sc, linewidth=2)
        _ax.fill_between(_t, 0, _sig, alpha=0.12, color=_sc)
        _ax.set_ylabel(_lbl, fontsize=9, labelpad=4)
        _ax.set_ylim(-0.3, 1.4)
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(["Lo", "Hi"], fontsize=8)
        _ax.grid(True, alpha=0.2, axis="x")
        _ax.spines["top"].set_visible(False)
        _ax.spines["right"].set_visible(False)

    # Annotate bit values on MOSI panel
    _ax_mosi = _axes[2]
    for _i, _b in enumerate(_bits_mosi):
        _xc = _t_pre + (_i + 0.5) * _t_per_bit
        _ax_mosi.text(_xc, 1.15, str(_b), ha="center", va="bottom",
                      fontsize=10, fontweight="bold", color=_sig_colors[2])

    # Rising-edge markers
    for _i in range(_n_bits):
        _tr = _t_pre + _i * _t_per_bit + 0.5 * _t_per_bit
        _axes[1].axvline(x=_tr, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        _axes[2].axvline(x=_tr, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

    _axes[-1].set_xlabel("Time (bit periods, 1 unit = 1 / f_SCLK)", fontsize=10)
    _axes[0].set_title(
        "SPI Mode 0 Timing Diagram — 8-bit Transfer\n"
        "(MOSI: 0xA5 = 1010 0101, MISO: 0x3C = 0011 1100)",
        fontsize=11,
    )
    _axes[0].annotate(
        "", xy=(_t_pre + _n_bits * _t_per_bit, 0.5), xytext=(_t_pre, 0.5),
        arrowprops=dict(arrowstyle="<->", color="navy", lw=1.5),
    )
    _axes[0].text(_t_pre + _n_bits * 0.5 * _t_per_bit, 0.65,
                  "8 clock cycles (active)", ha="center", fontsize=9, color="navy")

    fig_spi.tight_layout(h_pad=0.3)
    fig_spi
    return fig_spi


# --- 5.7.1 Sleep Mode Power Budget and Battery Life ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 5.7.1 Sleep Modes — Average Current vs Wake Interval

        A battery-powered sensor node wakes periodically, takes a reading, and
        returns to Stop mode.  Average current I_avg = (I_active × t_active +
        I_sleep × t_sleep) / T.  Battery life = C / I_avg.

        The plot shows how dramatically the wake interval affects average current
        and battery life for a 1000 mAh cell.
        """
    )
    return


@app.cell
def _(np, plt):
    _I_active_mA = 15.0      # mA while awake
    _I_sleep_uA  = 20.0      # μA in Stop mode
    _t_active_ms = 50.0      # ms active per wake cycle
    _C_mAh       = 1000.0    # battery capacity

    # Sweep wake interval from 1 s to 600 s
    _T_wake_s  = np.linspace(1, 600, 1000)
    _T_wake_ms = _T_wake_s * 1000

    _t_sleep_ms = np.maximum(_T_wake_ms - _t_active_ms, 0)
    _I_avg_mA   = (_I_active_mA * _t_active_ms
                   + (_I_sleep_uA / 1000) * _t_sleep_ms) / _T_wake_ms
    _life_h     = _C_mAh / _I_avg_mA
    _life_yr    = _life_h / 8760

    fig_pwr, (_ax_cur, _ax_life) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    _ax_cur.semilogy(_T_wake_s, _I_avg_mA, "b-", linewidth=2)
    _ax_cur.axhline(y=_I_sleep_uA / 1000, color="green", linestyle="--", alpha=0.6,
                    label=f"Sleep floor: {_I_sleep_uA} μA")

    for _Tex, _ccol, _clbl in [(60, "red", "60 s"), (5, "orange", "5 s")]:
        _tsl = _Tex * 1000 - _t_active_ms
        _Iex = (_I_active_mA * _t_active_ms
                + (_I_sleep_uA / 1000) * _tsl) / (_Tex * 1000)
        _ax_cur.plot(_Tex, _Iex, "o", color=_ccol, markersize=9, zorder=5)
        _ax_cur.annotate(
            f"T = {_Tex} s\nI_avg = {_Iex*1000:.0f} μA",
            xy=(_Tex, _Iex), xytext=(_Tex + 30, _Iex * 3),
            fontsize=9, color=_ccol,
            arrowprops=dict(arrowstyle="->", color=_ccol),
        )

    _ax_cur.set_ylabel("Average Current (mA)", fontsize=10)
    _ax_cur.set_title(
        "Battery-Powered Sensor: Average Current vs Wake Interval\n"
        f"(I_active = {_I_active_mA} mA for {_t_active_ms} ms, "
        f"I_sleep = {_I_sleep_uA} μA)",
        fontsize=11,
    )
    _ax_cur.legend(fontsize=9)
    _ax_cur.grid(True, alpha=0.3, which="both")

    _ax_life.plot(_T_wake_s, _life_yr, "r-", linewidth=2, label="Battery life")

    for _Tex, _ccol, _clbl in [(60, "red", "60 s"), (5, "orange", "5 s")]:
        _tsl = _Tex * 1000 - _t_active_ms
        _Iex = (_I_active_mA * _t_active_ms
                + (_I_sleep_uA / 1000) * _tsl) / (_Tex * 1000)
        _lex = (_C_mAh / _Iex) / 8760
        _ax_life.plot(_Tex, _lex, "o", color=_ccol, markersize=9, zorder=5)
        _ax_life.annotate(
            f"T = {_Tex} s\n{_lex:.2f} yr",
            xy=(_Tex, _lex), xytext=(_Tex + 30, _lex * 1.5),
            fontsize=9, color=_ccol,
            arrowprops=dict(arrowstyle="->", color=_ccol),
        )

    _ax_life.set_xlabel("Wake Interval (s)", fontsize=10)
    _ax_life.set_ylabel("Battery Life (years)", fontsize=10)
    _ax_life.set_title(f"Battery Life vs Wake Interval (C = {_C_mAh} mAh)", fontsize=11)
    _ax_life.legend(fontsize=9)
    _ax_life.grid(True, alpha=0.3)
    _ax_life.set_xlim(0, 600)
    _ax_life.set_ylim(0)

    fig_pwr.tight_layout()
    fig_pwr
    return fig_pwr


# --- 5.7.2 Low-Power Design: Frequency Scaling and Stop Mode ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 5.7.2 Low-Power Design — Frequency Scaling and Stop Mode

        An IoT device runs at 3.3 V.  The firmware divides time into three
        operating modes: heavy computation (168 MHz, 30 mA), communication
        (48 MHz with voltage scaling, 8 mA), and idle (Stop mode, 20 μA).

        Sweeping the idle fraction from 0 % to 90 % (while keeping
        computation fixed at 10 %) shows the dramatic savings from entering
        Stop mode during idle periods.
        """
    )
    return


@app.cell
def _(np, plt):
    _I_high = 30.0    # mA at 168 MHz
    _I_mid  =  8.0    # mA at 48 MHz
    _I_stop =  0.020  # mA Stop mode (20 μA)

    _f_compute = 0.10  # fixed 10 % at high speed

    _idle_frac = np.linspace(0, 0.90, 500)
    _comm_frac = 1.0 - _f_compute - _idle_frac

    _I_unopt = np.full_like(_idle_frac, _I_high)
    _I_opt   = (_f_compute * _I_high
                + _comm_frac  * _I_mid
                + _idle_frac  * _I_stop)
    _saving  = (_I_unopt - _I_opt) / _I_unopt * 100

    fig_opt, (_axc, _axs) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    _axc.plot(_idle_frac * 100, _I_unopt, "r-", linewidth=2,
              label="Unoptimized (always 168 MHz)")
    _axc.plot(_idle_frac * 100, _I_opt,   "b-", linewidth=2,
              label="Optimized (freq scaling + Stop mode)")

    _idle_ex = 0.70
    _comm_ex = 1.0 - _f_compute - _idle_ex
    _I_ex    = _f_compute * _I_high + _comm_ex * _I_mid + _idle_ex * _I_stop
    _axc.plot(_idle_ex * 100, _I_ex, "bo", markersize=10, zorder=5)
    _axc.annotate(
        f"Example §5.7.2\n70% idle → {_I_ex:.2f} mA\n"
        f"(vs {_I_high:.0f} mA unoptimized)",
        xy=(_idle_ex * 100, _I_ex),
        xytext=(_idle_ex * 100 - 35, _I_ex + 6),
        fontsize=9, color="blue",
        arrowprops=dict(arrowstyle="->", color="blue"),
    )
    _axc.set_ylabel("Average Current (mA)", fontsize=10)
    _axc.set_title(
        "IoT Device Average Current: Optimized vs Unoptimized\n"
        "(I_compute = 30 mA @ 168 MHz, I_comm = 8 mA @ 48 MHz, "
        "I_idle = 20 μA Stop mode)",
        fontsize=11,
    )
    _axc.legend(fontsize=9)
    _axc.grid(True, alpha=0.3)
    _axc.set_ylim(0, 35)

    _axs.plot(_idle_frac * 100, _saving, "g-", linewidth=2)
    _axs.fill_between(_idle_frac * 100, _saving, alpha=0.15, color="green")
    _saving_ex = (_I_high - _I_ex) / _I_high * 100
    _axs.plot(_idle_ex * 100, _saving_ex, "go", markersize=10, zorder=5)
    _axs.annotate(
        f"{_saving_ex:.1f}% saving at 70% idle",
        xy=(_idle_ex * 100, _saving_ex),
        xytext=(_idle_ex * 100 - 30, 68),
        fontsize=9, color="darkgreen",
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
    )
    _axs.set_xlabel("Idle Fraction (%)", fontsize=10)
    _axs.set_ylabel("Current Reduction (%)", fontsize=10)
    _axs.set_title("Power Saving (%) from Frequency Scaling + Stop Mode", fontsize=11)
    _axs.grid(True, alpha=0.3)
    _axs.set_xlim(0, 90)
    _axs.set_ylim(0, 100)

    fig_opt.tight_layout()
    fig_opt
    return fig_opt


if __name__ == "__main__":
    app.run()
