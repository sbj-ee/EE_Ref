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
        # Chapter 3: Semiconductors — Example Visualizations

        Interactive graphs for selected topics from Chapter 3, covering
        energy band diagrams, PN junction I-V characteristics, and
        MOSFET output and transfer characteristics.
        """
    )
    return


# --- 3.1.1 Energy Band Diagram ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 3.1.1 Energy Band Diagram — Conductor, Semiconductor, Insulator

        The electrical behavior of a solid is determined by its energy band
        structure. In a conductor, the valence and conduction bands overlap,
        providing abundant free carriers. In a semiconductor like silicon,
        a small bandgap (~1.12 eV) allows thermal excitation of electrons
        into the conduction band. In an insulator like SiO2, the bandgap
        is so large (~9 eV) that virtually no electrons reach the conduction
        band at room temperature.
        """
    )
    return


@app.cell
def _(plt):
    fig1, axes = plt.subplots(1, 3, figsize=(12, 5))

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
        ax.set_ylabel("Energy (eV)", fontsize=10) if ax == axes[0] else None
        ax.set_title(title, fontsize=10, fontweight="bold")

        if overlap:
            # Conductor: overlapping bands
            # Valence band (filled)
            ax.fill_between([1, 9], 2, 6, color="#4a90d9", alpha=0.5)
            ax.text(5, 4.0, "Valence\nBand", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#1a3d6e")
            # Conduction band overlapping
            ax.fill_between([1, 9], 4.5, 8.5, color="#e8a040", alpha=0.4)
            ax.text(5, 7.5, "Conduction\nBand", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#8b5e00")
            # Overlap region
            ax.annotate("Overlap", xy=(5, 5.25), fontsize=9, ha="center",
                        va="center", color="#cc3333", fontweight="bold")
            ax.annotate("", xy=(7.5, 6), xytext=(7.5, 4.5),
                        arrowprops=dict(arrowstyle="<->", color="#cc3333",
                                        linewidth=1.5))
            # Fermi level
            ax.axhline(y=5.25, xmin=0.1, xmax=0.9, color="black",
                        linestyle="--", linewidth=1.2)
            ax.text(9.2, 5.25, "E\u1da0", fontsize=9, va="center")
        else:
            # Valence band (filled)
            vb_top = 4.0
            vb_bottom = 1.0
            ax.fill_between([1, 9], vb_bottom, vb_top, color="#4a90d9",
                            alpha=0.5)
            ax.text(5, (vb_top + vb_bottom) / 2, "Valence\nBand",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="#1a3d6e")

            # Conduction band (empty)
            cb_bottom = vb_top + eg * (8.0 / 9.0)  # scale for display
            cb_top = cb_bottom + 3.0
            ax.fill_between([1, 9], cb_bottom, cb_top, color="#e8a040",
                            alpha=0.3)
            ax.text(5, (cb_top + cb_bottom) / 2, "Conduction\nBand",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                    color="#8b5e00")

            # Bandgap annotation
            mid_x = 5
            ax.annotate("", xy=(mid_x + 3, cb_bottom),
                        xytext=(mid_x + 3, vb_top),
                        arrowprops=dict(arrowstyle="<->", color="#cc3333",
                                        linewidth=1.5))
            gap_label = f"E\u2091 = {eg} eV" if eg < 5 else f"E\u2091 \u2248 {eg:.0f} eV"
            ax.text(mid_x + 3.2, (vb_top + cb_bottom) / 2, gap_label,
                    fontsize=9, va="center", ha="left", color="#cc3333",
                    fontweight="bold")

            # Bandgap shading
            ax.fill_between([1, 9], vb_top, cb_bottom, color="#ffcccc",
                            alpha=0.3, hatch="//")
            ax.text(2.5, (vb_top + cb_bottom) / 2, "Bandgap",
                    ha="center", va="center", fontsize=8, color="#cc3333",
                    style="italic")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    fig1.suptitle("Energy Band Diagrams: Conductor vs Semiconductor vs Insulator",
                  fontsize=12, fontweight="bold", y=1.02)
    fig1.tight_layout()
    fig1
    return


# --- 3.3 PN Junction I-V Curve ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 3.3 PN Junction I-V Curve — Shockley Diode Equation

        The current through a PN junction diode follows the Shockley equation:
        I = I_S * (e^(V / (n * V_T)) - 1), where I_S is the reverse saturation
        current, n is the ideality factor, and V_T = kT/q ~ 25.85 mV at 300 K.
        An ideal diode has n = 1 (diffusion current dominates), while n = 2
        represents significant recombination current. The left plot uses a
        linear scale showing the exponential rise and knee voltage (~0.7 V);
        the right plot uses a log scale to reveal the exponential region as
        a straight line.
        """
    )
    return


@app.cell
def _(np, plt):
    I_S = 2e-14   # reverse saturation current (A)
    V_T = 0.02585  # thermal voltage at 300 K (V)

    # Forward bias region
    V_fwd = np.linspace(0, 0.85, 2000)
    # Reverse bias region
    V_rev = np.linspace(-2.0, 0, 500)
    # Full range
    V_full = np.concatenate([V_rev, V_fwd[1:]])

    # Shockley equation for n=1 and n=2
    I_n1_full = I_S * (np.exp(V_full / (1 * V_T)) - 1)
    I_n2_full = I_S * (np.exp(V_full / (2 * V_T)) - 1)

    # Clip for display
    I_n1_clip = np.clip(I_n1_full, -1e-10, 0.1)
    I_n2_clip = np.clip(I_n2_full, -1e-10, 0.1)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Linear scale ---
    ax2a.plot(V_full * 1000, I_n1_clip * 1000, "b-", linewidth=2,
              label="n = 1 (ideal)")
    ax2a.plot(V_full * 1000, I_n2_clip * 1000, "r--", linewidth=2,
              label="n = 2 (recombination)")
    ax2a.axhline(y=0, color="black", linewidth=0.5)
    ax2a.axvline(x=0, color="black", linewidth=0.5)

    # Mark knee voltage for n=1
    knee_v = 700  # mV
    knee_idx = np.argmin(np.abs(V_full * 1000 - knee_v))
    knee_i = I_n1_clip[knee_idx] * 1000
    ax2a.plot(knee_v, knee_i, "ko", markersize=8, zorder=5)
    ax2a.annotate(f"Knee \u2248 0.7 V\n({knee_i:.1f} mA)",
                  xy=(knee_v, knee_i),
                  xytext=(450, knee_i * 0.85),
                  fontsize=9, color="black",
                  arrowprops=dict(arrowstyle="->", color="black"),
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                            alpha=0.8))

    # Mark reverse saturation
    ax2a.annotate(f"I\u209b \u2248 {I_S:.0e} A\n(leakage)",
                  xy=(-1500, -I_S * 1000),
                  xytext=(-1500, 20),
                  fontsize=9, color="gray",
                  arrowprops=dict(arrowstyle="->", color="gray"))

    ax2a.set_xlabel("Voltage (mV)", fontsize=10)
    ax2a.set_ylabel("Current (mA)", fontsize=10)
    ax2a.set_title("PN Junction I-V (Linear Scale)", fontsize=11,
                    fontweight="bold")
    ax2a.legend(fontsize=9, loc="upper left")
    ax2a.grid(True, alpha=0.3)
    ax2a.set_xlim(-2100, 900)
    ax2a.set_ylim(-5, 100)

    # --- Right: Log scale (forward bias only) ---
    V_log = np.linspace(0.1, 0.80, 1500)
    I_n1_log = I_S * (np.exp(V_log / (1 * V_T)) - 1)
    I_n2_log = I_S * (np.exp(V_log / (2 * V_T)) - 1)

    ax2b.semilogy(V_log * 1000, I_n1_log, "b-", linewidth=2,
                  label="n = 1 (ideal)")
    ax2b.semilogy(V_log * 1000, I_n2_log, "r--", linewidth=2,
                  label="n = 2 (recombination)")

    # Annotate slope difference
    ax2b.annotate("Slope = q/(nkT)\nn=1: 38.7 /V\nn=2: 19.3 /V",
                  xy=(350, 1e-6), fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                            alpha=0.8))

    # Mark knee voltage on log plot
    knee_idx_log = np.argmin(np.abs(V_log * 1000 - knee_v))
    ax2b.axvline(x=knee_v, color="gray", linestyle=":", alpha=0.7)
    ax2b.text(knee_v + 10, 1e-12, "0.7 V", fontsize=9, color="gray",
              rotation=90, va="bottom")

    ax2b.set_xlabel("Voltage (mV)", fontsize=10)
    ax2b.set_ylabel("Current (A)", fontsize=10)
    ax2b.set_title("PN Junction I-V (Log Scale — Forward Bias)",
                    fontsize=11, fontweight="bold")
    ax2b.legend(fontsize=9, loc="upper left")
    ax2b.grid(True, alpha=0.3, which="both")
    ax2b.set_xlim(100, 820)
    ax2b.set_ylim(1e-14, 1)

    fig2.tight_layout()
    fig2
    return


# --- 3.4 MOSFET Characteristics ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 3.5.2 MOSFET I-V Characteristics

        An N-channel enhancement MOSFET with threshold voltage V_th = 1 V
        and transconductance parameter k_n = 1 mA/V^2. The left plot shows
        output characteristics (I_DS vs V_DS) for several gate voltages,
        with the parabolic boundary V_DS = V_GS - V_th separating the
        linear (triode) and saturation regions. The right plot shows the
        transfer characteristic (I_DS vs V_GS) at V_DS = 5 V, illustrating
        the square-law relationship I_D = (k_n/2)(V_GS - V_th)^2.
        """
    )
    return


@app.cell
def _(np, plt):
    V_th = 1.0       # threshold voltage (V)
    k_n = 1.0e-3     # transconductance parameter (A/V^2)

    V_DS = np.linspace(0, 8, 500)
    V_GS_values = [2.0, 3.0, 4.0, 5.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Output characteristics (I_DS vs V_DS) ---
    for v_gs, color in zip(V_GS_values, colors):
        v_ov = v_gs - V_th  # overdrive voltage
        I_D = np.where(
            V_DS < v_ov,
            # Linear (triode) region
            k_n * ((v_gs - V_th) * V_DS - 0.5 * V_DS**2),
            # Saturation region
            0.5 * k_n * (v_gs - V_th)**2
        )
        ax3a.plot(V_DS, I_D * 1000, color=color, linewidth=2,
                  label=f"V\u0047\u0053 = {v_gs:.0f} V")

    # Parabolic boundary: V_DS = V_GS - V_th, I_D = k_n/2 * V_DS^2
    v_ds_boundary = np.linspace(0, max(V_GS_values) - V_th, 200)
    i_boundary = 0.5 * k_n * v_ds_boundary**2
    ax3a.plot(v_ds_boundary, i_boundary * 1000, "k--", linewidth=1.5,
              alpha=0.6, label="V\u1d30\u209b = V\u1d33\u209b \u2212 V\u209c\u2095")

    # Region labels
    ax3a.text(1.0, 7.0, "Linear\n(Triode)", fontsize=9, ha="center",
              style="italic", color="gray")
    ax3a.text(5.5, 7.0, "Saturation", fontsize=9, ha="center",
              style="italic", color="gray")

    ax3a.set_xlabel("V\u1d30\u209b (V)", fontsize=10)
    ax3a.set_ylabel("I\u1d30 (mA)", fontsize=10)
    ax3a.set_title("MOSFET Output Characteristics", fontsize=11,
                    fontweight="bold")
    ax3a.legend(fontsize=9, loc="center right")
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xlim(0, 8)
    ax3a.set_ylim(0, 9)

    # --- Right: Transfer characteristic (I_DS vs V_GS) at V_DS = 5 V ---
    V_GS_sweep = np.linspace(0, 6, 500)
    V_DS_fixed = 5.0

    I_D_transfer = np.where(
        V_GS_sweep <= V_th,
        0,
        np.where(
            V_DS_fixed < (V_GS_sweep - V_th),
            # Linear region (only at very high V_GS)
            k_n * ((V_GS_sweep - V_th) * V_DS_fixed - 0.5 * V_DS_fixed**2),
            # Saturation region
            0.5 * k_n * (V_GS_sweep - V_th)**2
        )
    )

    ax3b.plot(V_GS_sweep, I_D_transfer * 1000, "b-", linewidth=2.5)

    # Mark V_th
    ax3b.axvline(x=V_th, color="red", linestyle="--", linewidth=1.2,
                 alpha=0.7)
    ax3b.annotate(f"V\u209c\u2095 = {V_th:.0f} V",
                  xy=(V_th, 0.3),
                  xytext=(V_th + 0.8, 1.5),
                  fontsize=10, color="red", fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color="red"))

    # Mark square-law relationship
    ax3b.annotate("I\u1d30 = (k\u2099/2)(V\u1d33\u209b \u2212 V\u209c\u2095)\u00b2",
                  xy=(3.5, 0.5 * k_n * (3.5 - V_th)**2 * 1000),
                  xytext=(1.5, 7.5),
                  fontsize=10, color="blue",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                            alpha=0.8))

    # Mark a few operating points
    for v_gs in [2.0, 3.0, 4.0, 5.0]:
        i_d = 0.5 * k_n * (v_gs - V_th)**2 * 1000
        ax3b.plot(v_gs, i_d, "ko", markersize=5, zorder=5)
        ax3b.text(v_gs + 0.1, i_d + 0.2, f"{i_d:.1f} mA", fontsize=8,
                  color="black")

    ax3b.set_xlabel("V\u1d33\u209b (V)", fontsize=10)
    ax3b.set_ylabel("I\u1d30 (mA)", fontsize=10)
    ax3b.set_title(f"MOSFET Transfer Characteristic (V\u1d30\u209b = {V_DS_fixed:.0f} V)",
                    fontsize=11, fontweight="bold")
    ax3b.grid(True, alpha=0.3)
    ax3b.set_xlim(0, 6)
    ax3b.set_ylim(0, 13)

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
