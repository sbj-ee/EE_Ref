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
        # Chapter 11: Instrumentation and Measurement — Example Visualizations

        Interactive graphs for selected topics from Chapter 11,
        covering accuracy vs precision, Wheatstone bridge response,
        and thermocouple voltage-temperature characteristics.
        """
    )
    return


# --- 11.1.1 Accuracy and Precision ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 11.1.1 Accuracy and Precision

        Four target plots illustrating the distinction between accuracy and
        precision. The bullseye center represents the true value. Accuracy
        describes how close the measurements cluster to the center, while
        precision describes how tightly they cluster together. A measurement
        system can be precise but inaccurate (tight grouping, offset from
        center), accurate but imprecise (scattered around center), or any
        combination of the two.
        """
    )
    return


@app.cell
def _(np, plt):
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 10))

    rng = np.random.default_rng(42)

    scenarios = [
        ("High Accuracy\nHigh Precision", 0.0, 0.0, 0.08),
        ("High Accuracy\nLow Precision", 0.0, 0.0, 0.35),
        ("Low Accuracy\nHigh Precision", 0.5, 0.4, 0.08),
        ("Low Accuracy\nLow Precision", 0.45, -0.35, 0.35),
    ]

    for _idx, (title, cx, cy, spread) in enumerate(scenarios):
        ax = axes1[_idx // 2][_idx % 2]

        # Draw target rings
        for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
            circle = plt.Circle((0, 0), r, fill=False, color="gray",
                                 linewidth=1, linestyle="-", alpha=0.5)
            ax.add_patch(circle)

        # Colored rings for visual appeal
        ring_colors = ["#FFD700", "#FFA500", "#FF6347", "#CD5C5C", "#DCDCDC"]
        for _i, r in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
            ring = plt.Circle((0, 0), r, fill=True, color=ring_colors[_i],
                               alpha=0.15, zorder=0)
            ax.add_patch(ring)

        # Bullseye center
        ax.plot(0, 0, "k+", markersize=15, markeredgewidth=2, zorder=3)

        # Scatter measurement points
        n_pts = 20
        x_pts = rng.normal(cx, spread, n_pts)
        y_pts = rng.normal(cy, spread, n_pts)
        ax.scatter(x_pts, y_pts, c="blue", s=50, edgecolors="navy",
                    linewidths=0.8, alpha=0.85, zorder=4)

        # Mark the mean
        ax.plot(np.mean(x_pts), np.mean(y_pts), "r^", markersize=12,
                 markeredgecolor="darkred", markeredgewidth=1.5, zorder=5,
                 label="Mean")

        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

        if _idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig1.suptitle("Accuracy vs Precision: Target Analogy",
                   fontsize=14, fontweight="bold", y=0.98)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    fig1
    return


# --- 11.1.5 Wheatstone Bridge ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 11.1.5 Wheatstone Bridge Response

        Output voltage of a Wheatstone bridge as the sensing resistor changes
        by ΔR/R from -10% to +10%, with all arms nominally R = 350 Ω (typical
        strain gauge value) and excitation voltage V_ex = 10 V. The exact
        nonlinear response is compared against the commonly used linear
        approximation V_out ≈ V_ex × (ΔR/R) / 4. The shaded region highlights
        the nonlinearity error that grows with increasing |ΔR/R|.
        """
    )
    return


@app.cell
def _(np, plt):
    R_nom = 350.0  # nominal resistance (ohms)
    V_ex = 10.0    # excitation voltage (V)

    dr_ratio = np.linspace(-0.10, 0.10, 500)  # delta_R / R

    # Exact Wheatstone bridge output (quarter-bridge, single active arm)
    # V_out = V_ex * (R+dR) / (R+dR+R) - V_ex * R / (R+R)
    # V_out = V_ex * [(R+dR)/(2R+dR) - 1/2]
    # V_out = V_ex * dR / (2*(2R + dR))
    R_active = R_nom * (1 + dr_ratio)
    v_exact = V_ex * (R_active / (R_active + R_nom) - 0.5)

    # Linear approximation: V_out ≈ V_ex * (dR/R) / 4
    v_linear = V_ex * dr_ratio / 4.0

    # Nonlinearity error
    v_error = (v_exact - v_linear) * 1000  # in mV

    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(10, 8),
                                        gridspec_kw={"height_ratios": [3, 1]})

    # Main plot: exact vs linear
    ax2a.plot(dr_ratio * 100, v_exact * 1000, "b-", linewidth=2.5,
              label="Exact (nonlinear)")
    ax2a.plot(dr_ratio * 100, v_linear * 1000, "r--", linewidth=2,
              label="Linear approximation")

    # Shade the difference
    ax2a.fill_between(dr_ratio * 100, v_exact * 1000, v_linear * 1000,
                       alpha=0.2, color="orange", label="Nonlinearity error")

    # Annotate key points
    for pct in [-10, -5, 5, 10]:
        _idx = np.argmin(np.abs(dr_ratio * 100 - pct))
        err_mv = v_error[_idx]
        ax2a.annotate(f"{err_mv:+.2f} mV",
                       xy=(pct, v_exact[_idx] * 1000),
                       xytext=(pct + (2 if pct > 0 else -2), v_exact[_idx] * 1000 + 8),
                       fontsize=8, color="darkorange",
                       arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8))

    ax2a.set_ylabel("Output Voltage (mV)", fontsize=11)
    ax2a.set_title(f"Wheatstone Bridge: Exact vs Linear Response (R = {R_nom:.0f} Ω, "
                    f"V_ex = {V_ex:.0f} V)", fontsize=12, fontweight="bold")
    ax2a.legend(fontsize=10, loc="upper left")
    ax2a.grid(True, alpha=0.3)
    ax2a.axhline(0, color="black", linewidth=0.5)
    ax2a.axvline(0, color="black", linewidth=0.5)

    # Error subplot
    ax2b.plot(dr_ratio * 100, v_error, "darkorange", linewidth=2)
    ax2b.fill_between(dr_ratio * 100, v_error, 0, alpha=0.2, color="orange")
    ax2b.set_xlabel("ΔR/R (%)", fontsize=11)
    ax2b.set_ylabel("Error (mV)", fontsize=11)
    ax2b.set_title("Nonlinearity Error (Exact − Linear)", fontsize=11)
    ax2b.grid(True, alpha=0.3)
    ax2b.axhline(0, color="black", linewidth=0.5)

    fig2.tight_layout()
    fig2
    return


# --- 11.2.1 Thermocouple Response ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 11.2.1 Thermocouple Voltage vs Temperature

        Voltage-temperature curves for five common thermocouple types (J, K, T,
        E, S) from 0°C to their respective maximum useful temperatures, using
        standard ITS-90 polynomial coefficients. Type E has the highest
        sensitivity (~68 μV/°C), making it ideal for cryogenic and moderate-
        temperature work; Type S (platinum-rhodium) extends to 1768°C for
        high-temperature furnace measurements but has much lower sensitivity
        (~6 μV/°C).
        """
    )
    return


@app.cell
def _(np, plt):
    # Standard thermocouple polynomial coefficients (NIST ITS-90)
    # Simplified polynomials for the positive temperature range (0°C reference)
    # Coefficients give voltage in millivolts for temperature in degrees C

    tc_data = {
        "J": {
            "max_temp": 760,
            "color": "#1f77b4",
            # Type J: 0 to 760°C (iron-constantan)
            "coeffs": [0.0, 5.0381187815e-02, 3.0475836930e-05,
                        -8.5681065720e-08, 1.3228195295e-10,
                        -1.7052958337e-13, 2.0948090697e-16,
                        -1.2538395336e-19, 1.5631725697e-23],
        },
        "K": {
            "max_temp": 1372,
            "color": "#ff7f0e",
            # Type K: 0 to 1372°C (chromel-alumel)
            "coeffs": [-1.7600413686e-02, 3.8921204975e-02, 1.8558770032e-05,
                        -9.9457592874e-08, 3.1840945719e-10,
                        -5.6072844889e-13, 5.6075059059e-16,
                        -3.2020720003e-19, 9.7151147152e-23,
                        -1.2104721275e-26],
        },
        "T": {
            "max_temp": 400,
            "color": "#2ca02c",
            # Type T: 0 to 400°C (copper-constantan)
            "coeffs": [0.0, 3.8748106364e-02, 3.3292227880e-05,
                        2.0618243404e-07, -2.1882256846e-09,
                        1.0996880928e-11, -3.0815758772e-14,
                        4.5479135290e-17, -2.7512901673e-20],
        },
        "E": {
            "max_temp": 1000,
            "color": "#d62728",
            # Type E: 0 to 1000°C (chromel-constantan)
            "coeffs": [0.0, 5.8665508708e-02, 4.5410977124e-05,
                        -7.7998048686e-08, 2.5800160612e-10,
                        -5.9452583057e-13, 9.3214058667e-16,
                        -8.1819730750e-19, 3.8003286862e-22,
                        -7.2893246250e-26],
        },
        "S": {
            "max_temp": 1768,
            "color": "#9467bd",
            # Type S: 0 to 1768°C (platinum-rhodium/platinum)
            # Using simplified coefficients for 0-1064°C range
            "coeffs": [0.0, 5.4030544256e-03, 1.2593428974e-05,
                        -2.3247937549e-08, 3.2203091293e-11,
                        -3.3145945973e-14, 2.5575883544e-17,
                        -1.2507891902e-20, 2.7144077078e-24],
        },
    }

    fig3, ax3 = plt.subplots(figsize=(12, 7))

    for tc_type, data in tc_data.items():
        t_max = data["max_temp"]
        coeffs = data["coeffs"]
        color = data["color"]

        temp = np.linspace(0, t_max, 500)
        voltage = np.zeros_like(temp)
        for _i, c in enumerate(coeffs):
            voltage += c * temp ** _i

        ax3.plot(temp, voltage, color=color, linewidth=2.5, label=f"Type {tc_type}")

        # Label at the end of each curve
        ax3.annotate(f"  {tc_type}", xy=(temp[-1], voltage[-1]),
                      fontsize=12, fontweight="bold", color=color,
                      va="center")

    # Mark typical sensitivity at 500°C for each type that reaches it
    ax3.axvline(500, color="gray", linestyle=":", alpha=0.4)
    ax3.text(510, -2, "500°C", fontsize=9, color="gray", va="top")

    ax3.set_xlabel("Temperature (°C)", fontsize=12)
    ax3.set_ylabel("Thermocouple Voltage (mV)", fontsize=12)
    ax3.set_title("Thermocouple Voltage vs Temperature (Reference Junction at 0°C)",
                   fontsize=13, fontweight="bold")
    ax3.legend(fontsize=11, loc="upper left")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1850)
    ax3.set_ylim(-2, 80)

    # Add sensitivity annotation
    ax3.text(1400, 72, "Sensitivity (at 25°C):", fontsize=9, fontweight="bold",
              color="black", va="top")
    sensitivities = [("E", "~68 μV/°C"), ("J", "~52 μV/°C"), ("K", "~41 μV/°C"),
                      ("T", "~43 μV/°C"), ("S", "~6 μV/°C")]
    for _i, (tc, sens) in enumerate(sensitivities):
        ax3.text(1400, 68 - _i * 4, f"  Type {tc}: {sens}", fontsize=8,
                  color=tc_data[tc]["color"], va="top")

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
