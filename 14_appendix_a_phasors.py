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
        # Appendix A: Imaginary Numbers and Phasors — Visualizations

        Interactive graphs for selected examples from Appendix A,
        showing complex plane plots, phasor diagrams, and power triangles.
        """
    )
    return


# --- A.3.1 Polar Form: Complex plane plot ---

@app.cell
def _(mo):
    mo.md(
        """
        ## A.3.1 Polar Form — Complex Plane Plot

        The complex number Z = −3 + j4 lies in quadrant II of the complex plane.
        Its magnitude is |Z| = 5 and its angle is θ = 126.87°. The plot shows
        the rectangular components (real and imaginary projections) and the
        polar representation (magnitude and angle).
        """
    )
    return


@app.cell
def _(np, plt):
    Z_polar = -3 + 4j
    mag_polar = abs(Z_polar)
    angle_polar = np.degrees(np.angle(Z_polar))

    fig1, ax1 = plt.subplots(figsize=(7, 7))

    # Draw axes
    ax1.axhline(y=0, color="k", linewidth=0.5)
    ax1.axvline(x=0, color="k", linewidth=0.5)

    # Draw vector
    ax1.annotate("", xy=(Z_polar.real, Z_polar.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
    ax1.plot(Z_polar.real, Z_polar.imag, "bo", markersize=10, zorder=5)
    ax1.annotate(f"Z = {Z_polar.real:.0f} + j{Z_polar.imag:.0f}\n|Z| = {mag_polar:.0f}, θ = {angle_polar:.1f}°",
                 xy=(Z_polar.real, Z_polar.imag),
                 xytext=(Z_polar.real - 2.5, Z_polar.imag + 0.5),
                 fontsize=11, color="blue")

    # Draw projections
    ax1.plot([Z_polar.real, Z_polar.real], [0, Z_polar.imag], "r--", linewidth=1, alpha=0.6)
    ax1.plot([0, Z_polar.real], [Z_polar.imag, Z_polar.imag], "r--", linewidth=1, alpha=0.6)
    ax1.annotate(f"Re = {Z_polar.real:.0f}", xy=(Z_polar.real / 2, -0.5), fontsize=10, color="red", ha="center")
    ax1.annotate(f"Im = {Z_polar.imag:.0f}", xy=(0.3, Z_polar.imag / 2), fontsize=10, color="red")

    # Draw angle arc
    theta_arc = np.linspace(0, np.radians(angle_polar), 50)
    r_arc = 1.5
    ax1.plot(r_arc * np.cos(theta_arc), r_arc * np.sin(theta_arc), "g-", linewidth=1.5)
    ax1.annotate(f"θ = {angle_polar:.1f}°", xy=(r_arc * np.cos(np.radians(angle_polar / 2)),
                 r_arc * np.sin(np.radians(angle_polar / 2))),
                 xytext=(-0.5, 2), fontsize=10, color="green")

    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_xlabel("Real")
    ax1.set_ylabel("Imaginary")
    ax1.set_title("Complex Plane: Z = −3 + j4")
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1
    return


# --- A.4.3 Phasor Arithmetic: Vector addition ---

@app.cell
def _(mo):
    mo.md(
        """
        ## A.4.3 Phasor Arithmetic — Vector Addition

        Two voltage sources in series produce V₁ = 100∠0° V and V₂ = 60∠90° V.
        The total voltage V_total = 100 + j60 = 116.6∠30.96° V is the vector sum,
        which is less than the arithmetic sum (160 V) because they are out of phase.
        """
    )
    return


@app.cell
def _(np, plt):
    V1_pa = 100 + 0j
    V2_pa = 0 + 60j
    Vtot_pa = V1_pa + V2_pa

    fig2, ax2 = plt.subplots(figsize=(8, 7))

    ax2.axhline(y=0, color="k", linewidth=0.5)
    ax2.axvline(x=0, color="k", linewidth=0.5)

    # V1 from origin
    ax2.annotate("", xy=(V1_pa.real, V1_pa.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
    ax2.annotate("V₁ = 100∠0° V", xy=(50, -8), fontsize=11, color="blue")

    # V2 from tip of V1
    ax2.annotate("", xy=(Vtot_pa.real, Vtot_pa.imag), xytext=(V1_pa.real, V1_pa.imag),
                 arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    ax2.annotate("V₂ = 60∠90° V", xy=(V1_pa.real + 3, V1_pa.imag + 25), fontsize=11, color="red")

    # V_total from origin
    ax2.annotate("", xy=(Vtot_pa.real, Vtot_pa.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
    mag_tot = abs(Vtot_pa)
    ang_tot = np.degrees(np.angle(Vtot_pa))
    ax2.annotate(f"V_total = {mag_tot:.1f}∠{ang_tot:.1f}° V",
                 xy=(Vtot_pa.real / 2 - 15, Vtot_pa.imag / 2 + 5),
                 fontsize=11, color="green", fontweight="bold")

    # Dashed lines showing rectangle
    ax2.plot([Vtot_pa.real, Vtot_pa.real], [0, Vtot_pa.imag], "gray", linestyle=":", alpha=0.5)
    ax2.plot([0, Vtot_pa.real], [Vtot_pa.imag, Vtot_pa.imag], "gray", linestyle=":", alpha=0.5)

    ax2.set_xlim(-20, 130)
    ax2.set_ylim(-20, 80)
    ax2.set_xlabel("Real (V)")
    ax2.set_ylabel("Imaginary (V)")
    ax2.set_title("Phasor Addition: V₁ + V₂ = V_total")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2
    return


# --- A.4.4 Phasor Diagrams: Series RL circuit ---

@app.cell
def _(mo):
    mo.md(
        """
        ## A.4.4 Phasor Diagram — Series RL Circuit

        In a series RL circuit with I = 5∠0° A, R = 30 Ω, and X_L = 40 Ω:
        V_R = 150∠0° V (in phase with current), V_L = 200∠90° V (leads current by 90°),
        and V_total = 250∠53.13° V. The phasor diagram shows the voltage triangle.
        """
    )
    return


@app.cell
def _(np, plt):
    I_phasor = 5 + 0j
    R_pd = 30
    XL_pd = 40
    VR_pd = I_phasor * R_pd
    VL_pd = I_phasor * 1j * XL_pd
    Vtot_pd = VR_pd + VL_pd

    fig3, ax3 = plt.subplots(figsize=(8, 7))
    ax3.axhline(y=0, color="k", linewidth=0.5)
    ax3.axvline(x=0, color="k", linewidth=0.5)

    # Current (reference, scaled for visibility)
    I_scale = 20  # scale factor for display
    ax3.annotate("", xy=(I_phasor.real * I_scale, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="orange", lw=2))
    ax3.annotate("I = 5∠0° A", xy=(I_phasor.real * I_scale / 2, -15), fontsize=10, color="orange")

    # V_R (in phase with I)
    ax3.annotate("", xy=(VR_pd.real, VR_pd.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
    ax3.annotate("V_R = 150∠0° V", xy=(VR_pd.real / 2, -15), fontsize=10, color="blue", ha="center")

    # V_L (from tip of V_R, 90° leading)
    ax3.annotate("", xy=(Vtot_pd.real, Vtot_pd.imag), xytext=(VR_pd.real, VR_pd.imag),
                 arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    ax3.annotate("V_L = 200∠90° V", xy=(VR_pd.real + 5, VR_pd.imag + VL_pd.imag / 2),
                 fontsize=10, color="red")

    # V_total (from origin)
    ax3.annotate("", xy=(Vtot_pd.real, Vtot_pd.imag), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
    mag_vtot = abs(Vtot_pd)
    ang_vtot = np.degrees(np.angle(Vtot_pd))
    ax3.annotate(f"V_total = {mag_vtot:.0f}∠{ang_vtot:.1f}° V",
                 xy=(Vtot_pd.real / 2 - 30, Vtot_pd.imag / 2 + 10),
                 fontsize=11, color="green", fontweight="bold")

    # Angle arc
    theta_arc3 = np.linspace(0, np.radians(ang_vtot), 50)
    r_arc3 = 40
    ax3.plot(r_arc3 * np.cos(theta_arc3), r_arc3 * np.sin(theta_arc3), "g-", linewidth=1.5)
    ax3.annotate(f"θ = {ang_vtot:.1f}°", xy=(45, 15), fontsize=10, color="green")

    ax3.set_xlim(-30, 200)
    ax3.set_ylim(-30, 230)
    ax3.set_xlabel("Real (V)")
    ax3.set_ylabel("Imaginary (V)")
    ax3.set_title("Phasor Diagram: Series RL Circuit")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3
    return


# --- A.5.3 Power Triangle ---

@app.cell
def _(mo):
    mo.md(
        """
        ## A.5.3 Power in Phasor Form — Power Triangle

        A load draws I = 10∠−25° A from V = 240∠0° V. The complex power
        S = 2400∠25° VA decomposes into P = 2175 W (real) and Q = 1014 VAR
        (reactive, inductive). The power triangle visualizes this relationship.
        """
    )
    return


@app.cell
def _(np, plt):
    S_mag = 2400  # VA
    S_angle = 25  # degrees
    P_pt = S_mag * np.cos(np.radians(S_angle))  # 2175 W
    Q_pt = S_mag * np.sin(np.radians(S_angle))  # 1014 VAR

    fig4, ax4 = plt.subplots(figsize=(8, 6))

    # Draw the power triangle
    # P along horizontal
    ax4.annotate("", xy=(P_pt, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.5))
    ax4.annotate(f"P = {P_pt:.0f} W", xy=(P_pt / 2, -80), fontsize=12, color="blue", ha="center")

    # Q vertical from P
    ax4.annotate("", xy=(P_pt, Q_pt), xytext=(P_pt, 0),
                 arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    ax4.annotate(f"Q = {Q_pt:.0f} VAR", xy=(P_pt + 50, Q_pt / 2), fontsize=12, color="red")

    # S (hypotenuse)
    ax4.annotate("", xy=(P_pt, Q_pt), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5))
    ax4.annotate(f"S = {S_mag:.0f} VA", xy=(P_pt / 2 - 200, Q_pt / 2 + 50),
                 fontsize=12, color="green", fontweight="bold")

    # Angle arc
    theta_arc4 = np.linspace(0, np.radians(S_angle), 50)
    r_arc4 = 400
    ax4.plot(r_arc4 * np.cos(theta_arc4), r_arc4 * np.sin(theta_arc4), "g-", linewidth=1.5)
    ax4.annotate(f"φ = {S_angle}°\npf = {np.cos(np.radians(S_angle)):.3f}",
                 xy=(450, 80), fontsize=10, color="green")

    # Right angle marker
    sq_size = 60
    ax4.plot([P_pt - sq_size, P_pt - sq_size, P_pt], [0, sq_size, sq_size], "k-", linewidth=1)

    ax4.set_xlim(-200, 2700)
    ax4.set_ylim(-200, 1400)
    ax4.set_xlabel("Real Power (W)")
    ax4.set_ylabel("Reactive Power (VAR)")
    ax4.set_title("Power Triangle: S = P + jQ")
    ax4.set_aspect("equal")
    ax4.grid(True, alpha=0.3)

    fig4.tight_layout()
    fig4
    return


if __name__ == "__main__":
    app.run()
