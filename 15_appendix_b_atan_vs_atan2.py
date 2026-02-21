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
        # Appendix B: Arctangent and atan2 — Visualizations

        Interactive graphs illustrating the quadrant ambiguity of arctan
        and how atan2 correctly resolves angles in all four quadrants.
        """
    )
    return


# --- B.1.2 Quadrant Ambiguity ---

@app.cell
def _(mo):
    mo.md(
        """
        ## B.1.2 Quadrant Ambiguity

        Z₁ = 3 + j4 (quadrant I) and Z₂ = −3 − j4 (quadrant III) both produce
        arctan(4/3) = 53.13°, despite being 180° apart. The plain arctan function
        cannot distinguish between them because dividing b/a loses the individual signs.
        """
    )
    return


@app.cell
def _(np, plt):
    Z1_qa = 3 + 4j
    Z2_qa = -3 - 4j

    atan_angle = np.degrees(np.arctan(4 / 3))  # 53.13° for both
    atan2_Z1 = np.degrees(np.arctan2(4, 3))  # 53.13°
    atan2_Z2 = np.degrees(np.arctan2(-4, -3))  # -126.87°

    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax in (ax1a, ax1b):
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Left: arctan gives SAME angle for both
    ax1a.set_title("arctan(b/a) — WRONG for Z₂")
    # Z1
    ax1a.annotate("", xy=(Z1_qa.real, Z1_qa.imag), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="blue", lw=2))
    ax1a.plot(Z1_qa.real, Z1_qa.imag, "bo", markersize=8)
    ax1a.annotate(f"Z₁ = 3+j4\narctan = {atan_angle:.1f}° ✓",
                  xy=(Z1_qa.real, Z1_qa.imag), xytext=(3.5, 5), fontsize=9, color="blue")
    # Z2 with WRONG arctan angle
    ax1a.annotate("", xy=(Z2_qa.real, Z2_qa.imag), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax1a.plot(Z2_qa.real, Z2_qa.imag, "ro", markersize=8)
    ax1a.annotate(f"Z₂ = −3−j4\narctan = {atan_angle:.1f}° ✗",
                  xy=(Z2_qa.real, Z2_qa.imag), xytext=(-5.5, -5.5), fontsize=9, color="red")
    # Show the wrong angle direction for Z2
    theta_wrong = np.linspace(0, np.radians(atan_angle), 30)
    ax1a.plot(1.5 * np.cos(theta_wrong), 1.5 * np.sin(theta_wrong), "r--", linewidth=1.5)
    ax1a.annotate(f"Both → {atan_angle:.1f}°!", xy=(1.2, 1.2), fontsize=10, color="red", fontweight="bold")
    ax1a.set_xlabel("Real")
    ax1a.set_ylabel("Imaginary")

    # Right: atan2 gives CORRECT angles
    ax1b.set_title("atan2(b, a) — CORRECT for both")
    ax1b.annotate("", xy=(Z1_qa.real, Z1_qa.imag), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="blue", lw=2))
    ax1b.plot(Z1_qa.real, Z1_qa.imag, "bo", markersize=8)
    ax1b.annotate(f"Z₁: atan2 = {atan2_Z1:.1f}° ✓",
                  xy=(Z1_qa.real, Z1_qa.imag), xytext=(3.5, 5), fontsize=9, color="blue")

    ax1b.annotate("", xy=(Z2_qa.real, Z2_qa.imag), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color="red", lw=2))
    ax1b.plot(Z2_qa.real, Z2_qa.imag, "ro", markersize=8)
    ax1b.annotate(f"Z₂: atan2 = {atan2_Z2:.1f}° ✓",
                  xy=(Z2_qa.real, Z2_qa.imag), xytext=(-5.5, -5.5), fontsize=9, color="red")

    # Show correct angle arcs
    theta_z1 = np.linspace(0, np.radians(atan2_Z1), 30)
    ax1b.plot(1.5 * np.cos(theta_z1), 1.5 * np.sin(theta_z1), "b-", linewidth=1.5)
    theta_z2 = np.linspace(0, np.radians(atan2_Z2), 30)
    ax1b.plot(1.5 * np.cos(theta_z2), 1.5 * np.sin(theta_z2), "r-", linewidth=1.5)
    ax1b.set_xlabel("Real")
    ax1b.set_ylabel("Imaginary")

    fig1.tight_layout()
    fig1
    return


# --- B.2.1 atan2 Full Circle: Four quadrants ---

@app.cell
def _(mo):
    mo.md(
        """
        ## B.2.1 atan2 — Full-Circle Coverage

        Four points, one in each quadrant, demonstrate that atan2 returns distinct,
        correct angles spanning the full −180° to +180° range. Plain arctan would
        return only two distinct values (53.13° and −53.13°) for all four points.
        """
    )
    return


@app.cell
def _(np, plt):
    points = [
        (3, 4, "Q-I"),
        (-3, 4, "Q-II"),
        (-3, -4, "Q-III"),
        (3, -4, "Q-IV"),
    ]
    colors_fc = ["blue", "green", "red", "orange"]

    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.axhline(y=0, color="k", linewidth=0.5)
    ax2.axvline(x=0, color="k", linewidth=0.5)

    for (_a, _b, _label), _color in zip(points, colors_fc):
        _angle = np.degrees(np.arctan2(_b, _a))
        _atan_only = np.degrees(np.arctan(_b / _a))

        # Draw vector
        ax2.annotate("", xy=(_a, _b), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=_color, lw=2))
        ax2.plot(_a, _b, "o", color=_color, markersize=10, zorder=5)

        # Label with both angles
        _offset_x = 0.5 if _a > 0 else -3.5
        _offset_y = 0.5 if _b > 0 else -1.0
        ax2.annotate(f"{_label}: ({_a}, {_b})\natan2 = {_angle:.1f}°\narctan = {_atan_only:.1f}°",
                     xy=(_a + _offset_x, _b + _offset_y), fontsize=9, color=_color)

        # Draw angle arc
        _theta_arc = np.linspace(0, np.radians(_angle), 50)
        _r = 1.5
        ax2.plot(_r * np.cos(_theta_arc), _r * np.sin(_theta_arc), color=_color, linewidth=1.5, alpha=0.7)

    # Draw unit circle for reference
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax2.plot(5 * np.cos(theta_circle), 5 * np.sin(theta_circle), "k:", alpha=0.2, linewidth=0.5)

    ax2.set_xlim(-7, 7)
    ax2.set_ylim(-7, 7)
    ax2.set_xlabel("Real (a)")
    ax2.set_ylabel("Imaginary (b)")
    ax2.set_title("atan2 Returns Correct Angles in All Four Quadrants")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2
    return


# --- B.2.3 Special Cases: Axis points ---

@app.cell
def _(mo):
    mo.md(
        """
        ## B.2.3 Special Cases — Points on the Axes

        atan2 correctly handles the four axis cases where arctan(b/a) would
        encounter division by zero or ambiguity. These correspond to pure
        resistance (0°), pure inductance (+90°), pure negative resistance (±180°),
        and pure capacitance (−90°) in impedance analysis.
        """
    )
    return


@app.cell
def _(np, plt):
    axis_points = [
        (1, 0, "Pure R: 0°", "blue"),
        (0, 1, "Pure L: +90°", "red"),
        (-1, 0, "−R: ±180°", "green"),
        (0, -1, "Pure C: −90°", "orange"),
    ]

    fig3, ax3 = plt.subplots(figsize=(7, 7))
    ax3.axhline(y=0, color="k", linewidth=0.5)
    ax3.axvline(x=0, color="k", linewidth=0.5)

    # Unit circle
    theta_uc = np.linspace(0, 2 * np.pi, 200)
    ax3.plot(np.cos(theta_uc), np.sin(theta_uc), "k-", alpha=0.15, linewidth=1)

    for _a, _b, _label, _color in axis_points:
        _angle = np.degrees(np.arctan2(_b, _a))
        ax3.annotate("", xy=(_a, _b), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=_color, lw=3))
        ax3.plot(_a, _b, "o", color=_color, markersize=12, zorder=5)

        # Label positioning
        if _a == 1:
            ax3.annotate(f"{_label}\natan2({_b},{_a}) = {_angle:.0f}°",
                         xy=(_a + 0.1, _b + 0.1), fontsize=10, color=_color)
        elif _a == -1:
            ax3.annotate(f"{_label}\natan2({_b},{_a}) = {_angle:.0f}°",
                         xy=(_a - 0.1, _b + 0.15), fontsize=10, color=_color, ha="right")
        elif _b == 1:
            ax3.annotate(f"{_label}\natan2({_b},{_a}) = {_angle:.0f}°",
                         xy=(_a + 0.1, _b + 0.1), fontsize=10, color=_color)
        else:
            ax3.annotate(f"{_label}\natan2({_b},{_a}) = {_angle:.0f}°",
                         xy=(_a + 0.1, _b - 0.3), fontsize=10, color=_color)

    ax3.set_xlim(-1.8, 1.8)
    ax3.set_ylim(-1.8, 1.8)
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imaginary")
    ax3.set_title("atan2 Special Cases: Points on the Axes")
    ax3.set_aspect("equal")
    ax3.grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
