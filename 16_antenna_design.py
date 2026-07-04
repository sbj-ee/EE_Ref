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
        # Chapter 16: Antenna Design — Example Visualizations

        Interactive graph for phased array scanning and grating lobe analysis.
        """
    )
    return


# --- 16.5.3 Phased Array Pattern with Grating Lobes ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 16.5.3 Phased Array — Array Factor with Scanning and Grating Lobes

        A 32-element ULA at 10 GHz with d = 0.55λ steered to 45° from broadside.
        The n = −1 grating lobe is just outside visible space (sin θ = −1.111),
        but would enter at scan angles beyond 54.9°.
        """
    )
    return


@app.cell
def _(np, plt):
    N_elem = 32
    d_over_lam = 0.55  # element spacing / wavelength
    theta_scan_deg = 45  # scan angle from broadside

    theta = np.linspace(-90, 90, 4000)
    theta_rad = np.radians(theta)
    theta_scan_rad = np.radians(theta_scan_deg)

    # Progressive phase shift
    beta = -2 * np.pi * d_over_lam * np.sin(theta_scan_rad)

    # Array factor: AF = sin(N*psi/2) / (N*sin(psi/2))
    psi = 2 * np.pi * d_over_lam * np.sin(theta_rad) + beta
    # Avoid division by zero
    af_num = np.sin(N_elem * psi / 2)
    af_den = N_elem * np.sin(psi / 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        AF = np.where(np.abs(af_den) < 1e-10, 1.0, af_num / af_den)

    AF_dB = 20 * np.log10(np.abs(AF) + 1e-12)
    AF_dB = np.clip(AF_dB, -40, 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(theta, AF_dB, "b-", linewidth=1.5)
    ax.axvline(x=theta_scan_deg, color="red", linestyle="--", alpha=0.6,
               label=f"Main beam: θ₀ = {theta_scan_deg}°")
    ax.axhline(y=-3, color="gray", linestyle=":", alpha=0.5, label="−3 dB")

    # Mark grating lobe location (would be at sin_gl = sin(45) - 1/0.55 = -1.111)
    ax.annotate("Grating lobe\n(sin θ = −1.11)\noutside visible space",
                xy=(-90, -5), fontsize=9, color="orange",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Mark max grating-lobe-free scan
    ax.annotate("Max GL-free scan: ±54.9°",
                xy=(54.9, -15), fontsize=9, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))

    ax.set_xlabel("Angle from Broadside (°)")
    ax.set_ylabel("Array Factor (dB)")
    ax.set_title(f"Phased Array Pattern: {N_elem} elements, d = {d_over_lam}λ, scanned to {theta_scan_deg}°")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 2)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
