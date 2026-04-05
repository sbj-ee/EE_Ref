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
        # Chapter 19: Engineering Economics — Example Visualizations

        Interactive graphs for selected example problems from Chapter 19,
        covering compound interest, depreciation methods, and sensitivity analysis.
        """
    )
    return


# --- 19.1.2 Compound Interest Growth ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 19.1.2 Simple vs Compound vs Continuous Interest

        Starting with P = $100,000 at r = 5% nominal annual rate, the three
        interest methods diverge significantly over 20 years. Simple interest
        grows linearly, compound interest grows exponentially, and continuous
        compounding provides the upper bound.
        """
    )
    return


@app.cell
def _(np, plt):
    P = 100_000  # initial principal
    r = 0.05     # nominal annual rate
    years = np.arange(0, 21)

    # Simple interest: F = P(1 + r*n)
    f_simple = P * (1 + r * years)

    # Compound interest (annual): F = P(1 + r)^n
    f_compound = P * (1 + r) ** years

    # Continuous compounding: F = P * e^(rn)
    f_continuous = P * np.exp(r * years)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(years, f_simple / 1000, "r--", linewidth=2, label="Simple Interest")
    ax1.plot(years, f_compound / 1000, "b-", linewidth=2, label="Compound Interest (annual)")
    ax1.plot(years, f_continuous / 1000, "g:", linewidth=2, label="Continuous Compounding")

    # Annotate gap at year 20
    gap = f_compound[-1] - f_simple[-1]
    ax1.annotate(f"Gap: ${gap/1000:.1f}k",
                 xy=(20, f_compound[-1] / 1000), xytext=(16, f_compound[-1] / 1000 + 8),
                 fontsize=10, color="blue",
                 arrowprops=dict(arrowstyle="->", color="blue"))

    ax1.plot(20, f_simple[-1] / 1000, "rs", markersize=8, zorder=5)
    ax1.plot(20, f_compound[-1] / 1000, "bo", markersize=8, zorder=5)
    ax1.plot(20, f_continuous[-1] / 1000, "g^", markersize=8, zorder=5)

    # Add final values
    ax1.text(20.3, f_simple[-1] / 1000, f"${f_simple[-1]/1000:.0f}k", fontsize=9, color="red", va="center")
    ax1.text(20.3, f_compound[-1] / 1000, f"${f_compound[-1]/1000:.0f}k", fontsize=9, color="blue", va="center")
    ax1.text(20.3, f_continuous[-1] / 1000 + 2, f"${f_continuous[-1]/1000:.0f}k", fontsize=9, color="green", va="center")

    ax1.set_xlabel("Years")
    ax1.set_ylabel("Future Value ($k)")
    ax1.set_title(f"Growth of $100,000 at 5% Annual Rate: Simple vs Compound vs Continuous")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 22)
    fig1.tight_layout()
    fig1
    return


# --- 19.8 Depreciation Methods Comparison ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 19.8 Depreciation Methods — Book Value Comparison

        A $120,000 asset depreciated four ways over 10 years: Straight-Line (SL),
        200% Declining Balance (DDB), MACRS 10-year, and Units-of-Production (UOP)
        with variable annual usage. SL is linear; DDB and MACRS are front-loaded,
        reducing taxable income faster in early years.
        """
    )
    return


@app.cell
def _(np, plt):
    cost = 120_000
    salvage = 10_000
    life = 10

    years_dep = np.arange(0, life + 1)

    # Straight-Line
    d_sl = (cost - salvage) / life
    bv_sl = cost - d_sl * years_dep
    bv_sl = np.maximum(bv_sl, salvage)

    # 200% Declining Balance (DDB)
    d_rate = 2.0 / life  # 0.20
    bv_ddb = np.zeros(life + 1)
    bv_ddb[0] = cost
    for yr in range(1, life + 1):
        dep = d_rate * bv_ddb[yr - 1]
        # Switch to SL when SL gives larger deduction
        remaining_life = life - yr + 1
        sl_dep = (bv_ddb[yr - 1] - salvage) / remaining_life if remaining_life > 0 else 0
        dep = max(dep, sl_dep)
        bv_ddb[yr] = max(bv_ddb[yr - 1] - dep, salvage)

    # MACRS 10-year (IRS percentages, half-year convention, 11 years of deductions)
    macrs_pct = [10.00, 18.00, 14.40, 11.52, 9.22, 7.37, 6.55, 6.55, 6.56, 6.55, 3.28]
    bv_macrs = np.zeros(life + 1)
    bv_macrs[0] = cost
    for yr in range(1, life + 1):
        if yr - 1 < len(macrs_pct):
            dep = cost * macrs_pct[yr - 1] / 100
        else:
            dep = 0
        bv_macrs[yr] = max(bv_macrs[yr - 1] - dep, 0)

    # Units-of-Production (variable usage)
    total_units = 100_000
    annual_units = [8000, 12000, 14000, 11000, 10000, 9000, 8000, 10000, 9000, 9000]
    d_unit = (cost - salvage) / total_units
    bv_uop = np.zeros(life + 1)
    bv_uop[0] = cost
    for yr in range(1, life + 1):
        dep = d_unit * annual_units[yr - 1]
        bv_uop[yr] = max(bv_uop[yr - 1] - dep, salvage)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(years_dep, bv_sl / 1000, "b-", linewidth=2, marker="o", markersize=5, label="Straight-Line")
    ax2.plot(years_dep, bv_ddb / 1000, "r--", linewidth=2, marker="s", markersize=5, label="200% DDB")
    ax2.plot(years_dep, bv_macrs / 1000, color="orange", linestyle="-.", linewidth=2, marker="^", markersize=5, label="MACRS 10-year")
    ax2.plot(years_dep, bv_uop / 1000, "g:", linewidth=2, marker="d", markersize=5, label="Units-of-Production")

    ax2.axhline(y=salvage / 1000, color="gray", linestyle=":", alpha=0.5, label=f"Salvage = ${salvage/1000:.0f}k")

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Book Value ($k)")
    ax2.set_title(f"Depreciation Methods: $120,000 Asset over {life}-Year Life")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, life)
    ax2.set_ylim(0, 130)
    fig2.tight_layout()
    fig2
    return


# --- 19.13.2 NPV Sensitivity Analysis ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 19.13.2 NPV Sensitivity — Solar Carport Project

        A $1,000,000 solar carport with expected $160,000/year savings over
        15 years at 8% MARR. The NPV is highly sensitive to annual savings —
        a ±20% variation swings the NPV from $96k to $643k. The breakeven
        savings is about $116,826/year.
        """
    )
    return


@app.cell
def _(np, plt):
    initial_cost = 1_000_000
    PA_factor = 8.5595  # (P/A, 8%, 15)

    savings = np.linspace(80_000, 240_000, 300)
    npv = -initial_cost + savings * PA_factor

    # Breakeven
    breakeven = initial_cost / PA_factor  # $116,826

    # Example points
    s_low, s_base, s_high = 128_000, 160_000, 192_000
    npv_low = -initial_cost + s_low * PA_factor
    npv_base = -initial_cost + s_base * PA_factor
    npv_high = -initial_cost + s_high * PA_factor

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(savings / 1000, npv / 1000, "b-", linewidth=2)

    # Shade viable/not viable regions
    ax3.fill_between(savings / 1000, npv / 1000, 0, where=(npv >= 0),
                     alpha=0.1, color="green", label="NPV > 0 (viable)")
    ax3.fill_between(savings / 1000, npv / 1000, 0, where=(npv < 0),
                     alpha=0.1, color="red", label="NPV < 0 (not viable)")
    ax3.axhline(y=0, color="black", linewidth=0.8)

    # Breakeven point
    ax3.plot(breakeven / 1000, 0, "ko", markersize=10, zorder=5)
    ax3.annotate(f"Breakeven: ${breakeven/1000:.1f}k/yr",
                 xy=(breakeven / 1000, 0), xytext=(breakeven / 1000 - 15, -120),
                 fontsize=10, color="black",
                 arrowprops=dict(arrowstyle="->", color="black"),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Mark ±20% range
    for s, n, label, color in [(s_low, npv_low, "−20%", "red"),
                                (s_base, npv_base, "Base", "blue"),
                                (s_high, npv_high, "+20%", "green")]:
        ax3.axvline(x=s / 1000, color=color, linestyle="--", alpha=0.5)
        ax3.plot(s / 1000, n / 1000, "o", color=color, markersize=8, zorder=5)
        ax3.annotate(f"{label}\nNPV = ${n/1000:.0f}k",
                     xy=(s / 1000, n / 1000), xytext=(s / 1000 + 3, n / 1000 + 30),
                     fontsize=9, color=color)

    ax3.set_xlabel("Annual Savings ($k)")
    ax3.set_ylabel("NPV ($k)")
    ax3.set_title("NPV Sensitivity: $1M Solar Carport Project (MARR = 8%, 15-year life)")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
