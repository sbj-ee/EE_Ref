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
        # Chapter 6: Digital Logic — Example Visualizations

        Interactive graphs for selected topics from Chapter 6,
        covering number system conversions, Boolean logic gate truth tables,
        and state machine timing diagrams.
        """
    )
    return


# --- 6.1 Number System Conversions ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 6.1 Number System Conversions

        Decimal values 0–15 mapped to their binary (4-bit), hexadecimal, and
        octal representations. Each row is one number; the color intensity
        corresponds to the decimal value, making it easy to see how the four
        bases align. Binary columns show individual bit positions (8-4-2-1),
        while hex and octal each use a single column.
        """
    )
    return


@app.cell
def _(np, plt):
    fig1, ax1 = plt.subplots(figsize=(10, 8))

    n_values = 16
    col_labels = ["Decimal", "Bit 3\n(8)", "Bit 2\n(4)", "Bit 1\n(2)", "Bit 0\n(1)", "Hex", "Octal"]
    n_cols = len(col_labels)

    # Build the color grid (normalized 0-1 for colormap)
    color_grid = np.zeros((n_values, n_cols))
    text_grid = []

    for _i in range(n_values):
        row_text = []
        # Decimal
        row_text.append(str(_i))
        color_grid[_i, 0] = _i / 15.0
        # Binary bits (3 down to 0)
        bits = [(_i >> b) & 1 for b in [3, 2, 1, 0]]
        for _j, b in enumerate(bits):
            row_text.append(str(b))
            color_grid[_i, 1 + _j] = b  # 0 or 1
        # Hex
        row_text.append(f"{_i:X}")
        color_grid[_i, 5] = _i / 15.0
        # Octal
        row_text.append(f"{_i:o}")
        color_grid[_i, 6] = _i / 15.0
        text_grid.append(row_text)

    # Use a blue-orange diverging colormap for visibility
    cmap = plt.cm.YlOrRd

    ax1.imshow(color_grid, cmap=cmap, aspect="auto", alpha=0.55)

    # Add text annotations
    for _i in range(n_values):
        for _j in range(n_cols):
            ax1.text(_j, _i, text_grid[_i][_j], ha="center", va="center",
                     fontsize=11, fontweight="bold", color="black")

    # Column headers
    ax1.set_xticks(range(n_cols))
    ax1.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax1.xaxis.set_ticks_position("top")
    ax1.xaxis.set_label_position("top")

    # Row labels
    ax1.set_yticks(range(n_values))
    ax1.set_yticklabels([f"  {_i}" for _i in range(n_values)], fontsize=9)

    # Grid lines between cells
    for _i in range(n_values + 1):
        ax1.axhline(_i - 0.5, color="gray", linewidth=0.5, alpha=0.5)
    for _j in range(n_cols + 1):
        ax1.axvline(_j - 0.5, color="gray", linewidth=0.5, alpha=0.5)

    # Thicker line separating binary columns from hex/octal
    ax1.axvline(0.5, color="black", linewidth=1.5)
    ax1.axvline(4.5, color="black", linewidth=1.5)

    ax1.set_title("Number System Conversion Table: Decimal, Binary, Hexadecimal, Octal (0–15)",
                   fontsize=12, pad=35)
    fig1.tight_layout()
    fig1
    return


# --- 6.2 Boolean Logic Gate Truth Tables ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 6.2 Boolean Logic Gate Truth Tables

        Visual truth tables for the six standard 2-input logic gates: AND, OR,
        NAND, NOR, XOR, and XNOR. Each subplot shows a 2x2 colored grid where
        rows represent input A (0 or 1) and columns represent input B (0 or 1).
        Green cells indicate output = 1 and red cells indicate output = 0, with
        the output value printed in each cell.
        """
    )
    return


@app.cell
def _(np, plt):
    # Define gate functions
    gates = {
        "AND":  lambda a, b: a & b,
        "OR":   lambda a, b: a | b,
        "NAND": lambda a, b: 1 - (a & b),
        "NOR":  lambda a, b: 1 - (a | b),
        "XOR":  lambda a, b: a ^ b,
        "XNOR": lambda a, b: 1 - (a ^ b),
    }

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7))
    axes_flat = axes2.flatten()

    inputs = [0, 1]

    for _idx, (name, func) in enumerate(gates.items()):
        _ax = axes_flat[_idx]

        # Build 2x2 output grid: rows = A (0,1), cols = B (0,1)
        grid = np.array([[func(a, b) for b in inputs] for a in inputs])

        # Color: green for 1, red for 0
        colors = np.zeros((*grid.shape, 3))
        for _i in range(2):
            for _j in range(2):
                if grid[_i, _j] == 1:
                    colors[_i, _j] = [0.2, 0.7, 0.3]  # green
                else:
                    colors[_i, _j] = [0.85, 0.25, 0.25]  # red

        _ax.imshow(colors, aspect="equal")

        # Add output text in each cell
        for _i in range(2):
            for _j in range(2):
                _ax.text(_j, _i, str(grid[_i, _j]), ha="center", va="center",
                        fontsize=22, fontweight="bold", color="white")

        # Axis labels
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["B=0", "B=1"], fontsize=10)
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(["A=0", "A=1"], fontsize=10)
        _ax.set_title(f"{name} Gate", fontsize=13, fontweight="bold", pad=8)

        # Grid lines
        _ax.axhline(0.5, color="white", linewidth=2)
        _ax.axvline(0.5, color="white", linewidth=2)

    fig2.suptitle("2-Input Logic Gate Truth Tables (Green = 1, Red = 0)",
                   fontsize=13, fontweight="bold", y=1.02)
    fig2.tight_layout()
    fig2
    return


# --- 6.5.5 State Machine Timing ---

@app.cell
def _(mo):
    mo.md(
        """
        ## 6.5.5 State Machine Timing — 3-Bit Binary Counter

        Timing diagram for a 3-bit synchronous binary counter cycling through
        states 000 → 001 → 010 → ... → 111 → 000 over 16 clock cycles. The
        clock signal and three flip-flop outputs (Q0, Q1, Q2) are shown as
        digital waveforms stacked vertically. Q0 toggles every clock edge,
        Q1 every 2 edges, and Q2 every 4 edges — the fundamental binary
        counting pattern.
        """
    )
    return


@app.cell
def _(np, plt):
    n_cycles = 16
    # Time points: two points per cycle (for step function appearance)
    t = np.arange(0, n_cycles + 1)

    # Clock signal: high for first half, low for second half of each cycle
    t_clk = []
    clk = []
    for _i in range(n_cycles):
        t_clk.extend([_i, _i + 0.5, _i + 0.5, _i + 1.0])
        clk.extend([1, 1, 0, 0])
    t_clk = np.array(t_clk)
    clk = np.array(clk)

    # Counter outputs: Q0 toggles every cycle, Q1 every 2, Q2 every 4
    # Values change on rising edge (start of each cycle)
    counter_vals = [i % 8 for i in range(n_cycles + 1)]
    q0_vals = [(v >> 0) & 1 for v in counter_vals]
    q1_vals = [(v >> 1) & 1 for v in counter_vals]
    q2_vals = [(v >> 2) & 1 for v in counter_vals]

    signals = [
        ("CLK", t_clk, clk, "C0"),
        ("Q0 (LSB)", t, q0_vals, "C1"),
        ("Q1", t, q1_vals, "C2"),
        ("Q2 (MSB)", t, q2_vals, "C3"),
    ]

    fig3, axes3 = plt.subplots(4, 1, figsize=(14, 6), sharex=True)

    for _idx, (label, t_sig, sig, color) in enumerate(signals):
        _ax = axes3[_idx]
        if label == "CLK":
            _ax.plot(t_sig, np.array(sig), color=color, linewidth=2)
            _ax.set_ylim(-0.2, 1.4)
        else:
            _ax.step(t_sig, sig, where="post", color=color, linewidth=2)
            _ax.set_ylim(-0.2, 1.4)

        _ax.set_ylabel(label, fontsize=11, fontweight="bold", rotation=0,
                       labelpad=60, va="center")
        _ax.set_yticks([0, 1])
        _ax.set_yticklabels(["0", "1"], fontsize=9)
        _ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        _ax.fill_between(t_sig if label == "CLK" else t,
                         sig, 0, alpha=0.15, color=color, step=None if label == "CLK" else "post")

    # Add counter value annotations below Q2
    ax_bottom = axes3[3]
    for _i in range(n_cycles):
        val = _i % 8
        ax_bottom.text(_i + 0.5, -0.1, f"{val:03b}", ha="center", va="top",
                        fontsize=7, color="gray", fontstyle="italic")

    axes3[3].set_xlabel("Clock Cycle", fontsize=11)
    axes3[3].set_xlim(0, n_cycles)
    axes3[3].set_xticks(range(n_cycles + 1))

    axes3[0].set_title("3-Bit Binary Counter Timing Diagram (16 Clock Cycles)",
                         fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig3
    return


if __name__ == "__main__":
    app.run()
