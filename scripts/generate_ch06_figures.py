"""Generate the Chapter 6 logic drawings in the Chapter 13 / Appendix G style.

Figures 6.3 and 6.3.1-6.3.6 (gate symbols) and the four diagrams added with
the Chapter 6 sanity fixes: 6.2.2 (K-map), 6.4.1 (ripple-carry adder),
6.5.5 (Moore "101" detector) and 6.6.3 (setup/hold timing).

The house style is copied from the generator of the public CC BY 4.0
sbj-ee/OpAmp-Circuits set (gen/lib.py, gen/trim.py), which drew the
Appendix G schematics and the Chapter 13 circuits: schemdraw on the
matplotlib backend, ink #1b2733, 1.3 pt lines, DejaVu Sans with mathtext
in dejavusans, 11 pt labels, 14 pt regular title above the drawing, 11 pt
notes below it, open-circle terminals, PNG at 300 dpi, RGB on white,
trimmed to the drawing plus a 0.15 in margin.

Usage:  python scripts/generate_ch06_figures.py
Needs:  schemdraw (0.23), matplotlib, numpy, pillow (see requirements.txt).
"""
import os
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageChops
import schemdraw
import schemdraw.elements as elm
import schemdraw.logic as logic

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "images")

INK = "#1b2733"          # OpAmp-Circuits lib.INK
LW = 1.3                 # line weight (pt)
FS, FS_TITLE, FS_NOTE = 11, 14, 11
FONT = "DejaVu Sans"
DPI, PAD = 300, 45       # 45 px at 300 dpi = 0.15 in margin (trim.py)
BG = (255, 255, 255)
# accents for the diagrams that need to tell groups or paths apart
BLUE, ORANGE, GREEN, RED = "#2266cc", "#cc6600", "#228844", "#b22222"

matplotlib.rcParams["font.family"] = FONT
schemdraw.config(lw=LW, fontsize=FS, color=INK, font=FONT, unit=2.5,
                 inches_per_unit=0.5, margin=0.3)
schemdraw.use("matplotlib")


def trim_png(path):
    """Trim to the drawing plus PAD, flatten to RGB on white, 300 dpi (trim.py)."""
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        flat = Image.new("RGBA", im.size, BG + (255,))
        flat.alpha_composite(im)
        im = flat.convert("RGB")
    else:
        im = im.convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = ImageChops.difference(im, Image.new("RGB", im.size, BG)).getbbox()
    im.crop((max(0, x0 - PAD), max(0, y0 - PAD), min(W, x1 + PAD), min(H, y1 + PAD))).save(
        path, dpi=(DPI, DPI))


def save_fig(fig, name):
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.05, facecolor="white",
                transparent=False)
    plt.close(fig)
    trim_png(path)
    print(f"  Saved {name}")


# ------------------------------------------------------------------
# Gate symbols (Figs 6.3, 6.3.1-6.3.6)
# ------------------------------------------------------------------
GATES = {
    "and":  ("AND",  logic.And,  r"$Y = A \cdot B$"),
    "or":   ("OR",   logic.Or,   r"$Y = A + B$"),
    "not":  ("NOT",  logic.Not,  r"$Y = A'$"),
    "nand": ("NAND", logic.Nand, r"$Y = (A \cdot B)'$"),
    "nor":  ("NOR",  logic.Nor,  r"$Y = (A + B)'$"),
    "xor":  ("XOR",  logic.Xor,  r"$Y = A \oplus B$"),
    "xnor": ("XNOR", logic.Xnor, r"$Y = (A \oplus B)'$"),
}
GSCALE = 2.0       # gate body scale; inputs 1.0 unit apart
LEAD = 1.2         # input/output lead length beyond the gate's own stubs


def label(d, p, s, ha="center", va="center", fs=FS):
    d.add(elm.Label().at(p).label(s, halign=ha, valign=va, fontsize=fs, color=INK))


def term(d, p, s, side):
    d.add(elm.Dot(open=True, radius=0.11).at(p))
    x, y = p
    if side == "left":
        label(d, (x - 0.25, y), s, ha="right")
    else:
        label(d, (x + 0.25, y), s, ha="left")


def draw_gate(d, key, origin):
    """Draw one gate with equal-weight leads and open terminals; return its extent."""
    name, cls, expr = GATES[key]
    ox, oy = origin
    g = d.add(cls().scale(GSCALE).at((ox, oy)).anchor("out"))
    ins = [g.in1] + ([g.in2] if key != "not" else [])
    names = ["$A$", "$B$"]
    xs = min(p.x for p in ins) - LEAD
    for p, n in zip(ins, names):
        d.add(elm.Line().at((xs, p.y)).to((p.x, p.y)))
        term(d, (xs, p.y), n, "left")
    xo = g.out.x + LEAD
    d.add(elm.Line().at((g.out.x, g.out.y)).to((xo, g.out.y)))
    term(d, (xo, g.out.y), "$Y$", "right")
    ymid = g.out.y
    return g, ymid, xs, xo


def finish_drawing(d, name, title=None, notes=(), title_gap=0.9):
    bb = d.get_bbox()
    cx = (bb.xmin + bb.xmax) / 2
    if title:
        label(d, (cx, bb.ymax + title_gap), title, va="bottom", fs=FS_TITLE)
    y = bb.ymin - 1.0
    for n in notes:
        label(d, (cx, y), n, fs=FS_NOTE)
        y -= 0.75
    d.draw(show=False)
    fig = d.fig.getfig()
    fig.set_facecolor("white")
    d.fig.ax.set_facecolor("white")
    save_fig(fig, name)


def single_gate(key):
    name, cls, expr = GATES[key]
    d = schemdraw.Drawing(show=False)
    draw_gate(d, key, (0, 0))
    finish_drawing(d, f"ch06_gate_{key}.png", f"{name} Gate", [expr])


def gate_overview():
    d = schemdraw.Drawing(show=False)
    # Each gate is placed by its output pin; the column pitch leaves about
    # 3 units between one gate's Y label and the next gate's A label.
    pitch, row = 9.0, 5.2
    layout = [("and", 0, 0), ("nand", 1, 0), ("not", 2, 0),
              ("or", 0.5, -1), ("nor", 1.5, -1),
              ("xor", 0.5, -2), ("xnor", 1.5, -2)]
    for key, col, r in layout:
        name, cls, expr = GATES[key]
        ox, oy = col * pitch, r * row
        g, ymid, xs, xo = draw_gate(d, key, (ox, oy))
        cx = (xs + xo) / 2
        label(d, (cx, ymid + 1.3), name, va="bottom", fs=12)
        label(d, (cx, ymid - 1.75), expr, fs=FS_NOTE)
    finish_drawing(d, "ch06_gate_symbols.png", "Standard Logic Gate Symbols",
                   ["IEEE/ANSI distinctive-shape symbols; two-input gates (NOT is single-input)"])


# ------------------------------------------------------------------
# Fig 6.2.2: K-map for Example 6.2.2
# ------------------------------------------------------------------
def kmap():
    ms = {0, 1, 2, 5, 8, 9, 10}
    gray = ["00", "01", "11", "10"]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(-1.6, 7.0); ax.set_ylim(-1.7, 5.8); ax.set_aspect("equal"); ax.axis("off")
    for r, ab in enumerate(gray):
        for c, cd in enumerate(gray):
            m = int(ab + cd, 2)
            x, y = c, 3 - r
            ax.add_patch(Rectangle((x, y), 1, 1, fill=False, ec=INK, lw=LW))
            ax.text(x + 0.5, y + 0.6, "1" if m in ms else "0", ha="center", va="center",
                    fontsize=16, color=INK if m in ms else "#8a96a3")
            ax.text(x + 0.5, y + 0.33, f"m{m}", ha="center", va="center", fontsize=7.5,
                    color="#6b7785")
        ax.text(-0.15, 3 - r + 0.5, ab, ha="right", va="center", fontsize=FS, color=INK)
    for c, cd in enumerate(gray):
        ax.text(c + 0.5, 4.12, cd, ha="center", va="bottom", fontsize=FS, color=INK)
    ax.text(-0.15, 4.12, r"$AB \backslash CD$", ha="right", va="bottom", fontsize=FS, color=INK)

    clip = Rectangle((-0.06, -0.06), 4.12, 4.12, transform=ax.transData)

    def loop(x0, y0, x1, y1, color, inset, clipped=False):
        p = FancyBboxPatch((x0 + inset, y0 + inset), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset,
                           boxstyle="round,pad=0,rounding_size=0.28", fill=False,
                           ec=color, lw=2.4)
        ax.add_patch(p)
        if clipped:
            p.set_clip_path(clip)
    # B'C' = {0,1,8,9}: rows AB=00 (top) and AB=10 (bottom) wrap, columns CD=00,01
    i = 0.07
    loop(0, 3, 2, 4.8, BLUE, i, True); loop(0, -0.8, 2, 1, BLUE, i, True)
    # B'D' = {0,2,8,10}: the four corners
    j = 0.17
    loop(-0.8, 3, 1, 4.8, ORANGE, j, True); loop(3, 3, 4.8, 4.8, ORANGE, j, True)
    loop(-0.8, -0.8, 1, 1, ORANGE, j, True); loop(3, -0.8, 4.8, 1, ORANGE, j, True)
    # A'C'D = {1,5}: column CD=01, rows AB=00,01
    loop(1, 2, 2, 4, GREEN, 0.12)

    lx, ly = 4.6, 3.6
    for k, (col, term_, cells) in enumerate([(BLUE, "$B'C'$", "m0, m1, m8, m9 (rows wrap)"),
                                            (ORANGE, "$B'D'$", "m0, m2, m8, m10 (four corners)"),
                                            (GREEN, "$A'C'D$", "m1, m5")]):
        yy = ly - 1.05 * k
        ax.add_patch(FancyBboxPatch((lx, yy - 0.2), 0.5, 0.4, boxstyle="round,pad=0,rounding_size=0.12",
                                    fill=False, ec=col, lw=2.4))
        ax.text(lx + 0.7, yy + 0.08, term_, ha="left", va="center", fontsize=12, color=col)
        ax.text(lx + 0.7, yy - 0.32, cells, ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(2.7, 5.3, "Karnaugh Map for Example 6.2.2", ha="center", va="bottom",
            fontsize=FS_TITLE, color=INK)
    ax.text(2.7, 4.95, r"$F(A, B, C, D) = \Sigma m(0, 1, 2, 5, 8, 9, 10)$", ha="center",
            va="center", fontsize=FS_NOTE, color=INK)
    ax.text(2.7, -1.25, r"$F = B'C' + B'D' + A'C'D$", ha="center", va="center",
            fontsize=13, color=INK)
    save_fig(fig, "ch06_kmap_ex622.png")


# ------------------------------------------------------------------
# Fig 6.4.1: 4-bit ripple-carry adder, Example 6.4.1
# ------------------------------------------------------------------
def ripple_carry():
    A, B = 0b1011, 0b0110
    c = [0]
    s = []
    for i in range(4):
        a, b = A >> i & 1, B >> i & 1
        s.append(a ^ b ^ c[i]); c.append((a & b) | (c[i] & (a ^ b)))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(-1.6, 19.6); ax.set_ylim(-2.4, 4.4); ax.set_aspect("equal"); ax.axis("off")
    W, H, pitch = 2.8, 1.8, 4.6
    arrow = dict(arrowstyle="-|>", mutation_scale=11, lw=LW, color=INK, shrinkA=0, shrinkB=0)
    for k, i in enumerate([3, 2, 1, 0]):            # MSB on the left
        x = 1.0 + k * pitch
        ax.add_patch(Rectangle((x, 0), W, H, fill=False, ec=INK, lw=LW))
        ax.text(x + W / 2, H / 2 + 0.2, "FA", ha="center", va="center", fontsize=12, color=INK)
        ax.text(x + W / 2, H / 2 - 0.35, f"bit {i}", ha="center", va="center", fontsize=9,
                color="#6b7785")
        for dx, nm, v in ((0.55, "A", A >> i & 1), (2.25, "B", B >> i & 1)):
            ax.add_patch(FancyArrowPatch((x + dx, H + 1.1), (x + dx, H), **arrow))
            ax.text(x + dx, H + 1.2, f"${nm}_{i}$ = {v}", ha="center", va="bottom",
                    fontsize=10, color=INK)
        ax.add_patch(FancyArrowPatch((x + W / 2, 0), (x + W / 2, -1.0), **arrow))
        ax.text(x + W / 2, -1.1, f"$S_{i}$ = {s[i]}", ha="center", va="top", fontsize=11,
                color=INK)
        # carry out of this stage goes left into the next stage (or out as C_out)
        cin_y = H / 2
        x_left = x - pitch + W if i < 3 else -0.4
        cout_col = RED if c[i + 1] else INK
        ax.add_patch(FancyArrowPatch((x, cin_y), (x_left, cin_y),
                                     **{**arrow, "color": cout_col, "lw": 2.0 if c[i + 1] else LW}))
        lbl = "$C_{out}$" if i == 3 else f"$C_{i + 1}$"
        ax.text((x + x_left) / 2, cin_y + 0.15, f"{lbl} = {c[i + 1]}", ha="center",
                va="bottom", fontsize=10, color=cout_col)
    xr = 1.0 + 3 * pitch + W
    ax.add_patch(FancyArrowPatch((xr + 1.3, H / 2), (xr, H / 2), **arrow))
    ax.text(xr + 1.4, H / 2, "$C_{in}$ = 0", ha="left", va="center", fontsize=10, color=INK)
    ax.text(9.0, 4.05, "4-Bit Ripple-Carry Adder for Example 6.4.1: 1011 + 0110",
            ha="center", va="bottom", fontsize=FS_TITLE, color=INK)
    ax.text(9.0, -2.05, r"$C_{out}S_3S_2S_1S_0$ = 1 0001 = 17;  carries in red ripple"
            " through all four stages (4 full-adder delays)", ha="center", va="center",
            fontsize=FS_NOTE, color=INK)
    save_fig(fig, "ch06_ripple_carry.png")


# ------------------------------------------------------------------
# Fig 6.5.5: Moore state diagram, overlapping "101" detector
# ------------------------------------------------------------------
def moore_101():
    import numpy as np
    from matplotlib.path import Path
    R = 0.62
    pos = {"S0": (0.0, 0.0), "S1": (3.6, 0.0), "S2": (7.2, 0.0), "S3": (7.2, -3.6)}
    out = {"S0": 0, "S1": 0, "S2": 0, "S3": 1}
    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    ax.set_xlim(-2.6, 10.0); ax.set_ylim(-5.9, 3.7); ax.set_aspect("equal"); ax.axis("off")
    for st, (x, y) in pos.items():
        hi = out[st] == 1
        ax.add_patch(Circle((x, y), R, fill=False, ec=INK, lw=LW))
        if hi:                                   # output state drawn with a double ring
            ax.add_patch(Circle((x, y), R - 0.09, fill=False, ec=INK, lw=LW))
        ax.text(x, y + 0.16, st, ha="center", va="center", fontsize=12, color=INK)
        ax.text(x, y - 0.24, f"Z = {out[st]}", ha="center", va="center", fontsize=9.5,
                color=RED if hi else INK)
    arr = dict(arrowstyle="-|>", mutation_scale=12, lw=LW, color=INK, shrinkA=0, shrinkB=0)
    txt = lambda x, y, t, **k: ax.text(x, y, t, ha=k.get("ha", "center"), va="center",
                                        fontsize=10, color=INK)

    def straight(a, b, dx=0.0, dy=0.0):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        v = np.array([x1 - x0, y1 - y0], float); v /= np.hypot(*v)
        off = np.array([dx, dy])
        # start/end on the circles, shifted sideways by (dx, dy)
        p0 = np.array([x0, y0]) + off + v * np.sqrt(R**2 - off @ off)
        p1 = np.array([x1, y1]) + off - v * np.sqrt(R**2 - off @ off)
        ax.add_patch(FancyArrowPatch(p0, p1, **arr))
        return (p0 + p1) / 2

    def self_loop(st, theta_deg):
        x, y = pos[st]; t = np.radians(theta_deg); da = np.radians(30)
        p0 = np.array([x + R * np.cos(t + da), y + R * np.sin(t + da)])
        p1 = np.array([x + R * np.cos(t - da), y + R * np.sin(t - da)])
        c0 = np.array([x, y]) + 2.1 * R * np.array([np.cos(t + 1.3 * da), np.sin(t + 1.3 * da)]) * 1.25
        c1 = np.array([x, y]) + 2.1 * R * np.array([np.cos(t - 1.3 * da), np.sin(t - 1.3 * da)]) * 1.25
        path = Path([p0, c0, c1, p1], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        ax.add_patch(FancyArrowPatch(path=path, **arr))
        return np.array([x, y]) + 2.35 * R * np.array([np.cos(t), np.sin(t)])

    m = straight("S0", "S1"); txt(m[0], m[1] + 0.25, "X = 1")
    m = straight("S1", "S2"); txt(m[0], m[1] + 0.25, "X = 0")
    m = straight("S2", "S3", dx=0.25); txt(m[0] + 0.2, m[1], "X = 1", ha="left")
    m = straight("S3", "S2", dx=-0.25); txt(m[0] - 0.2, m[1], "X = 0", ha="right")
    m = straight("S3", "S1"); txt(m[0] - 0.55, m[1] - 0.45, "X = 1")
    # S2 -> S0 ("100" is not a prefix of "101"): arc over S1
    (x2, y2), (x0, y0) = pos["S2"], pos["S0"]
    ax.add_patch(FancyArrowPatch((x2 - 0.3, y2 + 0.55), (x0 + 0.3, y0 + 0.55),
                                 connectionstyle="arc3,rad=0.42", **arr))
    txt(3.6, 2.2, "X = 0")
    m = self_loop("S0", 180); txt(m[0] - 0.55, m[1], "X = 0")
    m = self_loop("S1", 270); txt(m[0] - 1.05, m[1] + 0.15, "X = 1")
    ax.text(3.7, 3.1, 'Moore State Diagram for Example 6.5.5: Overlapping "101" Detector',
            ha="center", va="bottom", fontsize=FS_TITLE, color=INK)
    notes = ['S0 = idle,  S1 = received "1",  S2 = received "10",  S3 = received "101" (Z = 1)',
             r"$Q_1Q_0$: S0 = 00, S1 = 01, S2 = 10, S3 = 11;   "
             r"$Q_1^{+} = Q_0X' + Q_1Q_0'X$,  $Q_0^{+} = X$,  $Z = Q_1Q_0$"]
    for k, n in enumerate(notes):
        ax.text(3.7, -4.85 - 0.6 * k, n, ha="center", va="center", fontsize=FS_NOTE, color=INK)
    save_fig(fig, "ch06_moore_101.png")


# ------------------------------------------------------------------
# Fig 6.6.3: setup/hold and critical-path timing, Example 6.6.3
# ------------------------------------------------------------------
def setup_hold():
    tcq, tcrit, tsu, th, tmin = 1.2, 7.3, 0.8, 0.3, 1.5
    T = tcq + tcrit + tsu                       # 9.3 ns at f_max
    fig = plt.figure(figsize=(10.5, 7.4))
    # top: block diagram
    axb = fig.add_axes([0.05, 0.70, 0.90, 0.24]); axb.axis("off")
    axb.set_xlim(0, 21); axb.set_ylim(-1.4, 3.0); axb.set_aspect("equal")
    arr = dict(arrowstyle="-|>", mutation_scale=11, lw=LW, color=INK, shrinkA=0, shrinkB=0)
    for x0, nm in ((1.0, "FF1\n(launch)"), (16.0, "FF2\n(capture)")):
        axb.add_patch(Rectangle((x0, 0), 2.4, 2.2, fill=False, ec=INK, lw=LW))
        axb.text(x0 + 1.2, 1.25, nm, ha="center", va="center", fontsize=10, color=INK)
        axb.add_patch(Polygon([(x0, 0.15), (x0 + 0.35, 0.35), (x0, 0.55)], closed=False,
                              fill=False, ec=INK, lw=LW))
        axb.plot([x0 - 0.6, x0], [0.35, 0.35], color=INK, lw=LW)
        axb.text(x0 + 0.45, 1.95, "D", fontsize=8, va="top", color=INK)
        axb.text(x0 + 2.3, 1.95, "Q", fontsize=8, va="top", ha="right", color=INK)
    axb.plot([0.4, 0.4, 15.4, 15.4], [0.35, -0.9, -0.9, 0.35], color=INK, lw=LW)
    axb.text(0.3, -0.9, "CLK", ha="right", va="center", fontsize=10, color=INK)
    gx = [5.0, 8.6, 12.2]
    for k, (x, td) in enumerate(zip(gx, (2.5, 1.8, 3.0))):
        axb.add_patch(FancyBboxPatch((x, 0.45), 2.4, 1.3, boxstyle="round,pad=0,rounding_size=0.25",
                                     fill=False, ec=INK, lw=LW))
        axb.text(x + 1.2, 1.1, f"gate {k + 1}\n{td} ns", ha="center", va="center", fontsize=9.5,
                 color=INK)
    for xa, xb in ((3.4, 5.0), (7.4, 8.6), (11.0, 12.2), (14.6, 16.0)):
        axb.add_patch(FancyArrowPatch((xa, 1.1), (xb, 1.1), **arr))
    axb.text(10.5, 2.25, r"$t_{crit}$ = 2.5 + 1.8 + 3.0 = 7.3 ns (longest path);"
             r"  $t_{min}$ = 1.5 ns (shortest path)", ha="center", va="bottom", fontsize=10,
             color=INK)
    # bottom: waveforms
    ax = fig.add_axes([0.11, 0.08, 0.86, 0.56])
    tmax = 12.6
    ax.set_xlim(-0.8, tmax); ax.set_ylim(-0.6, 7.2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(INK); ax.tick_params(colors=INK, labelsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Time (ns)", fontsize=10, color=INK)
    ax.set_xticks([0, tcq, tcq + tmin, tcq + tcrit, T, T + tcq + tmin])
    ax.set_xticklabels(["0", "1.2", "2.7", "8.5", "9.3", "12.0"])
    # clock
    yc, hc = 5.8, 0.8
    xs, ys = [-0.8], [yc]
    for e in (0.0, T):
        xs += [e, e, e + T / 2, e + T / 2]; ys += [yc, yc + hc, yc + hc, yc]
    xs += [tmax]; ys += [yc]
    ax.plot(xs, ys, color=INK, lw=LW)
    ax.text(-1.0, yc + hc / 2, "CLK", ha="right", va="center", fontsize=10, color=INK)

    def bus(y, segs, name):
        h = 0.35
        for (a, b, kind, txt) in segs:
            if kind == "x":
                ax.add_patch(Rectangle((a, y - h), b - a, 2 * h, fc="#d9dee4", ec="none",
                                       hatch="////", lw=0))
                ax.plot([a, b], [y + h, y + h], color=INK, lw=LW)
                ax.plot([a, b], [y - h, y - h], color=INK, lw=LW)
            else:
                sl = 0.12
                ax.add_patch(Polygon([(a, y), (a + sl, y + h), (b - sl, y + h), (b, y),
                                      (b - sl, y - h), (a + sl, y - h)], closed=True,
                                     fill=False, ec=INK, lw=LW))
                ax.text((a + b) / 2, y, txt, ha="center", va="center", fontsize=9, color=INK)
        ax.text(-1.0, y, name, ha="right", va="center", fontsize=10, color=INK)

    bus(4.2, [(-0.8, tcq, "d", "old"), (tcq, T + tcq, "d", "new data"),
              (T + tcq, tmax, "d", "next")], "FF1 Q")
    bus(2.6, [(-0.8, tcq + tmin, "d", "old"), (tcq + tmin, tcq + tcrit, "x", ""),
              (tcq + tcrit, T + tcq + tmin, "d", "new (stable)"),
              (T + tcq + tmin, tmax, "x", "")], "FF2 D")
    # windows at the capture edge (T) and at the launch edge (0)
    for a, b, col in ((T - tsu, T, BLUE), (T, T + th, ORANGE), (0.0, th, ORANGE)):
        ax.axvspan(a, b, ymin=0.06, ymax=0.88, color=col, alpha=0.18, lw=0)
    for x in (th, T + th):
        ax.text(x + 0.08, 1.6, r"$t_h$ 0.3", ha="left", va="center", fontsize=8.5, color=ORANGE)
    for e in (0.0, T):
        ax.axvline(e, ymin=0.06, ymax=0.92, color=INK, lw=0.8, ls=(0, (4, 3)))

    def dim(a, b, y, txt, col=INK):
        ax.add_patch(FancyArrowPatch((a, y), (b, y), arrowstyle="<|-|>", mutation_scale=9,
                                     lw=1.0, color=col, shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, y + 0.22, txt, ha="center", va="bottom", fontsize=9, color=col)
    dim(0, tcq, 0.9, r"$t_{cq}$ 1.2")
    dim(tcq, tcq + tcrit, 0.9, r"$t_{crit}$ 7.3 (setup path)")
    dim(tcq + tcrit, T, 0.2, r"$t_{su}$ 0.8", BLUE)
    dim(0, tcq + tmin, 0.2, r"$t_{cq} + t_{min}$ = 2.7")
    ax.text(T + 0.05, 6.95, r"$T = t_{cq} + t_{crit} + t_{su}$ = 9.3 ns  ($f_{max}$ = 107.5 MHz)",
            ha="right", va="bottom", fontsize=10, color=INK)
    dim(th, tcq + tmin, 3.2, "hold margin 2.7 − 0.3 = 2.4 ns", ORANGE)
    fig.text(0.5, 0.975, "Setup, Hold and Critical-Path Timing for Example 6.6.3",
             ha="center", va="top", fontsize=FS_TITLE, color=INK)
    save_fig(fig, "ch06_setup_hold.png")


if __name__ == "__main__":
    print("Chapter 6: logic drawings (OpAmp-Circuits style)")
    for k in ("and", "or", "not", "nand", "nor", "xor"):
        single_gate(k)
    gate_overview()
    kmap()
    ripple_carry()
    moore_101()
    setup_hold()
