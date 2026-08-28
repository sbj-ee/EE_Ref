#!/usr/bin/env python3
"""Generate press-ready case-bound cover wraps for the EE-Book volumes.

A case (hardcover) wrap is one flat sheet covering back board, spine and front
board, plus the turn-in that folds around the board edges. Its size depends on
the text-block thickness, so it cannot be fixed until the page count and paper
stock are known.

    spine (text block) = pages / PPI          PPI = pages per inch for the stock
    spine (case)       = text-block spine + 2 x board thickness
    board height       = trim height + 2 x square
    board width        = trim width  + square          (spine edge sits at the joint)
    wrap width         = 2 x (turn-in + board width + joint) + case spine
    wrap height        = board height + 2 x turn-in

Every parameter below is a constant you can change; the defaults are ordinary
trade practice for a sewn case binding, but **printers differ**. Before sending
artwork, get the binder's template and reconcile SQUARE, JOINT, TURN_IN and
BOARD_THICKNESS against it.

Usage:
    python3 scripts/generate_cover_wrap.py            # writes both wraps
    python3 scripts/generate_cover_wrap.py --guides   # also writes guide versions
"""

import argparse
import math
from pathlib import Path

# ─── Binding parameters (inches) ─────────────────────────────────────────────
TRIM_W, TRIM_H = 6.0, 9.0
PPI = 400.0            # 50# offset
BOARD_THICKNESS = 0.098   # 98-point binder's board
SQUARE = 0.125            # board overhang beyond the trim, head/foot/fore-edge
JOINT = 0.25              # hinge gap between board edge and spine panel
TURN_IN = 0.75            # material folded around the board edges

PT = 72.0                 # SVG user units per inch
LEATHER_FREQ = 0.65       # per user unit; matches the grain of cover.svg

# NOTE on output resolution. Do not hand the SVG, or a PDF made from it with
# rsvg-convert, straight to a printer. The leather and the embossed gold are SVG
# *filters*, and rsvg rasterises filtered content on the PDF point grid -- 72
# ppi -- no matter what -d/-p or the viewBox scale say. That was measured, not
# assumed: pdfimages -list on the rsvg PDF reports x-ppi 72. Across a 16.6 in
# sheet that is visibly blocky.
#
# render_press() below therefore rasterises the whole wrap to a PNG at
# PRESS_DPI, which rsvg does honour, and wraps that PNG in a PDF at the exact
# physical size.

VOLUMES = [
    dict(key="reference", pages=865,
         title=["ELECTRICAL", "ENGINEERING", "REFERENCE"],
         spine_title="ELECTRICAL ENGINEERING REFERENCE",
         subtitle="Editio Unica",
         blurb="Nineteen disciplines of electrical engineering,\nwith worked examples throughout."),
    dict(key="problems", pages=961,
         title=["ELECTRICAL", "ENGINEERING", "REFERENCE"],
         spine_title="EE REFERENCE · PROBLEM SETS",
         subtitle="Problem Sets",
         blurb="1,298 problems, every one worked in full,\nkeyed section by section to the reference."),
]
AUTHOR = "STEPHEN B. JOHNSON"


def geometry(pages):
    """Return every dimension of the wrap, in inches."""
    block_spine = pages / PPI
    case_spine = block_spine + 2 * BOARD_THICKNESS
    board_w = TRIM_W + SQUARE
    board_h = TRIM_H + 2 * SQUARE
    wrap_w = 2 * (TURN_IN + board_w + JOINT) + case_spine
    wrap_h = board_h + 2 * TURN_IN
    return dict(block_spine=block_spine, case_spine=case_spine,
                board_w=board_w, board_h=board_h, wrap_w=wrap_w, wrap_h=wrap_h)


def defs(scale):
    """Gradients and filters, carried over from cover.svg so the wrap matches."""
    return f"""  <defs>
    <filter id="leather" x="0%" y="0%" width="100%" height="100%">
      <feTurbulence type="fractalNoise" baseFrequency="{LEATHER_FREQ:.4f}" numOctaves="4" seed="2" result="noise"/>
      <feDiffuseLighting in="noise" lighting-color="#2d5a27" surfaceScale="1.5" result="lit">
        <feDistantLight azimuth="225" elevation="35"/>
      </feDiffuseLighting>
      <feComposite in="lit" in2="SourceGraphic" operator="multiply"/>
    </filter>
    <linearGradient id="gold" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f5d98e"/><stop offset="30%" stop-color="#d4a520"/>
      <stop offset="50%" stop-color="#f5e6a3"/><stop offset="70%" stop-color="#d4a520"/>
      <stop offset="100%" stop-color="#b8860b"/>
    </linearGradient>
    <linearGradient id="goldSpine" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#f5d98e"/><stop offset="30%" stop-color="#d4a520"/>
      <stop offset="50%" stop-color="#f5e6a3"/><stop offset="70%" stop-color="#d4a520"/>
      <stop offset="100%" stop-color="#b8860b"/>
    </linearGradient>
    <linearGradient id="goldLine" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#d4a520"/><stop offset="50%" stop-color="#f5e6a3"/>
      <stop offset="100%" stop-color="#b8860b"/>
    </linearGradient>
    <filter id="textEmboss" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="{0.6*scale:.2f}" dy="{0.6*scale:.2f}" stdDeviation="{0.4*scale:.2f}"
                    flood-color="#0a2e0a" flood-opacity="0.85"/>
    </filter>
  </defs>
"""


def build(vol, guides=False):
    g = geometry(vol["pages"])
    W, H = g["wrap_w"] * PT, g["wrap_h"] * PT
    # panel x-origins, left to right
    x_back = (TURN_IN) * PT
    x_spine = (TURN_IN + g["board_w"] + JOINT) * PT
    x_front = (TURN_IN + g["board_w"] + JOINT + g["case_spine"] + JOINT) * PT
    w_board = g["board_w"] * PT
    w_spine = g["case_spine"] * PT
    y_board = TURN_IN * PT
    h_board = g["board_h"] * PT
    # the front board of cover.svg was 560 units wide; scale its type to ours
    s = w_board / 560.0

    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.2f} {H:.2f}" '
      f'width="{g["wrap_w"]:.4f}in" height="{g["wrap_h"]:.4f}in">')
    a(f'  <!-- {vol["key"]} volume case wrap. {vol["pages"]} pp on 50# offset '
      f'(PPI {PPI:.0f}). Text-block spine {g["block_spine"]:.3f} in, '
      f'case spine {g["case_spine"]:.3f} in. Flat size '
      f'{g["wrap_w"]:.3f} x {g["wrap_h"]:.3f} in including {TURN_IN} in turn-in. -->')
    a(defs(s))

    # leather across the whole sheet, turn-in included, so the fold edges are covered
    a(f'  <rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" fill="#1e5a1e"/>')
    a(f'  <rect x="0" y="0" width="{W:.2f}" height="{H:.2f}" filter="url(#leather)" opacity="0.6"/>')
    # spine panel slightly darker, as on the original
    a(f'  <rect x="{x_spine:.2f}" y="0" width="{w_spine:.2f}" height="{H:.2f}" fill="#1a4d1a"/>')
    a(f'  <rect x="{x_spine:.2f}" y="0" width="{w_spine:.2f}" height="{H:.2f}" '
      f'filter="url(#leather)" opacity="0.7"/>')
    # joint shadows
    for jx in (x_spine, x_spine + w_spine):
        a(f'  <line x1="{jx:.2f}" y1="0" x2="{jx:.2f}" y2="{H:.2f}" '
          f'stroke="#0a2e0a" stroke-width="{2*s:.2f}" opacity="0.6"/>')

    # ── spine: raised bands + gold rules + type ──
    yb1, yb2 = y_board + 0.16 * h_board, y_board + 0.84 * h_board
    for yy in (yb1, yb2):
        a(f'  <line x1="{x_spine:.2f}" y1="{yy:.2f}" x2="{x_spine+w_spine:.2f}" y2="{yy:.2f}" '
          f'stroke="#0f3a0f" stroke-width="{3*s:.2f}" opacity="0.5"/>')
        a(f'  <line x1="{x_spine+6*s:.2f}" y1="{yy+9*s:.2f}" x2="{x_spine+w_spine-6*s:.2f}" '
          f'y2="{yy+9*s:.2f}" stroke="url(#goldLine)" stroke-width="{0.8*s:.2f}" opacity="0.7"/>')
    cx = x_spine + w_spine / 2
    # spine type reads bottom-to-top, matching cover.svg
    ty = (yb1 + yb2) / 2
    a(f'  <text x="{cx:.2f}" y="{ty:.2f}" fill="url(#goldSpine)" '
      f'font-family="Georgia, \'Times New Roman\', serif" font-size="{16*s:.2f}" '
      f'font-weight="bold" letter-spacing="{3*s:.2f}" text-anchor="middle" '
      f'filter="url(#textEmboss)" transform="rotate(-90, {cx:.2f}, {ty:.2f})">'
      f'{vol["spine_title"]}</text>')
    ay = yb2 + (y_board + h_board - yb2) * 0.55
    a(f'  <text x="{cx:.2f}" y="{ay:.2f}" fill="url(#goldSpine)" '
      f'font-family="Georgia, \'Times New Roman\', serif" font-size="{11*s:.2f}" '
      f'letter-spacing="{1*s:.2f}" text-anchor="middle" filter="url(#textEmboss)" '
      f'transform="rotate(-90, {cx:.2f}, {ay:.2f})">{AUTHOR}</text>')

    # ── board panels: double-rule frame on each ──
    for bx in (x_back, x_front):
        for inset, sw, op in ((30 * s, 1.5 * s, 0.85), (40 * s, 0.7 * s, 0.6)):
            a(f'  <rect x="{bx+inset:.2f}" y="{y_board+inset:.2f}" '
              f'width="{w_board-2*inset:.2f}" height="{h_board-2*inset:.2f}" fill="none" '
              f'stroke="url(#goldLine)" stroke-width="{sw:.2f}" opacity="{op}"/>')

    # ── front board type ──
    fx = x_front + w_board / 2
    base = y_board
    for i, line in enumerate(vol["title"]):
        a(f'  <text x="{fx:.2f}" y="{base+(330+48*i)*s:.2f}" fill="url(#gold)" '
          f'font-family="Georgia, \'Times New Roman\', serif" font-size="{38*s:.2f}" '
          f'font-weight="bold" text-anchor="middle" letter-spacing="{3*s:.2f}" '
          f'filter="url(#textEmboss)">{line}</text>')
    a(f'  <text x="{fx:.2f}" y="{base+535*s:.2f}" fill="url(#gold)" '
      f'font-family="Georgia, \'Times New Roman\', serif" font-size="{22*s:.2f}" '
      f'font-style="italic" text-anchor="middle" letter-spacing="{5*s:.2f}" '
      f'filter="url(#textEmboss)">{vol["subtitle"]}</text>')
    a(f'  <line x1="{fx-50*s:.2f}" y1="{base+560*s:.2f}" x2="{fx+50*s:.2f}" y2="{base+560*s:.2f}" '
      f'stroke="url(#goldLine)" stroke-width="{0.8*s:.2f}" opacity="0.6"/>')
    a(f'  <text x="{fx:.2f}" y="{base+640*s:.2f}" fill="url(#gold)" '
      f'font-family="Georgia, \'Times New Roman\', serif" font-size="{20*s:.2f}" '
      f'text-anchor="middle" letter-spacing="{4*s:.2f}" filter="url(#textEmboss)">{AUTHOR}</text>')

    # ── back board: centred ornament and blurb ──
    bx = x_back + w_board / 2
    a(f'  <line x1="{bx-60*s:.2f}" y1="{y_board+300*s:.2f}" x2="{bx+60*s:.2f}" '
      f'y2="{y_board+300*s:.2f}" stroke="url(#goldLine)" stroke-width="{0.8*s:.2f}" opacity="0.6"/>')
    for k, line in enumerate(vol["blurb"].split("\n")):
        a(f'  <text x="{bx:.2f}" y="{y_board+(340+30*k)*s:.2f}" fill="url(#gold)" '
          f'font-family="Georgia, \'Times New Roman\', serif" font-size="{15*s:.2f}" '
          f'font-style="italic" text-anchor="middle" opacity="0.9" '
          f'filter="url(#textEmboss)">{line}</text>')
    a(f'  <line x1="{bx-60*s:.2f}" y1="{y_board+(340+30*len(vol["blurb"].split(chr(10))))*s:.2f}" '
      f'x2="{bx+60*s:.2f}" y2="{y_board+(340+30*len(vol["blurb"].split(chr(10))))*s:.2f}" '
      f'stroke="url(#goldLine)" stroke-width="{0.8*s:.2f}" opacity="0.6"/>')

    if guides:
        a('  <g id="guides" fill="none" stroke-dasharray="6 4">')
        # turn-in / board edge (fold line)
        a(f'    <rect x="{x_back:.2f}" y="{y_board:.2f}" width="{w_board:.2f}" '
          f'height="{h_board:.2f}" stroke="#ff0000" stroke-width="1"/>')
        a(f'    <rect x="{x_front:.2f}" y="{y_board:.2f}" width="{w_board:.2f}" '
          f'height="{h_board:.2f}" stroke="#ff0000" stroke-width="1"/>')
        # joints
        for jx in (x_spine, x_spine + w_spine):
            a(f'    <line x1="{jx:.2f}" y1="0" x2="{jx:.2f}" y2="{H:.2f}" '
              f'stroke="#00aaff" stroke-width="1"/>')
        # safe area, 0.25 in inside each board edge
        for bx2 in (x_back, x_front):
            a(f'    <rect x="{bx2+0.25*PT:.2f}" y="{y_board+0.25*PT:.2f}" '
              f'width="{w_board-0.5*PT:.2f}" height="{h_board-0.5*PT:.2f}" '
              f'stroke="#ffff00" stroke-width="1"/>')
        a('  </g>')
        a(f'  <text x="{6:.0f}" y="{H-6:.0f}" font-family="monospace" font-size="7" fill="#ffffff">'
          f'{vol["key"]}: {vol["pages"]}pp @ PPI {PPI:.0f} | block spine '
          f'{g["block_spine"]:.3f}in | case spine {g["case_spine"]:.3f}in | flat '
          f'{g["wrap_w"]:.3f}x{g["wrap_h"]:.3f}in | red=board/fold blue=joint yellow=safe</text>')

    a('</svg>')
    return "\n".join(o), g


PRESS_DPI = 400


def render_press(svg_path, g, dpi=PRESS_DPI):
    """Rasterise the wrap at `dpi` and wrap it in a PDF at the exact trim size.

    Goes through PNG on purpose: rsvg honours -w for PNG output but pins
    filtered content to 72 ppi in PDF output (see the note at the top).
    """
    import subprocess
    px = int(round(g["wrap_w"] * dpi))
    png = svg_path.with_suffix(".png")
    subprocess.run(["rsvg-convert", "-w", str(px), "-o", str(png), str(svg_path)], check=True)
    pdf = svg_path.with_suffix(".pdf")
    subprocess.run(["magick", str(png), "-units", "PixelsPerInch", "-density", str(dpi),
                    "-quality", "100", str(pdf)], check=True)
    return png, pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guides", action="store_true", help="also emit guide-marked versions")
    ap.add_argument("--no-press", action="store_true", help="skip the PNG/PDF press render")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    for vol in VOLUMES:
        svg, g = build(vol, guides=False)
        p = root / f"cover-wrap-{vol['key']}.svg"
        p.write_text(svg, encoding="utf-8")
        print(f"{p.name}: {g['wrap_w']:.3f} x {g['wrap_h']:.3f} in  "
              f"(block spine {g['block_spine']:.3f}, case spine {g['case_spine']:.3f})")
        if not args.no_press:
            png, pdf = render_press(p, g)
            print(f"  + {png.name}, {pdf.name} at {PRESS_DPI} dpi")
        if args.guides:
            svg, _ = build(vol, guides=True)
            q = root / f"cover-wrap-{vol['key']}-guides.svg"
            q.write_text(svg, encoding="utf-8")
            print(f"  + {q.name}")


if __name__ == "__main__":
    main()
