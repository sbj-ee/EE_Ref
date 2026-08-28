# Print Specification

Press-ready specification for the bound edition. The screen editions
(`EE-Book.pdf`, `EE-Book-Problems.pdf`) are unaffected by anything here — they
remain US Letter with live coloured hyperlinks. The press files are built
alongside them by `./build-pdf.sh` as `EE-Book-print.pdf` and
`EE-Book-Problems-print.pdf`.

## Trim and imposition

| Item | Value |
|---|---|
| Trim size | 6 × 9 in (432 × 648 pt) |
| Sheet size in the PDF | 7 × 10 in (504 × 720 pt) |
| Slug | 0.5 in on all four sides |
| Crop marks | 0.375 in long, 0.3 pt, offset 0.125 in from each trim corner |
| Imposition | trim page centred on the sheet |
| Sides | two-sided, chapters open recto |
| Body size | 10 pt on the 4.25 in measure |

**Margins** (from the trim edge, mirrored on verso pages):

| | |
|---|---|
| Inner (gutter) | 1.00 in |
| Outer | 0.75 in |
| Top | 0.75 in |
| Bottom | 0.875 in |
| Text block | 4.25 × 7.375 in |

The gutter is deliberately wider than the outer margin. At the page counts below
the book is thick enough that a sewn block will not lie flat without it.

## Bleed

The interior has **no bleed and needs none**: no figure, rule, table or panel in
either volume runs to the trim edge, so there is nothing to bleed off. Every
figure is placed within the text measure.

Bleed applies only to the **cover wrap**, where the turn-in serves as the
bleed — see below.

## Figures

All 98 figures are regenerated at 300 dpi (`scripts/generate_images.py` and
`scripts/generate_appendix_g_figures.py`). Placed at the 4.25 in text measure,
the *lowest* effective resolution is now **445 dpi** and the median is 699 dpi,
comfortably clear of the 300 dpi floor commercial printing expects. Previously
the generators wrote 150 dpi and 93 of the 98 figures fell below 300 dpi at the
placed size, the worst at 147 dpi.

Figures remain **RGB**. If the printer requires CMYK separations, that is a
conversion step at prepress — flag it when the printer is chosen, since black
text rendered as four-colour black is a common and avoidable defect.

## Fonts

STIX Two Text, STIX Two Math and DejaVu Sans Mono, all embedded as subsets.
No substitutions and no Type 3 bitmaps.

## Spine width

Spine width depends on the paper, so it cannot be fixed until the stock is
chosen. The rule is:

```
spine (in) = interior page count / PPI      (PPI = pages per inch for the stock)
```

At the current build — **reference volume 869 pp, problem companion 961 pp** —
the text-block spine works out as:

| Stock | PPI | Reference volume (869 pp) | Problem companion (961 pp) |
|---|---|---|---|
| 50# offset (typical short-run) | 400 | **2.17 in** | **2.40 in** |
| 60# offset | 340 | 2.56 in | 2.83 in |
| 70# opaque (premium, less show-through) | 300 | 2.90 in | 3.20 in |

Re-read these after any content change that moves the page count.

Add the board and covering material for a case binding: typically 0.125 in per
board plus the leather and turn-in, so the finished spine is roughly
`text-block spine + 0.25 in`. The printer will give an exact figure for their
boards.

## Cover wrap

Produced by `scripts/generate_cover_wrap.py`, which carries over the visual
language of `cover.svg` (green leather, gold rules and type, raised spine
bands) and extends it across a full case wrap. `cover.svg`, the previous artwork, has been retired: it was a
perspective *mockup* of a front board (620 × 800 px, shared by both volumes and
not the 2:3 shape of the actual book). It is replaced by `cover-front-*.svg`,
flat 6 × 9 front covers generated from the same code as the wrap, which also
feed the EPUB covers and the README image.

Stock is **50# offset at 400 PPI**, which sets the text-block thickness:

```
text block  = pages / PPI
case spine  = text block + (2 × board thickness)
board       = trim + square          (square = 0.125 in head, foot, fore-edge)
wrap width  = (2 × turn-in) + (2 × board width) + (2 × joint) + case spine
wrap height = board height + (2 × turn-in)
```

with 98-point binder's board (0.098 in), a 0.25 in joint at each hinge and a
0.75 in turn-in on all four sides.

| | Reference | Problem sets |
|---|---|---|
| Interior pages | 869 | 961 |
| Text block | 2.173 in | 2.403 in |
| **Case spine** | **2.369 in** | **2.599 in** |
| **Flat wrap** | **16.619 × 10.750 in** | **16.849 × 10.750 in** |

The turn-in *is* the bleed — art runs the full flat size, and the outer 0.75 in
on every side wraps to the inside of the board, so nothing critical may sit
there. Output is `cover-wrap-{reference,problems}.pdf`, a single 400 dpi
image placed at exact physical size (verify with `pdfimages -list`). Matching
`-guides.svg` files overlay the board/fold line in red, the joints in blue and
the safe area in yellow; those are for checking only and must not be sent to
the binder.

**Spine type runs top-to-bottom**, the US and UK trade convention: upright on
the shelf the title reads downward, and with the book lying front-cover-up it
reads normally. The retired `cover.svg` had used `rotate(-90)`, the European
bottom-to-top direction; the wrap uses `rotate(90)`.

Both spine widths depend on the page counts above, so any reflow of the
interior invalidates the wrap. Regenerate it after every press build.

## Volume splitting

At 6 × 9 both volumes came out **over the practical limit**: the reference
volume is 869 pp and the problem companion 961 pp, against a rough ceiling of
about 800 interior pages for a sewn case binding. Beyond that the block is hard
to open flat and the spine takes more stress than the sewing will forgive. This
is a decision still to make; the options, in order of preference:

1. **Thinner stock** — 50# rather than 70# buys back roughly a third of the
   spine thickness for the same page count.
2. **Split the problem companion by chapter range** — it is already organised
   into 26 independent chapters and splits cleanly; the reference volume does
   not, because the dictionary and references must travel with the chapters.
3. **Move to 7 × 10** — a larger text block cuts the page count materially.

## What is verified

- **Page geometry** — `\textwidth` is 307.15 TeX-pt = 4.25 in exactly; recto and
  verso margins mirror correctly (recto text starts 1.0 in from the trim edge,
  verso 0.75 in).
- **Crop marks** — verified by rasterising sampled pages of each press file at
  72 dpi and counting ink in the slug at all four trim corners. 40 random pages
  of the reference volume and 30 of the problem companion: no page missing a
  mark. This check is worth repeating after any change to the preamble, because
  the marks are fragile in a specific way — see the note in `impose-print.tex`.
- **Figures** — minimum 445 dpi at the placed size, median 699 dpi.
- **Page sizes** — press files are 504 × 720 pt (7 × 10 in) with the 6 × 9 trim
  centred; screen files remain 612 × 792 pt (US Letter).

## Side effect of the 300 dpi figures

Raising the figure resolution roughly doubled the screen PDF: `EE-Book.pdf` went
from about 12 MB to 23 MB, and the EPUB likewise. That is the cost of one image
set serving both editions. If the download size matters more than screen zoom
quality, the alternative is to generate two sets — 150 dpi for screen, 300 dpi
for press — at the cost of a more complicated build and a second image
directory.

## What is not done

- CMYK conversion (needs the printer's profile).
- A physical proof. Nothing here substitutes for one.
