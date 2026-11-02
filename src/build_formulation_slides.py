"""Formulation slides (why, physics on a real photo, matrix form + E, BetaNet, depth, data) -> docs/formulation_slides.pptx
Run with the wec_rl venv after src/fig_slides_formulation.py. Notation follows Akkaynak & Treibitz: beta^D_c, beta^B_c."""
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "figures" / "4_presentation_slides"
EQ = F / "equations"
OUT = ROOT / "docs" / "formulation_slides.pptx"

INK, MUTED = RGBColor(0x1D, 0x2B, 0x36), RGBColor(0x5F, 0x6B, 0x73)
TEAL, CORAL = RGBColor(0x1C, 0x7C, 0x7D), RGBColor(0xC8, 0x55, 0x3D)
HEAD_FILL, RULE = RGBColor(0xEE, 0xF1, 0xF2), "C9D1D5"
HEAD, BODY = "Cambria", "Calibri"
SUP = {"D": "D", "B": "B"}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def beta(s, sub=""):
    """Runs for beta^s_sub, Akkaynak's notation."""
    runs = [("β", {}), (s, {"sup": True})]
    return runs + ([(sub, {"sub": True})] if sub else [])


def fill_runs(para, runs, size, color, bold, font):
    for t, o in ([(runs, {})] if isinstance(runs, str) else runs):
        r = para.add_run()
        r.text = t
        r.font.name = o.get("font", font)
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", False)
        r.font.color.rgb = o.get("color", color)
        if o.get("sup") or o.get("sub"):
            r.font._rPr.set("baseline", "30000" if o.get("sup") else "-25000")


def bullet(para, indent=0.25):
    pPr = para._p.get_or_add_pPr()
    pPr.set("marL", str(int(Inches(indent))))
    pPr.set("indent", str(-int(Inches(indent))))
    bu = etree.SubElement(pPr, qn("a:buChar"))
    bu.set("char", "•")


def text(slide, x, y, w, h, paras, size=15, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, bullets=False, space_after=6):
    """paras: str, or list of paragraphs, each a str or a list of (text, opts) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, 0)
    for i, p in enumerate([paras] if isinstance(paras, str) else paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        if bullets:
            bullet(para)
        fill_runs(para, p, size, color, bold, font)
    return tb


def pic(slide, path, x, y, w=None, h=None):
    W, H = Image.open(path).size
    w = w if w is not None else h * W / H
    h = h if h is not None else w * H / W
    slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return h


def title(slide, t):
    text(slide, 0.6, 0.4, 12.1, 0.7, t, size=30, bold=True, font=HEAD, anchor=MSO_ANCHOR.MIDDLE)


def rule(cell, side, color=RULE, width=9525):
    tcPr = cell._tc.get_or_add_tcPr()
    ln = etree.SubElement(tcPr, qn(f"a:{side}"), w=str(width))
    sf = etree.SubElement(ln, qn("a:solidFill"))
    etree.SubElement(sf, qn("a:srgbClr"), val=color)


def table(slide, x, y, col_w, rows, size=14, row_h=0.38, align_first=PP_ALIGN.LEFT):
    """Plain table: light header, thin horizontal rules, no banding."""
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(sum(col_w)),
                                 Inches(row_h * len(rows)))
    tbl = shp.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        tblPr.set(a, "0")
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            runs, color = (val if isinstance(val, tuple) else (val, None))
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEAD_FILL if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            for side in ("lnL", "lnR"):
                rule(cell, side, "FFFFFF", 0)
            rule(cell, "lnT", RULE if i <= 1 else "FFFFFF", 9525 if i <= 1 else 0)
            rule(cell, "lnB", RULE)
            p = cell.text_frame.paragraphs[0]
            p.alignment = align_first if j == 0 else PP_ALIGN.CENTER
            fill_runs(p, runs, size, color or INK, i == 0, BODY)
            if isinstance(runs, str) and runs in ("✓", "✗"):
                p.runs[0].font.name = "Segoe UI Symbol"
    return tbl


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ---------------------------------------------------------------- 1. why it matters
s = prs.slides.add_slide(BLANK)
title(s, "Why this matters")
h = pic(s, F / "raw_underwater_photos.png", 0.6, 1.3, w=6.5)
text(s, 0.6, 1.3 + h + 0.08, 6.5, 0.3, "Six real sites (UIEB, Sea-thru): every water gives its own colour cast",
     size=12, color=MUTED)
text(s, 0.6, 4.55, 3.1, 0.4, "What true-colour photos enable", size=16, bold=True, color=TEAL)
text(s, 0.6, 4.95, 3.1, 2.3, ["Reef health surveys: coral condition is scored by its colour",
                              "Detection that works in any water: fish counts, crown-of-thorns starfish surveys, "
                              "pipeline and hull inspection",
                              "Live correction on ROVs and AUVs"], size=13, bullets=True, space_after=4)
text(s, 4.0, 4.55, 3.1, 0.4, "If it stays unsolved", size=16, bold=True, color=CORAL)
text(s, 4.0, 4.95, 3.1, 2.3, ["Coral scores track the water, not the coral",
                              "Detectors need retraining at every new site",
                              "Correction stays offline, after the dive"], size=13, bullets=True, space_after=4)
stats = [("25 %", "of all marine species live on coral reefs, which cover under 1 % of the seafloor",
          "NOAA", TEAL),
         ("34 %", "of red light left after only 5 m, even in the clearest ocean",
          "Jerlov type I, β = 0.22 /m", INK),
         ("29°", "colour error on a grey chart in a raw Red Sea photo (0° = true grey)",
          "Sea-thru D1, our measurement", INK),
         ("0.4 s", "per photo for Sea-thru, the best physics fix: re-solved for every image, so offline",
          "our run, 240 × 320 px", INK)]
for k, (big, what, src, col) in enumerate(stats):
    y = 1.3 + k * 1.38
    text(s, 7.6, y, 1.85, 0.75, big, size=36, bold=True, color=col, font=HEAD)
    text(s, 9.5, y + 0.05, 3.25, 0.85, [what, [(src, {"size": 11, "color": MUTED})]], size=14, space_after=2)
notes(s,
      "Why it is useful: coral reefs hold about a quarter of all marine species on under 1 % of the seafloor (NOAA), and "
      "much of how we monitor them is photographic. Coral health is literally scored by colour: the CoralWatch Coral "
      "Health Chart (Siebeck et al., Coral Reefs 2006) matches a coral to a 6-step colour chart, so a photo with the "
      "water's tint removed gives a fairer score. The same goes for detectors: fish counting, crown-of-thorns starfish "
      "surveys on the Great Barrier Reef, pipeline and ship-hull inspection all work better when every water looks the "
      "same.\n"
      "Why the colour is wrong in the first place: even in the clearest ocean only about a third of the red light "
      "survives 5 m, so photos turn blue-green, and every water does it differently (left). On a real Red Sea photo a "
      "grey chart patch is 29 degrees away from grey: it looks cyan.\n"
      "If it stays unsolved: a coral's colour score changes with the water it was photographed in, not with its health; "
      "a detector trained at one site has to be retrained at the next; and the best physics fix, Sea-thru, re-solves an "
      "optimisation for every image (about 0.4 s at our size, 1.2 s at 600 x 400), so correction happens after the dive "
      "instead of on the robot.")

# ---------------------------------------------------------------- 2. physics, on a real photo
s = prs.slides.add_slide(BLANK)
title(s, "How water changes a photo, and how to undo it")
h = pic(s, F / "real_restoration_steps.png", 0.6, 1.3, w=7.3)
text(s, 0.6, 1.3 + h + 0.12, 7.3, 0.9,
     [[("Real photo, undone with BetaNet's 12 numbers for it. ", {"bold": True}),
       ("At 1–3 m little is lost along the line of sight (step 5): most red was lost on the sunlight's way down (step 6)",
        {})]], size=13, color=MUTED)
text(s, 8.3, 1.3, 4.45, 0.35, "Older models: one β per colour for both terms", size=14, color=MUTED)
pic(s, EQ / "eq_old_model.png", 8.3, 1.72, w=3.6)
text(s, 8.3, 2.5, 4.45, 0.35, "Akkaynak & Treibitz (2018): two different βs", size=14, bold=True, color=TEAL)
pic(s, EQ / "eq_revised.png", 8.3, 2.92, w=4.45)
table(s, 8.3, 3.95, [1.65, 1.4, 1.4], [
    ["", beta("D", "c"), beta("B", "c")],
    ["Name", "attenuation", "backscatter"],
    ["Acts by", "× multiply", "+ add"],
    ["You see", "colour loss", "contrast loss"],
    ["Object-dependent", ("yes", CORAL), ("no", TEAL)],
], size=14)
text(s, 8.3, 6.0, 4.45, 0.6, "c = R, G, B.  z = range, camera to object (not depth below the surface)",
     size=12, color=MUTED)
notes(s,
      "Each pixel is the sum of two parts. The direct signal is the object's own light, fading exponentially with range z "
      "at rate beta-D. Backscatter is sunlight scattered into the camera by particles: a haze that builds with range at "
      "rate beta-B towards the water colour B-infinity. Notation is Akkaynak's: superscript D direct, superscript B "
      "backscatter, subscript c the colour channel. Older models used one beta for both; Akkaynak & Treibitz (CVPR 2018) "
      "showed they differ: beta-D depends on the object's colour and on range, beta-B does not.\n"
      "The figure runs the model backwards on a real Red Sea photo, using the 12 numbers BetaNet predicted for it: "
      "(3) the predicted haze, (4) subtract it and contrast comes back, (5) multiply by e^(beta-D z) to undo the fading "
      "along the line of sight, (6) divide by the sunlight colour E. Step 5 is small here because the scene is only "
      "1-3 m away; the big colour change is step 6, because most of the red was lost on the sunlight's way down from "
      "the surface. That is exactly why we added E (next slide).\n"
      "Careful with words: z is range, camera to object. Depth below the surface is a different quantity; it sets E.")

# ---------------------------------------------------------------- 3. our model
s = prs.slides.add_slide(BLANK)
title(s, "Our model: matrix form + sunlight colour E")
pic(s, EQ / "eq_matrix.png", 0.6, 1.3, w=12.1)
text(s, 0.6, 3.45, 5.9, 0.35, "Undo the water (our physics layer)", size=16, bold=True, color=TEAL)
pic(s, EQ / "eq_inverse.png", 0.6, 3.9, w=5.6)
text(s, 0.6, 4.75, 5.9, 0.35, "The network predicts only these 12 numbers", size=16, bold=True, color=TEAL)
pic(s, EQ / "eq_theta.png", 0.6, 5.2, w=3.5)
text(s, 0.6, 5.9, 5.9, 0.9, ["Ill-posed: 3HW pixels, 3HW + 12 unknowns → learn θ from many waters",
                             "Assumes one water per image, sunlight only (no strobes), no blur"],
     size=13, color=MUTED, bullets=True, space_after=3)
h = pic(s, F / "sunlight_colour_fix.png", 6.9, 3.45, w=5.85)
text(s, 6.9, 3.45 + h + 0.1, 5.85, 0.7,
     [[("Without E, ", {"bold": True, "color": CORAL}),
       ("missing red is blamed on distance, so the red gain maxes out → magenta", {})],
      [("Real Red Sea frames (Sea-thru D1, D4), network's own output", {"size": 12, "color": MUTED})]], size=14,
     space_after=2)
notes(s,
      "Same physics for all three colours at once, for one pixel. The pixel's colour vector [I_R, I_G, I_B] equals the "
      "fading matrix times the sunlight colour times the clean colour, plus the haze build-up times the water colour. "
      "Every matrix is diagonal: red only affects red. z changes per pixel; the 12 coefficients are shared by the image.\n"
      "What we added: the sunlight colour E (coral). Without it, 'no red' can only be explained by distance, so beta-D "
      "for red goes to its limit and undoing it multiplies red by up to 50x: magenta (middle column). With E (v2) the "
      "result is neutral; on UIEB this took PSNR 15.9 -> 18.0 dB. E is normalised so its brightest channel is 1, "
      "because the brightness of the light and of the object cannot be told apart.\n"
      "Why the matrix form: a diagonal matrix inverts entry by entry, so the physics layer is closed form: subtract the "
      "haze, multiply by e^(+beta-D z) (capped at 50x so noise in far, dark pixels is not blown up), divide by E.\n"
      "Why learn it: for an H x W image we observe 3HW numbers but the unknowns are the 3HW clean pixels plus 12 "
      "coefficients, so one image alone cannot pin them down. Sea-thru adds a hand-made assumption (the darkest pixels "
      "are pure haze); we learn the prior from many simulated waters.\n"
      "Assumptions: one water per image (constant beta; Sea-thru lets beta-D vary with range), natural sunlight only "
      "(no strobes, which deep AUVs use), no forward-scatter blur, linear camera (we undo sRGB gamma).")

# ---------------------------------------------------------------- 4. BetaNet
s = prs.slides.add_slide(BLANK)
title(s, "BetaNet: photo + range in, 12 numbers out")
h = pic(s, F / "betanet_architecture.png", 0.6, 1.25, w=12.1)
pic(s, EQ / "eq_problem.png", 0.6, 1.25 + h + 0.35, w=5.4)
text(s, 0.6, 1.25 + h + 1.05, 5.6, 0.8,
     ["Coefficients only: the physics layer makes the image, so the network cannot invent content"],
     size=13, color=MUTED)
for k, (big, small) in enumerate([("11.3 M", "parameters"), ("3.2 ms", "per image, laptop GPU"),
                                  ("~10 min", "training, RTX 3050")]):
    x = 6.9 + k * 2.0
    text(s, x, 1.25 + h + 0.3, 1.9, 0.6, big, size=28, bold=True, color=INK, font=HEAD)
    text(s, x, 1.25 + h + 0.95, 1.9, 0.6, small, size=13, color=MUTED)
notes(s,
      "Input: a 4-channel 240 x 320 tensor. Three channels are the photo in linear RGB, square-rooted (a gentle "
      "brightening, closer to what ImageNet networks expect) and ImageNet-normalised; the fourth is the range map in "
      "metres divided by 10.\n"
      "Network: a standard ResNet-18 pre-trained on ImageNet, so it already knows edges and textures. We changed the "
      "first convolution to take 4 channels; the depth filter starts as the mean of the RGB filters. Four residual "
      "stages shrink the image to an 8 x 10 grid of 512 features, global average pooling gives one 512-vector for the "
      "whole image, and a small head (512 -> 256 -> 12) outputs the 12 numbers. 11.3 M parameters, of which the head is "
      "0.13 M.\n"
      "Outputs are kept physical by their activations: beta-D and beta-B through exp (always positive, about 0.0002 to "
      "3 per metre), B-infinity through a sigmoid (0 to 1), E through a sigmoid divided by its largest channel (only its "
      "colour matters).\n"
      "Formally: theta-hat = f_w(I, z). The network outputs coefficients only; the closed-form physics layer from the "
      "previous slide produces the image, so it can only undo water, not invent content. 3.2 ms per image on a laptop "
      "RTX 3050 versus about 0.4 s for Sea-thru: roughly 120x faster.")

# ---------------------------------------------------------------- 5. depth
s = prs.slides.add_slide(BLANK)
title(s, "Why not Depth Anything V2 as is")
h = pic(s, F / "depth_maps_dav2.png", 0.6, 1.3, w=6.0)
text(s, 0.6, 1.3 + h + 0.25, 6.0, 0.35, "Why we need range at all", size=16, bold=True, color=TEAL)
text(s, 0.6, 1.3 + h + 0.7, 6.0, 1.0, ["z multiplies every β: without it, close + murky looks like far + clear",
                                       "β is per metre, so no metric range means no units"],
     size=14, bullets=True, space_after=3)
text(s, 7.05, 1.3, 5.7, 0.35, "Scale error on 24 real frames", size=16, bold=True, color=TEAL)
h2 = pic(s, F / "depth_scale_per_frame.png", 7.05, 1.75, w=5.7)
table(s, 7.05, 1.75 + h2 + 0.25, [4.2, 1.5], [
    ["Depth source", "AbsRel ↓"],
    ["DAv2 outdoor model, as is", ("2.59", CORAL)],
    ["DAv2 indoor model, as is", ("0.76", CORAL)],
    ["DAv2 indoor + 1 range reading", ("0.12", TEAL)],
    ["Best case: scale fitted to truth", ("0.07", TEAL)],
], size=14, row_h=0.36)
text(s, 7.05, 1.75 + h2 + 2.15, 5.7, 0.35, "AbsRel = mean |predicted − true| / true:  0.76 = 76 % off on average",
     size=12, color=MUTED)
notes(s,
      "Why range: z multiplies every coefficient. Without it, a faint red object close up in murky water and a bright "
      "red object far away in clear water give the same pixel, and beta, which is per metre, has no units.\n"
      "Where range can come from: Sea-thru has structure-from-motion; a robot could use stereo or sonar. Without a sensor, "
      "Depth Anything V2 (NeurIPS 2024) estimates depth from one photo. Its metric models are fine-tuned on indoor "
      "scenes (Hypersim) and outdoor driving (Virtual KITTI). AbsRel = average of |predicted - true| / true over all "
      "pixels, against Sea-thru's SfM range on 24 real frames.\n"
      "The maps: the shape is right (diver, rocks, seafloor) but the numbers are not; the outdoor model puts everything "
      "far away. The dot plot: one dot per frame, x = predicted / true. The indoor model is right on average but "
      "inconsistent, from half to 2.5x: AbsRel 0.76. The outdoor model is consistently about 3.5x too far (haze looks "
      "like distance on land): 2.59.\n"
      "Fix: one known distance (an altimeter or sonar reading; we simulate it by matching the median of the true range) "
      "rescales the map: 0.12, close to the best possible 0.07. So DAv2 alone is not usable for metric beta; DAv2 + one "
      "reading is a workable fallback.")

# ---------------------------------------------------------------- 6. data
s = prs.slides.add_slide(BLANK)
title(s, "Training data: real rooms, simulated water")
Y, N = ("✓", TEAL), ("✗", CORAL)
table(s, 0.6, 1.3, [2.25, 1.05, 1.0, 1.0, 1.0], [
    ["Dataset", "Water", "Depth", "Known β", "Use"],
    ["NYU-v2 + our water", ("simulated", MUTED), Y, Y, ("Train", TEAL)],
    ["Sea-thru D1–D5", "real", Y, N, "Test"],
    ["UIEB (890 photos)", "real", N, N, "Test"],
    ["SQUID (57 pairs)", "real", Y, N, "Later"],
], size=14)
text(s, 0.6, 3.3, 6.3, 0.3, "2,881 train · 288 val · 654 test rooms · 240 × 320 px", size=13, color=MUTED)
text(s, 0.6, 3.85, 6.3, 0.35, "Synthetic water: new random water every batch", size=16, bold=True, color=TEAL)
table(s, 0.6, 4.3, [1.95, 1.55], [
    ["Water physics", "Range"],
    ["Jerlov type", "10, interpolated"],
    ["Turbidity", "×0.6–1.6"],
    ["Per-colour jitter", "±15 %"],
    [beta("B") + [(" / ", {})] + beta("D"), "×0.5–2"],
    ["Sun depth (→ E)", "0.5–12 m"],
    [[("B", {}), ("∞", {"sup": True}), (" brightness", {})], "0.15–0.65"],
], size=13, row_h=0.34)
table(s, 4.3, 4.3, [1.6, 1.0], [
    ["Noise", "Range"],
    ["Depth stretch", "×0.7–2.0"],
    ["Exposure", "×0.7–1.3"],
    ["Sensor noise σ", "0.2–1 %"],
    ["Left-right flip", "50 %"],
], size=13, row_h=0.34)
h = pic(s, F / "jerlov_six_waters.png", 7.3, 1.3, w=5.45)
text(s, 7.3, 1.3 + h + 0.1, 5.45, 0.35,
     [[("Jerlov water types:  ", {"bold": True})] + beta("D") + [(" = −ln N", {})]], size=14)
text(s, 7.3, 5.45, 5.45, 1.3,
     [[("Why rooms? ", {"bold": True, "color": TEAL}), ("Only data with metric depth and a known answer", {})],
      [("Real dives have no labels → test only", {"color": MUTED})],
      [("All physics in linear RGB (sRGB gamma removed first)", {"color": MUTED, "size": 13})]], size=15)
notes(s,
      "Supervised training needs real-looking water, metric depth, and the true answer. No real dataset has all three: "
      "Sea-thru and SQUID have depth but nobody knows the true coefficients; UIEB has no depth and its references are "
      "the result volunteers preferred out of 12 enhancement methods, not true colours. So we take NYU-v2 indoor rooms "
      "(Kinect metric depth) and add water ourselves, so we know the coefficients exactly. Real dives are for testing.\n"
      "Jerlov types: the 10 standard optical water classes from oceanography (N. Jerlov, Marine Optics, 1976). I, IA, IB, "
      "II, III are open ocean, clear and blue; 1, 3, 5, 7, 9 coastal, greener and murkier. Each is published as N, the "
      "fraction of light left after 1 m; we use the RGB table from UWCNN (Li et al., Pattern Recognition 2020), beta-D = -ln N.\n"
      "Recipe per batch: ocean or coastal, a random point between two neighbouring types, turbidity x0.6-1.6, per-colour "
      "jitter 15 %, beta-B = beta-D x 0.5-2 (different, as Akkaynak says), sunlight colour E from 0.5-12 m of descent "
      "through the same water, water colour B-inf from E and the scattering albedo.\n"
      "Noise and augmentation: depth stretched x0.7-2 so scenes aren't all room-sized, exposure x0.7-1.3, Gaussian sensor "
      "noise sigma 0.002-0.01 (0.2-1 % of full scale), 50 % horizontal flip. All physics in linear RGB, because the "
      "add-and-multiply model only holds for linear light.")

OUT.parent.mkdir(exist_ok=True)
prs.save(OUT)
print(OUT)
