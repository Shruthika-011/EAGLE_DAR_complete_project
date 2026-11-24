"""Disaster Damage Assessment Report (PDF) renderer. ReportLab only; no data is computed here.

Every figure comes from `ReportData`, which the caller fills from the stored predictions, the deterministic
reasoning engine and the evaluation files. Text is tagged by provenance so a reader can tell
  COMPUTED          deterministic result of the spatial reasoning engine,
  MODEL PREDICTION  output of the YOLO + Siamese models,
  AI EXPLANATION    wording only (template or verified LLM), never a source of numbers.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from xml.sax.saxutils import escape

import cv2
import numpy as np
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Circle, Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, HRFlowable, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from backend.services.report_content import CLASS_DEFINITIONS, LABEL, QUADS, affected_counts, pct
from reasoning.damage_map import CLASS_COLORS_BGR
from reasoning.schema import CLASS_NAMES, Scene

NAVY = colors.HexColor("#0a1d3a")
BLUE = colors.HexColor("#2563eb")
LIGHT = colors.HexColor("#eef2f8")
LINE = colors.HexColor("#cbd5e1")
MUTED = colors.HexColor("#64748b")
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

CLASS_HEX = {n: "#%02x%02x%02x" % (r, g, b) for n, (b, g, r) in CLASS_COLORS_BGR.items()}
CLASS_TITLE = {"no-damage": "No Damage", "major-damage": "Major Damage", "destroyed": "Destroyed"}

_SUBS = {"×": "x", "–": "-", "—": "-", "≥": ">=", "≤": "<=", "→": "->", "≈": "~", "’": "'", "“": '"', "”": '"', "…": "..."}


def clean(s) -> str:
    """Standard PDF fonts are Latin-1: map common symbols, replace anything else, then escape for XML markup."""
    s = str(s)
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    return escape(s.encode("cp1252", "replace").decode("cp1252"))


@dataclass
class ReportData:
    report_id: str
    analysis_id: str
    generated_utc: str
    meta: dict                       # analysis meta.json
    preds: dict                      # sanitised predictions.json
    scene: Scene
    facts: dict                      # reasoning.report_facts.build_report_facts
    pre: np.ndarray
    post: np.ndarray
    map_full: np.ndarray             # three-class map WITH legend panel
    map_plain: np.ndarray            # three-class map without legend
    findings: list
    recommendations: list
    explanation: object              # explain.Explanation
    qa_user: list                    # [{id, question, answer, region, qtype, supporting_ids, crop (ndarray|None)}]
    qa_standard: list
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- styles

def styles():
    base = dict(fontName="Helvetica", textColor=colors.HexColor("#1e293b"))
    return {
        "body": ParagraphStyle("body", fontSize=9.4, leading=13, spaceAfter=4, **base),
        "small": ParagraphStyle("small", fontSize=7.8, leading=10, textColor=MUTED, fontName="Helvetica"),
        "cell": ParagraphStyle("cell", fontSize=7.8, leading=9.6, **base),
        "cellb": ParagraphStyle("cellb", fontSize=7.8, leading=9.6, fontName="Helvetica-Bold", textColor=colors.HexColor("#1e293b")),
        "th": ParagraphStyle("th", fontSize=7.8, leading=9.6, fontName="Helvetica-Bold", textColor=colors.white),
        "h1": ParagraphStyle("h1", fontSize=15, leading=19, fontName="Helvetica-Bold", textColor=NAVY, spaceBefore=4, spaceAfter=2, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontSize=11, leading=14, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=8, spaceAfter=3, keepWithNext=1),
        "bullet": ParagraphStyle("bullet", fontSize=9.4, leading=13, leftIndent=12, bulletIndent=2, spaceAfter=3, **base),
        "caption": ParagraphStyle("caption", fontSize=7.8, leading=10, textColor=MUTED, fontName="Helvetica-Oblique", alignment=TA_CENTER),
        "kpi": ParagraphStyle("kpi", fontSize=19, leading=22, fontName="Helvetica-Bold", textColor=NAVY, alignment=TA_CENTER),
        "kpil": ParagraphStyle("kpil", fontSize=7.8, leading=10, textColor=MUTED, fontName="Helvetica", alignment=TA_CENTER),
        "box": ParagraphStyle("box", fontSize=8.6, leading=11.6, **base),
    }


S = styles()
TAGS = {"computed": ("COMPUTED", "#0a1d3a"), "model": ("MODEL PREDICTION", "#2563eb"), "ai": ("AI EXPLANATION", "#64748b")}


def tag(kind: str) -> str:
    label, col = TAGS[kind]
    return f'<font color="white" backColor="{col}" size="6.5">&nbsp;{label}&nbsp;</font>&nbsp;'


def P(text, style="body"):
    return Paragraph(text, S[style])


def H1(num, title):
    return [P(f"{num}. {clean(title)}", "h1"), HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceAfter=6)]


def bullets(items, style="bullet"):
    return [Paragraph(t, S[style], bulletText="•") for t in items]


def table(data, widths, header=True, zebra=True, extra=None, repeat=1):
    t = Table(data, colWidths=widths, repeatRows=repeat if header else 0, splitByRow=1)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.25, LINE),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    if zebra:
        st += [("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, LIGHT])]
    t.setStyle(TableStyle(st + (extra or [])))
    return t


def th(*cols):
    return [Paragraph(clean(c), S["th"]) for c in cols]


def cell(x, bold=False):
    return Paragraph(clean(x), S["cellb" if bold else "cell"])


def box(text_html, color="#fff7ed", border="#fdba74"):
    t = Table([[Paragraph(text_html, S["box"])]], colWidths=[CONTENT_W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(color)), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


# --------------------------------------------------------------------------- images

def jpeg_image(img_bgr: np.ndarray, width: float, quality: int = 90, max_px: int = 1400) -> Image:
    h, w = img_bgr.shape[:2]
    if w > max_px:
        img_bgr = cv2.resize(img_bgr, (max_px, int(h * max_px / w)), interpolation=cv2.INTER_AREA)
        h, w = img_bgr.shape[:2]
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode image for the PDF")
    return Image(io.BytesIO(buf.tobytes()), width=width, height=width * h / w)


def fit_image(img_bgr: np.ndarray, max_w: float, max_h: float, quality: int = 88, max_px: int = 1100) -> Image:
    """Scale to fit inside max_w x max_h keeping the aspect ratio."""
    h, w = img_bgr.shape[:2]
    scale = min(max_w / w, max_h / h)
    im = jpeg_image(img_bgr, w * scale, quality, max_px)
    im.drawHeight, im.drawWidth = h * scale, w * scale
    return im


def square_thumb(img: np.ndarray, bbox, size_pt: float, pad: int = 24) -> Image:
    """Centre-cropped square thumbnail around a building, so before/after pairs line up."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    half = max(x2 - x1, y2 - y1) // 2 + pad
    half = max(half, 24)
    a, b = max(0, cx - half), max(0, cy - half)
    c = img[b:min(h, cy + half), a:min(w, cx + half)]
    c = cv2.resize(c, (200, 200), interpolation=cv2.INTER_AREA)
    return jpeg_image(c, size_pt, 88, 200)


def crop_box(img: np.ndarray, bbox, pad=24) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    return img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


# --------------------------------------------------------------------------- charts

def class_pie(counts: dict, total: int, affected_share) -> Drawing:
    d = Drawing(169, 150)
    if total == 0:
        d.add(String(84, 75, "No buildings detected", textAnchor="middle", fontSize=9, fillColor=MUTED))
        return d
    pie = Pie()
    pie.x, pie.y, pie.width, pie.height = 10, 5, 140, 140
    pie.data = [max(counts[n], 0) for n in CLASS_NAMES]
    pie.labels = None
    pie.sideLabels = 0
    pie.slices.strokeColor = colors.white
    pie.slices.strokeWidth = 1.2
    for i, n in enumerate(CLASS_NAMES):
        pie.slices[i].fillColor = colors.HexColor(CLASS_HEX[n])
    d.add(pie)
    d.add(Circle(80, 75, 42, fillColor=colors.white, strokeColor=colors.white))
    d.add(String(80, 76, pct(affected_share), textAnchor="middle", fontSize=15, fontName="Helvetica-Bold", fillColor=NAVY))
    d.add(String(80, 63, "affected", textAnchor="middle", fontSize=8, fillColor=MUTED))
    return d


def histogram(values: list[float]) -> Drawing:
    bins = [0] * 10
    for v in values:
        bins[min(9, max(0, int(v * 10)))] += 1
    d = Drawing(CONTENT_W, 165)
    bc = VerticalBarChart()
    bc.x, bc.y, bc.width, bc.height = 36, 34, CONTENT_W - 56, 112
    bc.data = [bins]
    bc.categoryAxis.categoryNames = [f"{i / 10:.1f}-{(i + 1) / 10:.1f}" for i in range(10)]
    bc.categoryAxis.labels.fontSize = 7
    bc.categoryAxis.labels.angle = 0
    bc.valueAxis.labels.fontSize = 7
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueStep = max(1, int(np.ceil(max(bins) / 5))) if max(bins) else 1
    bc.valueAxis.valueMax = (max(bins) or 1) * 1.1
    bc.bars[0].fillColor = BLUE
    bc.barLabelFormat = "%d"
    bc.barLabels.fontSize = 7
    bc.barLabels.nudge = 7
    d.add(bc)
    d.add(String(CONTENT_W / 2, 6, "Model confidence (detection score x damage-class probability)", textAnchor="middle", fontSize=7.5, fillColor=MUTED))
    return d


def quadrant_diagram(facts: dict) -> Drawing:
    w = 230
    d = Drawing(w, 188)
    pos = {"top-left": (10, 90), "top-right": (120, 90), "bottom-left": (10, 10), "bottom-right": (120, 10)}
    mx = max([facts["quadrants"][q]["damaged"] for q in QUADS] + [1])
    for q in QUADS:
        x, y = pos[q]
        n = facts["quadrants"][q]["damaged"]
        shade = 0.92 - 0.55 * (n / mx)
        d.add(Rect(x, y, 100, 72, fillColor=colors.Color(1, shade, shade), strokeColor=LINE))
        d.add(String(x + 50, y + 44, str(n), textAnchor="middle", fontSize=18, fontName="Helvetica-Bold", fillColor=NAVY))
        d.add(String(x + 50, y + 16, q, textAnchor="middle", fontSize=7.5, fillColor=MUTED))
    d.add(String(w / 2, 176, "Affected buildings per quadrant", textAnchor="middle", fontSize=8, fillColor=MUTED))
    return d


# --------------------------------------------------------------------------- page furniture

class NumberedCanvas(rl_canvas.Canvas):
    page_total = 0

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        NumberedCanvas.page_total = n
        for state in self._saved:
            self.__dict__.update(state)
            if self._pageNumber > 1:                           # cover has no running footer
                self.setFont("Helvetica", 8)
                self.setFillColor(MUTED)
                self.drawRightString(PAGE_W - MARGIN, 11 * mm, f"Page {self._pageNumber} of {n}")
            super().showPage()
        super().save()


def make_page_callbacks(d: ReportData):
    short = d.analysis_id[:8]

    def cover(c, doc):
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, PAGE_H - 290, PAGE_W, 290, stroke=0, fill=1)
        c.setFillColor(BLUE)
        c.rect(0, PAGE_H - 296, PAGE_W, 6, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 38)
        c.drawString(MARGIN, PAGE_H - 110, "EAGLE-DAR")
        c.setFont("Helvetica", 11)
        c.drawString(MARGIN, PAGE_H - 132, "Evidence-Aware Geospatial Learning and Explanation for")
        c.drawString(MARGIN, PAGE_H - 146, "Disaster Assessment and Reasoning")
        c.setFont("Helvetica-Bold", 24)
        c.drawString(MARGIN, PAGE_H - 214, "Disaster Damage Assessment Report")
        c.setFont("Helvetica", 11)
        c.drawString(MARGIN, PAGE_H - 238, "AI-assisted building-level damage assessment from before/after satellite imagery")
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.drawString(MARGIN, 11 * mm, "AI-assisted assessment - model predictions require field verification")
        c.restoreState()

    def body(c, doc):
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN, PAGE_H - 9 * mm, "EAGLE-DAR  |  Disaster Damage Assessment Report")
        c.setFont("Helvetica", 8)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 9 * mm, f"Analysis {short}")
        c.setStrokeColor(LINE)
        c.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN, 11 * mm, "AI-assisted assessment - verify before operational use")
        c.restoreState()
    return cover, body


# --------------------------------------------------------------------------- sections

def sec_cover(d: ReportData):
    ctx = d.meta.get("context") or {}
    img = d.meta["image"]
    thumbs = Table([[jpeg_image(d.pre, 235, 85, 700), jpeg_image(d.post, 235, 85, 700)],
                    [Paragraph("Before disaster", S["caption"]), Paragraph("After disaster", S["caption"])]],
                   colWidths=[CONTENT_W / 2] * 2)
    thumbs.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    rows = [
        ("Disaster name", ctx.get("disaster_name") or "Not supplied"),
        ("Analysis ID", d.analysis_id),
        ("Report ID", d.report_id),
        ("Analysis date/time (UTC)", d.meta["created_utc"]),
        ("Report generated (UTC)", d.generated_utc),
        ("Image pair", f"{img['before']['display_name']}  /  {img['after']['display_name']}"),
        ("Image size", f"{img['width']} x {img['height']} px"),
    ]
    if ctx.get("district") or ctx.get("state"):
        rows.insert(1, ("Location (as entered by user)", ", ".join(x for x in (ctx.get("district"), ctx.get("state")) if x)))
    t = table([[cell(k, True), cell(v)] for k, v in rows], [150, CONTENT_W - 150], header=False)
    return [Spacer(1, 6), thumbs, Spacer(1, 14), t, Spacer(1, 10),
            P("Location labels, if shown, are exactly what the user typed; they are not derived from the imagery, which carries "
              "no geographic coordinates.", "small"), NextPageTemplate("body"), PageBreak()]


def kpi_cell(value, label):
    return [Paragraph(clean(value), S["kpi"]), Paragraph(clean(label), S["kpil"])]


def sec_summary(d: ReportData):
    a = affected_counts(d.facts)
    cells = [kpi_cell(a["total"], "Total detected buildings"), kpi_cell(a["unaffected"], "Unaffected (no damage)"),
             kpi_cell(a["affected"], "Affected (major + destroyed)"), kpi_cell(a["severe"], "Severely damaged (destroyed)"),
             kpi_cell(pct(a["damaged_fraction"]), "Damage percentage (affected / total)"), kpi_cell(f"{d.seconds:.1f} s", "Analysis duration (model inference)")]
    t = Table([cells[:3], cells[3:]], colWidths=[CONTENT_W / 3] * 3)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.4, LINE), ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    out = H1(2, "Executive Summary") + [t, Spacer(1, 6)]
    out.append(P(tag("computed") + clean("Affected = major damage + destroyed (minor damage is reported within major damage).")))
    out.append(P(tag("ai") + clean(d.explanation.paragraphs["overall"])))
    out.append(Spacer(1, 4))
    out.append(box(f"{tag('computed')} deterministic results of the spatial reasoning engine &nbsp; {tag('model')} output of the YOLO + Siamese models "
                   f"&nbsp; {tag('ai')} wording only ({clean(d.explanation.source)}); it never supplies numbers.", "#f1f5f9", "#cbd5e1"))
    return out


def sec_before_after(d: ReportData):
    out = H1(3, "Before-and-After Satellite Image Analysis")
    w = CONTENT_W / 2 - 4
    t = Table([[jpeg_image(d.pre, w, 90), jpeg_image(d.post, w, 90)],
               [Paragraph("Before disaster (original)", S["caption"]), Paragraph("After disaster (original)", S["caption"])]],
              colWidths=[CONTENT_W / 2] * 2)
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    out += [t, Spacer(1, 6), jpeg_image(d.map_plain, CONTENT_W * 0.62, 90),
            Paragraph("Annotated after image: model-predicted building footprints coloured by predicted damage class", S["caption"]), Spacer(1, 6)]
    f = d.facts
    a = affected_counts(f)
    obs = (f"The model compared each detected building's before and after appearance. It flagged {a['affected']} of {a['total']} "
           f"detected buildings as affected between the two images ({pct(a['damaged_fraction'])}); the rest were classified as no damage "
           f"({a['unaffected']}). Differences between the images that are not buildings "
           "(vegetation, water, lighting, cloud) are outside the model's classification.")
    out.append(P(tag("model") + clean(obs)))
    al = (f["data_quality"].get("alignment") or {})
    if al:
        out.append(P(tag("computed") + clean(f"Before/after alignment check: {al.get('status')}, estimated shift {al.get('shift_px', 0):.1f} px "
                                              f"({al.get('method')}). {al.get('note') or ''}")))
    pri = [p for p in f["priority_buildings"] if p["damage"] == "destroyed"][:4]
    if pri:
        out.append(P("Buildings predicted destroyed (highest model confidence), before | after", "h2"))
        cells = []
        for p in pri:
            bb = p["bbox"]
            pair = Table([[square_thumb(d.pre, bb, 54), square_thumb(d.post, bb, 54)],
                          [Paragraph(clean(f"ID {p['building_id']}  conf {p['confidence']:.2f}" if p["confidence"] is not None else f"ID {p['building_id']}"), S["caption"]), ""]],
                         colWidths=[56, 56])
            pair.setStyle(TableStyle([("SPAN", (0, 1), (1, 1)), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
            cells.append(pair)
        t2 = Table([cells], colWidths=[CONTENT_W / 4] * len(cells), hAlign='LEFT')
        out += [t2, P("Crops are model predictions shown for field verification. They are not confirmation that a building was destroyed.", "small")]
    return out


def sec_classification(d: ReportData):
    f = d.facts
    total = f["scene"]["total_buildings"]
    c = f["damage_counts"]
    rows = [th("", "Damage class", "Buildings", "Share")]
    extra = []
    for i, n in enumerate(CLASS_NAMES, start=1):
        rows.append(["", cell(CLASS_TITLE[n], True), cell(c[n]), cell(pct(c[n] / total) if total else "n/a")])
        extra.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(CLASS_HEX[n])))
    rows.append(["", cell("Total detected", True), cell(total), cell("100.0%" if total else "n/a")])
    rows.append(["", cell("Affected (major + destroyed)", True), cell(f["damaged_total"]), cell(pct(f["damaged_fraction"]))])
    t = table(rows, [14, 200, 60, CONTENT_W - 169 - 274], extra=extra)
    layout = Table([[class_pie(c, total, f["damaged_fraction"]), t]], colWidths=[169, CONTENT_W - 169])
    layout.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER"),
                                ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (1, 0), (1, 0), 0)]))
    out = H1(4, "Building Damage Classification") + [P(tag("model") + clean(f"{total} buildings were detected and classified into three damage categories.")), Spacer(1, 4), layout,
                                                       Spacer(1, 6), P("The three damage categories", "h2")]
    defs = [[cell("", False), cell(CLASS_TITLE[n], True), cell(CLASS_DEFINITIONS[n])] for n in CLASS_NAMES]
    ex = [("BACKGROUND", (0, i), (0, i), colors.HexColor(CLASS_HEX[n])) for i, n in enumerate(CLASS_NAMES)]
    out += [table(defs, [14, 90, CONTENT_W - 104], header=False, extra=ex),
            Spacer(1, 4), P("Category descriptions paraphrase the xBD joint damage scale used by the PS-04 labels. Colours match the damage map and the dashboard.", "small")]
    return out


def conf_level(b) -> str:
    return (b.metadata or {}).get("confidence_level") or "-"


def sec_building_table(d: ReportData):
    cited: dict = {}
    for q in d.qa_user:
        for bid in q["supporting_ids"]:
            cited.setdefault(bid, []).append(q["id"])
    bs = sorted(d.scene.buildings, key=lambda b: b.building_id)
    out = H1(5, "Building-Level Assessment")
    if not bs:
        return out + [P(tag("model") + "No buildings were detected, so there are no building-level rows.")]
    out.append(P(tag("model") + clean(f"All {len(bs)} detected buildings are listed. Model confidence is an uncalibrated score "
                                      "(detection score x damage-class probability); it is not the probability that the prediction is correct. "
                                      "Evidence = bounding box in image pixels (x1,y1,x2,y2) and any user question that cites the building.")))
    rows = [th("", "ID", "Damage class", "Confidence", "Level", "Centre (x, y)", "Evidence")]
    extra = []
    for i, b in enumerate(bs, start=1):
        ev = f"bbox {b.bbox[0]},{b.bbox[1]},{b.bbox[2]},{b.bbox[3]}" + (f";  cited by {', '.join(cited[b.building_id])}" if b.building_id in cited else "")
        rows.append(["", cell(b.building_id), cell(CLASS_TITLE[b.damage_name]), cell("n/a" if b.confidence is None else f"{b.confidence:.3f}"),
                     cell(conf_level(b)), cell(f"{b.centre[0]:.1f}, {b.centre[1]:.1f}"), cell(ev)])
        extra.append(("BACKGROUND", (0, i), (0, i), colors.HexColor(CLASS_HEX[b.damage_name])))
    out.append(table(rows, [12, 28, 78, 52, 40, 80, CONTENT_W - 290], extra=extra))
    return out


def sec_map(d: ReportData):
    out = H1(6, "Five-Class Damage Map")
    out += [P(tag("model") + clean(f"{len(d.scene.buildings)} predicted building footprints are drawn from the same prediction data as the tables and charts in this report. "
                                    "Dashed lines mark the image-centre split used by the spatial rules. Thin outlines indicate low model confidence.")),
            jpeg_image(d.map_full, CONTENT_W, 93, 1800),
            Paragraph("Damage map. Orientation: top of the image is the top of the supplied after image; origin (0,0) is the top-left corner. "
                      "Coordinates are image pixels, not geographic coordinates.", S["caption"])]
    return out


def sec_spatial(d: ReportData):
    f = d.facts
    rows = [th("Quadrant", *[CLASS_TITLE[n] for n in CLASS_NAMES], "Affected", "Region (x1,y1,x2,y2)")]
    for q in QUADS:
        r = f["quadrants"][q]
        rows.append([cell(q, True), *[cell(r[n]) for n in CLASS_NAMES], cell(r["damaged"], True), cell(", ".join(str(v) for v in r["region"]))])
    qt = table(rows, [62, 40, 40, 44, 56, 38, 42, CONTENT_W - 322])
    out = H1(7, "Spatial Damage Assessment")
    out += [P(tag("computed") + clean("Rules (official PS-04 conventions): the image is split at its centre; a building belongs to the quadrant or half that "
                                       "contains its centre; 'damaged' means major + destroyed; counts are individual buildings, not pixels.")),
            qt, Spacer(1, 8)]
    diag = quadrant_diagram(f)
    hc = f["half_comparisons"]
    hr = [th("Comparison", "Damaged (first half)", "Damaged (second half)", "More damaged")]
    for key, (a, b) in (("left_vs_right", ("left", "right")), ("top_vs_bottom", ("top", "bottom"))):
        dc = hc[key].get("damaged_counts")
        if dc:
            more = a if dc[a] > dc[b] else b if dc[b] > dc[a] else "equal"
            hr.append([cell(f"{a} vs {b}", True), cell(f"{a}: {dc[a]}"), cell(f"{b}: {dc[b]}"), cell(more)])
        else:
            hr.append([cell(f"{a} vs {b}", True), cell("n/a"), cell("n/a"), cell("cannot be determined (no buildings)")])
    side = Table([[diag, table(hr, [58, 62, 62, CONTENT_W - 240 - 182 - 6])]], colWidths=[240, CONTENT_W - 240])
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    out.append(side)
    mq = f["most_damaged_quadrant"]
    qc = mq.get("quadrant_counts") or {}
    if mq["answer"] in QUADS:
        txt = f"Quadrant with the most damaged buildings: {mq['answer']} ({qc.get(mq['answer'])} buildings)."
    elif mq.get("tie_unresolved") and f["damaged_total"] > 0:
        txt = "No single quadrant has the most damaged buildings (tie); the answer is 'cannot be determined' under the official convention used here."
    elif f["damaged_total"] == 0 and f["scene"]["total_buildings"] > 0:
        txt = "No damaged buildings were detected in any quadrant, so no quadrant can be named."
    else:
        txt = "No quadrant statistics are available because no buildings were detected."
    out += [Spacer(1, 6), P(tag("computed") + clean(txt)),
            P(tag("ai") + clean(d.explanation.paragraphs["spatial"]))]
    return out


def region_text(r) -> str:
    return "none" if r is None else (r if isinstance(r, str) else "[" + ", ".join(str(v) for v in r) + "]")


def qa_rows(items):
    rows = [th("Question ID", "Question", "Official answer", "Evidence region", "Supporting building IDs")]
    for q in items:
        ids = q["supporting_ids"]
        shown = ", ".join(str(i) for i in ids[:14]) + (f" (+{len(ids) - 14} more)" if len(ids) > 14 else "") if ids else "-"
        rows.append([cell(q["id"], True), cell(q["question"]), cell(q["answer"], True), cell(region_text(q["region"])), cell(shown)])
    return table(rows, [82, 140, 52, 80, CONTENT_W - 354])


def evidence_grid(items):
    cells, caps = [], []
    w = CONTENT_W / 2 - 8
    for q in items:
        if q.get("crop") is None:
            continue
        cells.append(fit_image(q["crop"], w, 195, 85, 900))
        ids = q["supporting_ids"]
        caps.append(Paragraph(clean(f"{q['id']}: answer {q['answer']}; region {region_text(q['region'])}; "
                                    f"{len(ids)} supporting building(s)"), S["caption"]))
    out = []
    for i in range(0, len(cells), 2):
        row_i, row_c = cells[i:i + 2], caps[i:i + 2]
        while len(row_i) < 2:
            row_i.append("")
            row_c.append("")
        t = Table([row_i, row_c], colWidths=[CONTENT_W / 2] * 2)
        t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        out.append(KeepTogether([t, Spacer(1, 6)]))
    return out


def sec_qa(d: ReportData):
    out = H1(8, "Evidence and Question-Answering Results")
    out.append(P(tag("computed") + clean("All answers below are computed by the deterministic spatial reasoning engine from the predicted buildings. "
                                          "No language model calculates or alters a count, answer or evidence region. Official answers follow the PS-04 submission format "
                                          "(id, answer, region); supporting building IDs are shown here for traceability only.")))
    if d.qa_user:
        out += [P("Questions asked for this analysis", "h2"), qa_rows(d.qa_user)]
    else:
        out.append(P("No questions were asked for this analysis in the application. The standard assessment queries below are computed automatically.", "small"))
    out += [P("Standard assessment queries (computed automatically)", "h2"), qa_rows(d.qa_standard), Spacer(1, 4),
            P("Evidence regions and supporting buildings", "h2"),
            P("Magenta dashed rectangles mark the official evidence region; thick outlines mark supporting buildings; other buildings are faded.", "small")]
    out += evidence_grid(d.qa_user + [q for q in d.qa_standard if q.get("crop") is not None])
    return out


def sec_confidence(d: ReportData):
    f = d.facts
    conf = f["confidence"]
    vals = [b.confidence for b in d.scene.buildings if b.confidence is not None]
    out = H1(9, "Confidence and Reliability Assessment")
    if not vals:
        return out + [P(tag("model") + "No building-level confidence values are available (no buildings were detected).")]
    lv = conf["levels"]
    st = table([[cell("Mean model confidence", True), cell(f"{conf['mean']:.3f}"), cell("Minimum / maximum", True), cell(f"{conf['min']:.3f} / {conf['max']:.3f}")],
                [cell("High / medium / low", True), cell(f"{lv['high']} / {lv['medium']} / {lv['low']}"), cell("Below threshold", True),
                 cell(f"{len(conf['low_confidence_building_ids'])} building(s) < {conf['low_confidence_threshold']}")]],
               [100, 100, 100, CONTENT_W - 300], header=False, zebra=False)
    out += [P(tag("model") + clean("Model confidence = detection score x damage-class probability. It is an uncalibrated indicator of how strongly the model supports "
                                    "a prediction. It is NOT the probability that the prediction is correct, and the mean is NOT the model's accuracy.")),
            st, Spacer(1, 6), histogram(vals)]
    low = sorted(conf["low_confidence_building_ids"])
    out.append(P("Low-confidence buildings", "h2"))
    if low:
        out.append(P(tag("computed") + clean(f"{len(low)} building(s) fall below the {conf['low_confidence_threshold']} threshold; all IDs are listed: " + ", ".join(str(i) for i in low))))
    else:
        out.append(P(tag("computed") + "No building falls below the low-confidence threshold."))
    lowest = sorted((b for b in d.scene.buildings if b.confidence is not None), key=lambda b: b.confidence)[:15]
    rows = [th("", "Building ID", "Damage class", "Confidence")] + [["", cell(b.building_id), cell(CLASS_TITLE[b.damage_name]), cell(f"{b.confidence:.3f}")] for b in lowest]
    ex = [("BACKGROUND", (0, i), (0, i), colors.HexColor(CLASS_HEX[b.damage_name])) for i, b in enumerate(lowest, start=1)]
    out += [P("Fifteen lowest-confidence buildings", "h2"), table(rows, [14, 70, 110, 70], extra=ex),
            Spacer(1, 4), P(tag("ai") + clean(d.explanation.paragraphs["reliability"]))]
    return out


def sec_findings(d: ReportData):
    out = H1(10, "Key Findings")
    out.append(P(tag("computed") + "Each finding is generated from the structured results; the source field names the data it comes from."))
    out += bullets([f"{clean(x['text'])} <font size='7' color='#64748b'>[{clean(x['source'])}]</font>" for x in d.findings])
    return out


def sec_recs(d: ReportData):
    out = H1(11, "Disaster Response Recommendations")
    out.append(box("<b>AI-assisted recommendations requiring human verification.</b> They are rule-based suggestions triggered by the results in this report. "
                   "They are not confirmed ground truth, not safety declarations and not official guidance."))
    out.append(Spacer(1, 4))
    out += bullets([f"{clean(x['text'])} <font size='7' color='#64748b'>[trigger: {clean(x['trigger'])}]</font>" for x in d.recommendations])
    return out


def sec_method(d: ReportData):
    p = d.preds
    prm = p.get("parameters", {})
    mdl = p.get("models", {})
    ev = (d.facts["data_quality"].get("evaluation_context") or {})
    out = H1(12, "Methodology and Technical Details")
    steps = [("1. Validation", "Both images are validated (JPG/PNG, identical dimensions, size limits) and stored under a unique analysis ID."),
             ("2. Alignment check", "Global before/after offset is measured by phase correlation (ORB cross-check); optional translation registration."),
             ("3. Building detection", "YOLOv8 segmentation on the after image gives building polygons (footprints), centres and bounding boxes."),
             ("4. Crop matching", "For each footprint the same pixel window is cut from the before and after images and resized, as in training."),
             ("5. Damage classification", "A Siamese ResNet18 compares the two crops (absolute feature difference) and outputs three-class probabilities."),
             ("6. Building confidence", "Model confidence = detection score x probability of the predicted class (uncalibrated)."),
             ("7. Spatial reasoning", "Deterministic code computes counts, quadrants, half comparisons and evidence regions (PS-04 rules)."),
             ("8. Report", f"This report: computed facts plus {clean(d.explanation.source)} wording.")]
    out.append(table([[cell(a, True), cell(b)] for a, b in steps], [110, CONTENT_W - 110], header=False))
    info = [("Detector weights", f"{mdl.get('yolo', {}).get('path', 'n/a')} (sha256 {mdl.get('yolo', {}).get('sha256_16', 'n/a')}...)"),
            ("Classifier weights", f"{mdl.get('siamese', {}).get('path', 'n/a')} (sha256 {mdl.get('siamese', {}).get('sha256_16', 'n/a')}...)"),
            ("Detection settings", f"confidence {prm.get('yolo_conf', 'n/a')}, image size {prm.get('yolo_imgsz', 'n/a')}, IoU {prm.get('yolo_iou', 'n/a')}"),
            ("Crop settings", f"padding {prm.get('crop_padding_px', 'n/a')} px, size {prm.get('crop_size_px', 'n/a')} px"),
            ("Device / software", f"{prm.get('device', 'n/a')}; torch {prm.get('torch', 'n/a')}; python {prm.get('python', 'n/a')}"),
            ("Alignment mode", prm.get("alignment_mode", "n/a")),
            ("Explanation source", d.explanation.source + (f" (fallback: {d.explanation.fallback_reason})" if d.explanation.fallback_reason else ""))]
    out += [P("Models and settings used for this analysis", "h2"), table([[cell(a, True), cell(b)] for a, b in info], [110, CONTENT_W - 110], header=False)]
    sv, yv = ev.get("siamese_validation"), ev.get("yolo_validation")
    if sv or yv:
        out.append(P("Measured model performance (held-out validation data, not training data)", "h2"))
        rows = [th("Measure", "Value")]
        if yv:
            rows += [[cell("Detector mask precision / recall / mAP50"), cell(f"{yv['mask_precision']:.3f} / {yv['mask_recall']:.3f} / {yv['mask_mAP50']:.3f}")]]
        if sv:
            rows += [[cell("Classifier accuracy / macro F1"), cell(f"{sv['accuracy']:.3f} / {sv['macro_f1']:.3f}  ({sv['n_val']} validation pairs)")],
                     [cell("Classifier recall by class"), cell(", ".join(f"{CLASS_TITLE[k]} {v:.2f}" for k, v in sv["recall_by_class"].items()))]]
        out += [table(rows, [190, CONTENT_W - 190]),
                P("Validation data was also used to choose the saved checkpoint, so these figures are indicative, not an independent test. "
                  "The validation set is dominated by the No Damage class; see the evaluation report for imbalance and the full confusion matrix.", "small")]
    else:
        out.append(P("No evaluation files were found, so measured model performance is not quoted here."))
    return out


def sec_limits(d: ReportData):
    ev = (d.facts["data_quality"].get("evaluation_context") or {})
    out = H1(13, "Limitations and Disclaimer")
    items = ["Predictions are based only on the supplied satellite imagery and the two models; they are estimates, not confirmed ground truth.",
             "Model predictions require field verification before any operational decision.",
             "Model confidence scores indicate how strongly the model supports a prediction; they do not guarantee correctness and are not calibrated.",
             "Geographic information is unavailable: the images carry no georeferencing, so coordinates are image pixels and no district, area or location is inferred.",
             "The detector can miss buildings and the classifier can miss damage classes, so counts are not a complete census."]
    yv, sv = ev.get("yolo_validation"), ev.get("siamese_validation")
    if yv:
        items.append(f"Measured: on held-out validation images the detector's mask recall was {yv['mask_recall']:.2f}, so buildings it misses are absent from every count.")
    if sv:
        items.append("Measured per-class recall of the damage classifier on validation pairs: " + ", ".join(f"{CLASS_TITLE[k]} {v:.2f}" for k, v in sv["recall_by_class"].items())
                     + ". Classes with near-zero recall are effectively not detected, so their counts are unreliable.")
    items.append("This report has not been validated against the official PS-04 hidden-test evaluation, and no accuracy target is claimed.")
    items.append("The report supports, and does not replace, professional disaster assessment.")
    out += bullets([clean(i) for i in items])
    return out


# --------------------------------------------------------------------------- build

def build_pdf(path, d: ReportData, progress: Callable[[str, float], None] | None = None) -> int:
    """Write the PDF to `path`; return the page count."""
    note = progress or (lambda *_: None)
    cover_cb, body_cb = make_page_callbacks(d)
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=20 * mm, bottomMargin=19 * mm,
                          title="EAGLE-DAR Disaster Damage Assessment Report", author="EAGLE-DAR",
                          subject=f"Analysis {d.analysis_id}", creator="EAGLE-DAR report generator")
    cover_frame = Frame(MARGIN, 19 * mm, CONTENT_W, PAGE_H - 296 - 19 * mm - 12, id="cover", leftPadding=0, rightPadding=0, topPadding=6, bottomPadding=0)
    body_frame = Frame(MARGIN, 19 * mm, CONTENT_W, PAGE_H - 20 * mm - 19 * mm, id="body", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="cover", frames=[cover_frame], onPage=cover_cb),
                          PageTemplate(id="body", frames=[body_frame], onPage=body_cb)])
    story: list = []
    def findings_and_recs(dd):
        return sec_findings(dd) + [Spacer(1, 8)] + sec_recs(dd)

    def method_and_limits(dd):
        return sec_method(dd) + [Spacer(1, 8)] + sec_limits(dd)

    # (name, builder, break before): "page" = always new page, a number = new page only if less than that many points remain
    stages = [("Cover page", sec_cover, None), ("Executive summary", sec_summary, None), ("Image analysis", sec_before_after, 330),
              ("Damage classification", sec_classification, 330), ("Building-level table", sec_building_table, "page"),
              ("Damage map", sec_map, 470), ("Spatial assessment", sec_spatial, 330), ("Evidence and questions", sec_qa, 300),
              ("Confidence assessment", sec_confidence, 420), ("Findings and recommendations", findings_and_recs, 300),
              ("Methodology and limitations", method_and_limits, 300)]
    for i, (name, fn, brk) in enumerate(stages):
        note(f"Building section: {name}", 0.35 + 0.5 * i / len(stages))
        if brk == "page":
            story.append(PageBreak())
        elif brk:
            story.append(CondPageBreak(brk))
        story += fn(d)
    note("Writing PDF file", 0.9)
    doc.build(story, canvasmaker=NumberedCanvas)
    return NumberedCanvas.page_total
