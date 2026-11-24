import { Card, PageHeader } from '../components/common'
import { DAMAGE_META, DAMAGE_ORDER } from '../lib/damage'

export function About() {
  return (
    <>
      <PageHeader title="About EAGLE-DAR" subtitle="Evidence-Aware Geospatial Learning and Explanation for Disaster Assessment and Reasoning" />
      <div className="grid-2">
        <Card title="Project">
          <p>EAGLE-DAR compares before and after satellite images of a disaster area, finds each building, grades its damage in three classes and explains the result with evidence: a colour-coded map, building-level confidence scores and a written assessment report.</p>
          <p className="muted small">Research prototype. It is not an official government system and carries no government endorsement.</p>
        </Card>
        <Card title="Problem statement (PS-04)">
          <blockquote>Answer counting and comparison questions about before-and-after satellite images of disaster areas, and say &lsquo;cannot be determined&rsquo; when the images cannot answer.</blockquote>
          <p className="muted small">Quoted unchanged from the PS-04 dataset description.</p>
        </Card>
      </div>
      <Card title="AI pipeline">
        <ol className="pipeline">
          <li><b>Validate</b><span>JPG/PNG, identical dimensions, size limits</span></li>
          <li><b>Align check</b><span>Measure before/after offset</span></li>
          <li><b>Segment</b><span>YOLOv8-seg finds building polygons on the after image</span></li>
          <li><b>Classify</b><span>Siamese ResNet18 compares each building&rsquo;s before/after crop into three classes</span></li>
          <li><b>Reason</b><span>Deterministic Python: counts, quadrants, comparisons, evidence regions</span></li>
          <li><b>Report</b><span>Map, confidence, assessment report</span></li>
        </ol>
      </Card>
      <div className="grid-2">
        <Card title="Three project novelties">
          <ol className="novelties">
            <li><b>Building-level confidence score</b> for every building.</li>
            <li><b>Three-class colour-coded building damage map.</b>
              <div className="legend legend-compact">{DAMAGE_ORDER.map((n) => <span key={n} className="chip"><i style={{ background: DAMAGE_META[n].color }} />{DAMAGE_META[n].label}</span>)}</div></li>
            <li><b>Detailed AI-generated disaster assessment report</b> with images.</li>
          </ol>
        </Card>
        <Card title="Technology stack">
          <ul className="bullets">
            <li>Models: YOLOv8-seg (Ultralytics), Siamese ResNet18 (PyTorch)</li>
            <li>Backend: Python, FastAPI, OpenCV, Shapely</li>
            <li>Spatial reasoning: deterministic Python engine, no LLM arithmetic</li>
            <li>Frontend: React, TypeScript, Vite, Recharts, React Router</li>
          </ul>
        </Card>
      </div>
      <Card title="Data and limitations">
        <p>Training and demonstration imagery comes from xBD / xView2 (Gupta et al., 2019; Maxar Open Data), CC BY-NC-SA 4.0, used here for non-commercial purposes with attribution.</p>
        <p>All results are model estimates. Building recall and major-damage recall are limited; the Reports page quotes the measured figures. Pixel coordinates in this interface are image coordinates, not geographic coordinates.</p>
      </Card>
    </>
  )
}
