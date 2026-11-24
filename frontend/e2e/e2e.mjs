// End-to-end test of the whole platform in a real browser, following the hackathon demonstration sequence.
//
//   backend:   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000      (project root)
//   frontend:  npm run dev                                                          (http://localhost:5173)
//   run:       npm run e2e        (uses an installed Chrome/Edge; set CHROME_PATH to override)
//
// Real models and real practice images are used; nothing is mocked except where a test deliberately simulates a
// failure (aborted requests, a failed report status). Screenshots go to ../outputs/demo (git-ignored).
import { chromium } from 'playwright-core'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(HERE, '..', '..')
const UI = process.env.UI_URL || 'http://localhost:5173'
const API = process.env.API_URL || 'http://127.0.0.1:8000'
const PRACTICE = path.join(ROOT, 'dataset_source', 'practice', 'images')
const SHOTS = path.join(ROOT, 'outputs', 'demo'); fs.mkdirSync(SHOTS, { recursive: true })
const TMP = path.join(ROOT, 'outputs', 'demo', 'tmp'); fs.mkdirSync(TMP, { recursive: true })
const CHROME = process.env.CHROME_PATH || [
  'C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'].find((p) => fs.existsSync(p))
if (!CHROME) { console.error('No Chrome/Edge found. Set CHROME_PATH.'); process.exit(2) }

let failed = 0, passed = 0
const check = (label, cond, extra = '') => { console.log(`${cond ? 'PASS' : 'FAIL'} ${label}${extra !== '' ? '  [' + extra + ']' : ''}`); cond ? passed++ : failed++ }
const num = (s) => Number(String(s).replace(/[^\d.]/g, ''))
const section = (t) => console.log(`\n--- ${t}`)
const api = async (p, init) => (await fetch(API + p, init)).json()
const COLOURS = { 'no-damage': '#4caf50', 'major-damage': '#ff9800', destroyed: '#f44336' }

const browser = await chromium.launch({ executablePath: CHROME, headless: true })
const ctx = await browser.newContext({ viewport: { width: 1672, height: 941 }, acceptDownloads: true })
const page = await ctx.newPage()
const problems = []
page.on('pageerror', (e) => problems.push('pageerror: ' + e.message))
page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) problems.push('console: ' + m.text()) })
let analyzePosts = 0                                  // analyses actually CREATED (HTTP 201), not attempts
ctx.on('response', (r) => { if (r.url().endsWith('/api/analyze') && r.request().method() === 'POST' && r.status() === 201) analyzePosts++ })
const shot = (n) => page.screenshot({ path: path.join(SHOTS, n + '.png') })
const pair = (id) => [path.join(PRACTICE, `${id}_pre.jpg`), path.join(PRACTICE, `${id}_post.jpg`)]

async function upload(pg, id, { name } = {}) {
  await pg.goto(UI + '/upload')
  const inputs = pg.locator('input[type=file]')
  const [b, a] = pair(id)
  await inputs.nth(0).setInputFiles(b); await inputs.nth(1).setInputFiles(a)
  await pg.getByText('1024 × 1024 px').first().waitFor()
  if (name) await pg.getByPlaceholder('e.g. Midwest flooding').fill(name)
  await pg.getByRole('button', { name: /Start Disaster Analysis/ }).click()
  await pg.getByText('Analysis complete').waitFor({ timeout: 240000 })
  const id_ = (await pg.locator('.banner-ok code').textContent()).trim()
  await pg.waitForURL(`${UI}/?a=${id_}`, { timeout: 15000 }).catch(() => {})
  await pg.locator('.kpi-value').first().waitFor()
  return id_
}

// =====================================================================================================
section('1. Dashboard opens (empty state) and the backend is reachable')
await page.goto(UI + '/')
await page.getByText('System Online').waitFor({ timeout: 20000 })
check('system online, from the backend health check', true)
check('empty state before any analysis; KPIs are not sample numbers', (await page.getByText('No analysis yet').count()) > 0 && (await page.locator('.kpi-value').allTextContents()).every((t) => t.trim() === '—'))
await shot('01_dashboard_empty')

section('2. Upload validation (Test 1) and failure handling (Test 7)')
await page.goto(UI + '/upload')
const inputs = page.locator('input[type=file]')
await inputs.nth(1).setInputFiles(pair('pb67aa0fe')[1])
await page.getByText('1024 × 1024 px').first().waitFor()
check('missing BEFORE image: start disabled with a clear hint', (await page.getByRole('button', { name: /Start Disaster Analysis/ }).isDisabled()) && (await page.getByText('Add both images to continue.').count()) > 0)
await page.reload()
await inputs.nth(0).setInputFiles(pair('pb67aa0fe')[0])
await page.getByText('1024 × 1024 px').first().waitFor()
check('missing AFTER image: start disabled', await page.getByRole('button', { name: /Start Disaster Analysis/ }).isDisabled())
await inputs.nth(1).setInputFiles(path.join(HERE, 'fixtures', 'not_an_image.gif'))
check('invalid file format rejected before upload', await page.getByText('Unsupported format').isVisible())
const truncated = path.join(TMP, 'corrupt.jpg'); fs.writeFileSync(truncated, fs.readFileSync(pair('pb67aa0fe')[1]).subarray(0, 4000))
await inputs.nth(1).setInputFiles(truncated)
await page.waitForTimeout(800)
const clientCaught = (await page.locator('.inline-error').count()) > 0 || (await page.getByRole('button', { name: /Start Disaster Analysis/ }).isDisabled())
if (!clientCaught) {   // browsers can partly decode a truncated JPEG: the backend must then reject it with a clear message
  await page.getByRole('button', { name: /Start Disaster Analysis/ }).click()
  await page.locator('.banner-error').waitFor({ timeout: 30000 })
  const msg = (await page.locator('.banner-error').innerText()).toLowerCase()
  check('corrupt image: a clear error is shown (backend validation) and nothing is created', msg.includes('corrupted') && analyzePosts === 0, msg.split(String.fromCharCode(10))[0])
} else check('corrupt image rejected in the browser before upload', true)
await inputs.nth(1).setInputFiles(path.join(HERE, 'fixtures', 'synthetic_512.jpg'))
await page.locator('.inline-error', { hasText: 'identical dimensions' }).waitFor()
check('different image sizes flagged; start disabled', await page.getByRole('button', { name: /Start Disaster Analysis/ }).isDisabled())
// server-side validation as well (bypassing the browser checks)
const fd = new FormData(); fd.append('before', new Blob([fs.readFileSync(truncated)]), 'x.jpg'); fd.append('after', new Blob([fs.readFileSync(pair('pb67aa0fe')[1])]), 'y.jpg')
const r422 = await fetch(API + '/api/analyze', { method: 'POST', body: fd })
check('backend also rejects a corrupt image (422 with an error code)', r422.status === 422 && (await r422.json()).error.code === 'corrupted_image')
const fd2 = new FormData(); fd2.append('after', new Blob([fs.readFileSync(pair('pb67aa0fe')[1])]), 'y.jpg')
check('backend rejects a missing before file (422)', (await fetch(API + '/api/analyze', { method: 'POST', body: fd2 })).status === 422)

section('3. Real upload -> analysis (Tests 1-2); demo steps 2-6')
await page.goto(UI + '/upload')
const t0 = Date.now()
const id = await (async () => {
  const [b, a] = pair('pb67aa0fe')
  await inputs.nth(0).setInputFiles(b); await inputs.nth(1).setInputFiles(a)
  await page.getByText('1024 × 1024 px').first().waitFor()
  await page.getByPlaceholder('e.g. Midwest flooding').fill('Demo: coastal settlement')
  await page.getByPlaceholder('As you want it shown').nth(0).fill('Demo District')
  await shot('02_upload_ready')
  await page.getByRole('button', { name: /Start Disaster Analysis/ }).dblclick()        // a double click must create ONE analysis
  await page.getByText(/Uploading images|Processing/).first().waitFor({ timeout: 10000 })
  await shot('03_processing')
  await page.getByText('Analysis complete').waitFor({ timeout: 240000 })
  const i = (await page.locator('.banner-ok code').textContent()).trim()
  await page.waitForURL(`${UI}/?a=${i}`, { timeout: 15000 })
  return i
})()
const wall = ((Date.now() - t0) / 1000).toFixed(1)
check('analysis ID created and shown', /^[0-9a-f]{32}$/.test(id), id)
check('double click created exactly one analysis', analyzePosts === 1, `POSTs=${analyzePosts}`)
const A = await api(`/api/analysis/${id}?include_polygons=true`)
check('YOLO detected buildings with polygons', A.buildings.length > 0 && A.buildings.every((b) => b.polygon_xy && b.polygon_xy.length >= 3), `${A.buildings.length} buildings`)
check('Siamese produced a three-class prediction for every building', A.buildings.every((b) => b.class_id >= 0 && b.class_id <= 2 && b.predicted_damage in COLOURS))
check('every building has an id, bbox, centre and a real confidence', A.buildings.every((b) => Number.isInteger(b.building_id) && b.bbox_xyxy.length === 4 && b.centre_xy.length === 2 && b.building_confidence > 0 && b.building_confidence <= 1))
check('building ids are unique', new Set(A.buildings.map((b) => b.building_id)).size === A.buildings.length)
check('measured analysis duration is reported', A.parameters.inference_seconds > 0, `${A.parameters.inference_seconds}s model, ${wall}s including upload`)

section('4. Dashboard (Test 4); demo steps 6-7')
await page.locator('.kpi-value').first().waitFor()
await page.waitForTimeout(800)
const st = A.statistics, c = st.damage_counts, total = st.scene.total_buildings
const kv = (await page.locator('.kpi-value').allTextContents()).map((t) => t.trim())
check('KPI total = backend', num(kv[0]) === total, `${kv[0]}`)
check('KPI affected = major+destroyed', num(kv[1]) === c['major-damage'] + c.destroyed && num(kv[1]) === st.damaged_total, `${kv[1]}`)
check('KPI severe = destroyed', num(kv[2]) === c.destroyed)
check('KPI duration = measured inference time', Math.abs(num(kv[3]) - A.parameters.inference_seconds) < 0.06, kv[3])
check('donut centre = affected share', Math.abs(num(await page.locator('.donut-centre b').textContent()) - (st.damaged_total / total) * 100) < 0.06)
const legend = await page.locator('.class-table').first().innerText()
check('class table counts equal the backend counts for all three classes', Object.entries({ 'No Damage': c['no-damage'], 'Major Damage': c['major-damage'], Destroyed: c.destroyed }).every(([k, v]) => new RegExp(`${k}\\s+${v}\\s+[\\d.]+%`).test(legend.replace(/\n/g, ' '))))
check('key insights are shown and derived (affected %, destroyed, quadrant)', (await page.locator('.insights li').count()) >= 4 && (await page.locator('.insights').innerText()).includes(`${st.damaged_total} of ${total}`))
check('before/after are the uploaded images served by the backend', (await page.locator('.compare img').evaluateAll((e) => e.every((i) => i.complete && i.naturalWidth > 0 && i.src.includes('/image/')))))
check('location summary shows only user-typed labels + unavailable notice', (await page.getByText('Demo District').count()) > 0 && (await page.getByText('Geographic location unavailable').count()) > 0)
check('confidence is labelled as an uncalibrated model score', (await page.getByText('uncalibrated').count()) > 0)
await shot('04_dashboard')

section('5. Damage map (Test 3 / Novelty 2); demo steps 8-9')
await page.getByRole('link', { name: 'Damage Map' }).click(); await page.waitForURL(/\/map/)
await page.locator('.mapview svg polygon').first().waitFor()
const polys = page.locator('.mapview svg polygon')
check('one polygon per building, matching the dashboard data', (await polys.count()) === A.buildings.length)
const pointsOK = await polys.evaluateAll((els, exp) => els.every((e, i) => e.getAttribute('points') === exp[i]), A.buildings.map((b) => b.polygon_xy.map(([x, y]) => `${x},${y}`).join(' ')))
check('polygon coordinates are exactly the backend footprints', pointsOK)
check('SVG viewBox equals the image size (correct coordinate transform)', (await page.locator('.mapview svg').first().getAttribute('viewBox')) === `0 0 ${A.image.width} ${A.image.height}`)
const fillsOK = await polys.evaluateAll((els, exp) => els.every((e, i) => e.getAttribute('fill') === exp[i]), A.buildings.map((b) => COLOURS[b.predicted_damage]))
check('every polygon has its three-class colour', fillsOK)
const present = [...new Set(A.buildings.map((b) => b.predicted_damage))]
check('map and donut use identical data (class counts equal)', await (async () => {
  const counts = await polys.evaluateAll((els) => els.reduce((m, e) => ((m[e.getAttribute('fill')] = (m[e.getAttribute('fill')] || 0) + 1), m), {}))
  return Object.entries(c).every(([k, v]) => (counts[COLOURS[k]] || 0) === v)
})(), `classes present: ${present.join(', ')}`)
const lowN = A.buildings.filter((b) => b.building_confidence < st.confidence.low_confidence_threshold).length
const lowDom = await polys.evaluateAll((els) => els.filter((e) => e.getAttribute('data-low-confidence') === 'true').length)
check('low-confidence buildings are highlighted (count equals the backend)', lowDom === lowN, `${lowDom}`)
const destroyed = A.buildings.find((b) => b.predicted_damage === 'destroyed') ?? A.buildings[0]
const idx = A.buildings.indexOf(destroyed)
await polys.nth(idx).hover({ force: true })
const tip = await page.locator('.tooltip').innerText()
check('hover shows id, class and model confidence', tip.includes(`#${destroyed.building_id}`) && tip.includes(destroyed.building_confidence.toFixed(3)), tip.replace(/\n/g, ' | '))
await polys.nth(idx).click({ force: true })
const details = await page.locator('.details').innerText()
check('selecting a building opens its details (class, confidence, probabilities)', details.includes(`#${destroyed.building_id}`) && details.includes(destroyed.building_confidence.toFixed(3)) && (await page.locator('.probs li').count()) === 5)
check('confidence wording is not accuracy', details.includes('not the probability that this prediction is correct'))
const mapEl = page.locator('.mapview'); const mb = await mapEl.boundingBox()
const s0 = await mapEl.locator('> div').first().getAttribute('style')
await page.mouse.move(mb.x + mb.width / 2, mb.y + mb.height / 2); await page.mouse.wheel(0, -500); await page.waitForTimeout(300)
check('zoom works', (await mapEl.locator('> div').first().getAttribute('style')) !== s0)
await page.mouse.move(mb.x + 200, mb.y + 200); await page.mouse.down(); await page.mouse.move(mb.x + 120, mb.y + 160, { steps: 5 }); await page.mouse.up()
check('pan works while zoomed', (await mapEl.locator('> div').first().getAttribute('style')).includes('translate('))
await page.getByRole('button', { name: 'Reset view' }).click()
const total0 = await polys.count()
await page.locator('.map-bar .chip', { hasText: 'No Damage' }).click()
check('class filter hides only that class', (await polys.count()) === total0 - c['no-damage'])
await page.locator('.map-bar .chip', { hasText: 'No Damage' }).click()

section('6. Spatial questions (Test 5 / PS-04); demo steps 10-11')
const asked = [
  ['Count the destroyed buildings in the image.', 'count'],
  ['Count the damaged buildings (major or destroyed) in the image.', 'count'],
  ['Which quadrant has the most damaged buildings?', 'quadrant'],
  ['Is the left half of the image more damaged than the right half?', 'halves'],
  ['Which building in the image is a hospital?', 'cbd'],
]
for (const [q] of asked) {
  await page.locator('.ask input').fill(q)
  await page.getByRole('button', { name: /^Ask$/ }).click()
  await page.locator('.answer-main b').waitFor()
  await page.waitForFunction((t) => document.querySelector('.answer-main b')?.textContent !== t, '__never__', { timeout: 100 }).catch(() => {})
  await page.waitForTimeout(500)
  const ui = (await page.locator('.answer-main b').innerText()).trim()
  const got = await api('/api/questions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ analysis_id: id, questions: [{ question: q }] }) })
  check(`answer equals the spatial engine: "${q.slice(0, 48)}…" -> ${ui}`, String(got.results[0].official.answer) === ui)
}
check('type-4 question shows exactly "cannot be determined" and no evidence region', (await page.locator('.answer-main b').innerText()).trim() === 'cannot be determined' && (await page.locator('.mapview svg rect').count()) === 0)
await page.locator('.ask input').fill('Count the destroyed buildings in the image.')
await page.getByRole('button', { name: /^Ask$/ }).click(); await page.waitForTimeout(800)
check('evidence region is drawn and the supporting buildings are emphasised', (await page.locator('.mapview svg rect').count()) === 1)
await page.locator('details summary', { hasText: 'Official record' }).click()
const rec = JSON.parse(await page.locator('.answer pre').innerText())
check('official record has exactly id, answer, region', JSON.stringify(Object.keys(rec)) === '["id","answer","region"]', JSON.stringify(rec))
const ex = await api(`/api/analysis/${id}/export`)
check('export contains only official fields', ex.length >= 5 && ex.every((r) => JSON.stringify(Object.keys(r)) === '["id","answer","region"]'))
await shot('05_map_question')
const cited = (await api(`/api/analysis/${id}/questions`)).questions.find((q) => q.supporting_building_ids.length)
if (cited) {
  const cb = A.buildings.findIndex((b) => b.building_id === cited.supporting_building_ids[0])
  await page.locator('.mapview svg polygon').nth(cb).click({ force: true })
  if ((await page.locator('.details h3').count()) === 0) await page.locator('.mapview svg polygon').nth(cb).click({ force: true })   // click toggles selection
  check('inspector shows which questions cite the building (evidence reference)', (await page.locator('.details').innerText()).includes('Cited by'))
}

section('7. Analytics (Test 4)')
await page.getByRole('link', { name: 'Analytics' }).first().click(); await page.waitForURL(/\/analytics/); await page.waitForTimeout(700)
check('confidence distribution + other charts render', (await page.locator('.recharts-surface').count()) >= 5 && (await page.getByText('Confidence distribution').count()) > 0)
await shot('06_analytics')

section('8. PDF report (Test 6 / Novelty 3); demo steps 12-17')
await page.getByRole('link', { name: 'Reports' }).first().click(); await page.waitForURL(/\/reports/)
check('report page shows the analysis id and no report yet', (await page.getByText(id).count()) > 0 && (await page.getByTestId('report-status').innerText()).includes('No report'))
let sawProgress = false, stages = new Set()
await page.getByRole('button', { name: 'Generate Disaster Assessment Report' }).click()
const end = Date.now() + 120000
while (Date.now() < end) {
  const status = await page.getByTestId('report-status').innerText()
  if (/Generating|Queued|Starting/.test(status)) { sawProgress = true; const s = await page.locator('.progress-label b').first().innerText().catch(() => ''); if (s) stages.add(s) }
  if (/Complete|Failed/.test(status)) break
  await page.waitForTimeout(150)
}
check('generation progress and stages were displayed from the backend', sawProgress, `${stages.size} stage(s) seen`)
check('status reaches Complete', (await page.getByTestId('report-status').innerText()).includes('Complete'))
const reportId = (await page.locator('.kv dd.mono').nth(1).innerText()).trim()
check('report id shown', /^[0-9a-f]{32}$/.test(reportId))
const meta = await api(`/api/reports/${reportId}`)
check('backend confirms the report: complete, bound to this analysis, with pages and a hash', meta.status === 'complete' && meta.analysis_id === id && meta.pages >= 8 && /^[0-9a-f]{64}$/.test(meta.sha256), `${meta.pages} pages, ${(meta.bytes / 1024).toFixed(0)} KB, ${meta.generation_seconds}s`)
await page.locator('iframe.pdf-frame').waitFor()
check('PDF preview points at the backend report', (await page.locator('iframe.pdf-frame').getAttribute('src')).includes(`/api/reports/${reportId}/download`))
const [dl] = await Promise.all([page.waitForEvent('download', { timeout: 60000 }), page.getByRole('link', { name: 'Download PDF' }).click()])
const pdfPath = path.join(TMP, 'downloaded_report.pdf'); await dl.saveAs(pdfPath)
const pdfBytes = fs.readFileSync(pdfPath)
check('downloaded file is a valid PDF of the expected size', pdfBytes.subarray(0, 5).toString() === '%PDF-' && pdfBytes.length === meta.bytes && /^eagle-dar-report-[0-9a-f]{8}-[0-9a-f]{8}\.pdf$/.test(dl.suggestedFilename()), dl.suggestedFilename())
const pageCount = (pdfBytes.toString('latin1').match(/\/Type\s*\/Page[^s]/g) || []).length
check('PDF page count equals the backend metadata', pageCount === meta.pages, `${pageCount}`)
await shot('07_report_page')
await page.getByRole('button', { name: 'Regenerate Report' }).click()
await page.waitForFunction(() => /Generating|Queued|Starting/.test(document.querySelector('[data-testid=report-status]')?.textContent || ''), null, { timeout: 8000 }).catch(() => {})
await page.getByTestId('report-status').filter({ hasText: 'Complete' }).waitFor({ timeout: 120000 })
const hist = await api(`/api/analysis/${id}/reports`)
check('regenerating creates a NEW report and keeps the old one', hist.length >= 2 && new Set(hist.map((h) => h.report_id)).size === hist.length && hist.some((h) => h.report_id === reportId))
check('history table lists both reports', (await page.locator('.tbl tbody tr').count()) >= 2)
await page.goto(UI + '/')
await page.getByRole('link', { name: 'Download PDF Report' }).waitFor({ timeout: 15000 })
check('dashboard Action Center offers the PDF only after the backend completed it', true)

section('9. No duplicate inference (navigation and refresh)')
for (const p of ['/results', '/map', '/analytics', '/reports', '/settings', '/about', '/']) { await page.goto(UI + p); await page.waitForTimeout(250) }
await page.reload(); await page.waitForTimeout(500)
check('navigating and refreshing never re-ran inference', analyzePosts === 1, `analyze POSTs in this session: ${analyzePosts}`)
const sameAfter = await api(`/api/analysis/${id}`)
check('the same analysis id and predictions are returned after refresh', sameAfter.analysis_id === id && JSON.stringify(sameAfter.statistics.damage_counts) === JSON.stringify(c))

section('10. Failure handling (Test 7)')
// report failure reported by the backend
const f = await ctx.newPage()
await f.addInitScript((i) => localStorage.setItem('eagle-dar.currentAnalysis', i), id)
await f.route(/\/api\/reports\/[0-9a-f]{32}$/, (r) => r.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ...meta, status: 'failed', stage: 'Failed', progress: 0.5, error: 'PDF generation failed (RuntimeError). The analysis is unaffected; try again. See server logs.', download_url: null }) }))
await f.goto(UI + '/reports')
await f.getByRole('button', { name: /Generate|Regenerate/ }).first().click()
await f.getByText('PDF generation failed (RuntimeError)').first().waitFor({ timeout: 15000 })
check('a failed PDF generation shows a clear error and a retry', (await f.getByRole('button', { name: 'Retry' }).count()) === 1 && (await f.getByText('analysis is unaffected').count()) > 0)
await f.close()
// backend unavailable
const off = await ctx.newPage()
await off.route('http://127.0.0.1:8000/**', (r) => r.abort())
await off.goto(UI + '/')
await off.getByText('Backend Offline').waitFor({ timeout: 15000 })
check('backend unavailable: shown honestly, with retry', (await off.getByText(/network error/i).count()) > 0 && (await off.getByRole('button', { name: 'Retry' }).count()) === 1)
await off.goto(UI + '/upload'); check('analysis disabled while the backend is offline', await off.getByRole('button', { name: /Start Disaster Analysis/ }).isDisabled())
await off.close()
// unknown analysis
const nf = await ctx.newPage()
await nf.addInitScript(() => localStorage.setItem('eagle-dar.currentAnalysis', '0'.repeat(32)))
await nf.goto(UI + '/'); await nf.getByText(/analysis not found/i).waitFor({ timeout: 10000 })
await nf.getByRole('button', { name: 'Clear selection' }).click(); await nf.getByText('No analysis yet').first().waitFor()
check('unknown analysis id handled; selection can be cleared', true)
await nf.close()
// no buildings detected
const z = await ctx.newPage()
const zid = await upload(z, 'pb82851c2', { name: 'No-building scene' })
const Z = await api(`/api/analysis/${zid}?include_polygons=true`)
check('image with no detections returns an analysis with zero buildings', Z.statistics.scene.total_buildings === 0 && Z.buildings.length === 0)
check('dashboard says no buildings were detected (no invented results)', (await z.getByText(/No buildings were detected/).count()) > 0 && (await z.locator('.kpi-value').first().innerText()).trim() === '0')
await z.goto(UI + '/reports'); await z.getByRole('button', { name: 'Generate Disaster Assessment Report' }).click()
await z.getByTestId('report-status').filter({ hasText: 'Complete' }).waitFor({ timeout: 120000 })
check('a report is still produced when no buildings were detected', true)
await z.goto(UI + '/map'); check('empty map renders without errors', (await z.locator('.mapview svg polygon').count()) === 0)
await z.close()

section('11. Responsive layouts')
for (const [name, vp] of [['tablet', { width: 820, height: 1100 }], ['phone', { width: 390, height: 844 }]]) {
  const m = await browser.newContext({ viewport: vp })
  const mp = await m.newPage()
  await mp.addInitScript((i) => localStorage.setItem('eagle-dar.currentAnalysis', i), id)
  await mp.goto(UI + '/'); await mp.locator('.kpi-value').first().waitFor({ timeout: 15000 })
  check(`${name}: no horizontal overflow`, !(await mp.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 2)))
  if (name === 'phone') { await mp.getByRole('button', { name: 'Open navigation' }).click(); await mp.waitForTimeout(500); check('phone: navigation drawer opens', (await mp.locator('.sidebar.open').count()) === 1) }
  await mp.screenshot({ path: path.join(SHOTS, `08_${name}.png`) })
  await m.close()
}

console.log('\nBrowser console/page errors:', problems.length ? '\n  ' + [...new Set(problems)].join('\n  ') : 'none')
console.log(`\n${passed} passed, ${failed} failed${problems.length ? ', browser errors present' : ''}`)
await browser.close()
process.exit(failed || problems.length ? 1 : 0)
