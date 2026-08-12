// Every design type, end to end, through the real app and real WebAssembly.
//
//   uv run python scripts/build_web_app.py -o docs/_build/web
//   cd web && npm ci && node test-fuzz.mjs ../docs/_build/web [rounds] [seed]
//
// test-wasm.mjs drives bootstrap.py directly; test-dom.mjs drives app.js
// against stubs. Neither can catch the two halves disagreeing — a field the
// page reads that Python never sends, an option the form spells differently
// from the template that consumes it, a design whose own workbook will not fit.
// This runs the loop a user runs, for every template the page offers, with
// randomized-but-valid inputs: fill the form, generate, take the workbook out
// of the virtual filesystem, write responses into it, drop it back on step 2,
// and check what the page then shows.
//
// Randomization is seeded and the seed is printed, so a failure replays with
//   node test-fuzz.mjs ../docs/_build/web 3 <seed>

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { loadPyodide } from "pyodide";

import { installFakeDom, loadApp } from "./fake-dom.mjs";

const APP = path.resolve(process.argv[2] ?? "../docs/_build/web");
const ROUNDS = Number(process.argv[3] ?? 2);
const SEED = Number(process.argv[4] ?? 12345);

const say = (m) => console.log(m);
const clean = (e) =>
  String(e?.stack ?? e?.message ?? e)
    .split("\n")
    .filter((l) => l.length < 300)
    .slice(0, 8)
    .join("\n");

for (const evt of ["uncaughtException", "unhandledRejection"]) {
  process.on(evt, (e) => {
    console.error(`FATAL:\n${clean(e)}`);
    process.exit(1);
  });
}

/** Deterministic PRNG, so a failing round can be replayed from its seed. */
function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── serve the built bundle, boot Pyodide, install the wheel ─────────

const wheelName = JSON.parse(fs.readFileSync(`${APP}/wheels/manifest.json`, "utf8")).wheel;
const MIME = { ".whl": "application/octet-stream", ".json": "application/json" };
const server = http.createServer((req, res) => {
  const rel = decodeURIComponent(new URL(req.url, "http://x").pathname).replace(/^\/+/, "");
  const file = path.join(APP, rel);
  if (!file.startsWith(APP) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    res.writeHead(404).end("not found");
    return;
  }
  res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "text/plain" });
  fs.createReadStream(file).pipe(res);
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const BASE = `http://127.0.0.1:${server.address().port}`;

say(`booting pyodide (seed ${SEED}, ${ROUNDS} round(s) per template)…`);
const pyodide = await loadPyodide();
await pyodide.loadPackage(["micropip", "numpy", "scipy", "sympy"]);
pyodide.globals.set("_wheel_url", `${BASE}/wheels/${wheelName}`);
await pyodide.runPythonAsync(`
import micropip
await micropip.install("openpyxl")
await micropip.install(_wheel_url, deps=False)
`);
pyodide.globals.delete("_wheel_url");
pyodide.FS.mkdirTree("/work");
await pyodide.runPythonAsync(fs.readFileSync(`${APP}/bootstrap.py`, "utf8"));

// ── the page, driven for real ───────────────────────────────────────

const { nodes, missing } = installFakeDom();
const app = await loadApp();
app.state.pyodide = pyodide;
app.loadTemplates(); // the real catalogue, straight out of bootstrap.py

const TEMPLATES = [...app.state.templates.values()];
say(`${TEMPLATES.length} templates offered\n`);

/** Fill step 1's form the way a user would, with valid random choices. */
function fillForm(t, random) {
  nodes["template"].value = t.name;
  app.selectTemplate();

  const min = t.min ?? 1;
  // Capped: a mixture design at 8 factors has 92 parameters, and searching for
  // one is minutes of WebAssembly for no extra coverage of the *page*.
  const max = Math.min(t.max ?? 4, 4);
  const n_factors = min >= max ? min : min + Math.floor(random() * (max - min + 1));
  const style = t.factors ?? "bounds";
  const NAMES = ["T", "P", "F", "c", "d", "e", "g", "h", "j", "k", "m", "q"];

  app.state.factorRows = Array.from({ length: n_factors }, (_, i) => {
    const name = NAMES[i];
    if (style === "levels") {
      // A Latin-square design of order k needs k levels per factor, and the
      // family needs enough mutually orthogonal squares to exist at that
      // order — which is why the factor count sets it.
      const k = n_factors;
      return { name, levels: Array.from({ length: k }, (_, j) => `${name}${j + 1}`).join(",") };
    }
    if (style === "levels2") return { name, low: "lo", high: "hi" };
    // Bounds worth trying: a unit box, a wide one, and one far from zero.
    const lo = [0, -5, 300, 0.001][Math.floor(random() * 4)];
    const span = [1, 10, 200, 0.5][Math.floor(random() * 4)];
    return { name, low: lo, high: lo + span };
  });

  if (t.model_editor) {
    // Two models, both nonlinear in their parameters, over the first factor.
    const x = app.state.factorRows[0].name;
    const choice = random() < 0.5;
    document.getElementById("expression").value = choice
      ? `k0 * exp(-Ea / (8.314 * ${x}))`
      : `A * ${x} / (B + ${x})`;
    app.state.paramRows = choice
      ? [
          { name: "k0", value: 2.0 },
          { name: "Ea", value: 5000.0 },
        ]
      : [
          { name: "A", value: 1.5 },
          { name: "B", value: 0.4 },
        ];
  }
  // Options: legal values only — this is a fuzz over designs, not over input
  // validation, which test-dom.mjs covers.
  for (const el of document.getElementById("opts").querySelectorAll("[data-opt-key]")) {
    const key = el.dataset.optKey;
    if (key === "n") el.value = 6 + Math.floor(random() * 6);
    else if (key === "center_points") {
      // Centre points are the midpoint of the factor box, which a design made
      // of text levels has no way to express; the CLI rejects the pair.
      el.value = style === "levels2" ? 0 : Math.floor(random() * 3);
    } else if (key === "replicates") el.value = 1 + Math.floor(random() * 2);
    else if (key === "degree") el.value = 1 + Math.floor(random() * 3);
    else if (key === "seed") el.value = Math.floor(random() * 1000);
    else if (key === "response") el.value = "y";
    else if (key === "outside_bounds") el.checked = random() < 0.5;
  }
}

/**
 * Fill the response column of the workbook in the virtual filesystem, then
 * hand it back as the File a drop would deliver.
 */
function fillAndPackage(filePath, random) {
  const noise = random() < 0.5 ? 0.0 : 0.05;
  pyodide.globals.set("_fuzz_path", filePath);
  pyodide.globals.set("_fuzz_noise", noise);
  pyodide.globals.set("_fuzz_seed", Math.floor(random() * 1e6));
  pyodide.runPython(`
import random as _r
import openpyxl

book = openpyxl.load_workbook(_fuzz_path)
sheet = book["runs"]
head = [c.value for c in sheet[1]]
# The response is whatever column the campaign named; everything else in the
# row is either metadata or a factor.
meta = {"run_id", "batch", "replicate", "measured_at", "notes"}
from discopt.doe.workbook import Workbook
response = Workbook.open(_fuzz_path).response_name()
rcol = head.index(response) + 1
factors = [h for h in head if h not in meta and h != response]
rand = _r.Random(_fuzz_seed)
for row in sheet.iter_rows(min_row=2):
    vals = dict(zip(head, [c.value for c in row]))
    if vals.get("run_id") is None:
        continue
    # A linear-ish response in whatever the factors are, numeric or not, so
    # every template gets data with real structure in it.
    total = 1.0
    for i, f in enumerate(factors):
        v = vals.get(f)
        try:
            total += (i + 1) * float(v)
        except (TypeError, ValueError):
            total += (i + 1) * (hash(str(v)) % 7)
    if _fuzz_noise:
        total += rand.gauss(0.0, _fuzz_noise)
    row[rcol - 1].value = float(total)
book.save(_fuzz_path)
`);
  pyodide.globals.delete("_fuzz_path");
  pyodide.globals.delete("_fuzz_noise");
  pyodide.globals.delete("_fuzz_seed");

  const bytes = pyodide.FS.readFile(filePath);
  return {
    name: path.basename(filePath).replace(/\.xlsx$/, "") + "-campaign.xlsx",
    size: bytes.length,
    arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length),
  };
}

// ── run it ──────────────────────────────────────────────────────────

let failures = 0;
const report = (label, problems, detail = "") => {
  if (problems.length) {
    failures++;
    say(`  FAIL  ${label}\n        ${problems.join("\n        ")}${detail}`);
  } else {
    say(`  PASS  ${label}`);
  }
};

const COMBINATORIAL = ["latin-square", "graeco-latin", "hyper-graeco-latin", "factorial-2level"];

for (const t of TEMPLATES) {
  for (let round = 0; round < ROUNDS; round++) {
    const seed = SEED + round * 1000 + t.name.length;
    const random = rng(seed);
    const label = `${t.name} (round ${round + 1}, seed ${seed})`;
    const problems = [];

    // Step 1: fill the form and generate.
    fillForm(t, random);
    const factors = app.state.factorRows.map((r) => r.name).join(",");
    try {
      await app.generate();
    } catch (e) {
      report(label, [`generate() threw: ${clean(e)}`]);
      continue;
    }
    if (!nodes["design-error"].hidden) {
      report(label, [`design failed with ${factors}: ${nodes["design-error"].visibleText}`]);
      continue;
    }
    if (nodes["design-result"].hidden) {
      report(label, ["no design was shown"]);
      continue;
    }
    // A design with no runs in it is not a design.
    if (!/\b\d+\b/.test(nodes["design-summary"].visibleText)) {
      problems.push(`no run count in the summary: ${nodes["design-summary"].visibleText}`);
    }

    // Step 2: fill the workbook in and drop it back.
    const file = fillAndPackage("/work/design.xlsx", random);
    await app.acceptFile(file);

    const error = nodes["analyze-error"].visibleText;
    const shown = nodes["analyze-result"].visibleText;
    if (error) problems.push(`analysis reported: ${error.split("\n")[0]}`);
    if (!shown.includes("Campaign")) problems.push("no campaign table");
    if (!nodes["analyze-status"].hidden) problems.push("the progress line was left showing");
    if (app.state.analyzing) problems.push("the page was left marked busy");

    // Whichever analysis applies to this design has to be the one shown.
    if (COMBINATORIAL.includes(t.name)) {
      if (!shown.includes("ANOVA over factor levels")) problems.push("no factor-level ANOVA");
    } else {
      if (!shown.includes("Fitted coefficients")) problems.push("no fitted coefficients");
      if (!shown.includes("Regression ANOVA")) problems.push("no regression ANOVA");
      // Every model design writes its equation out.
      if (!nodes["analyze-result"].markup.includes("equation") && !shown.includes(" = ")) {
        problems.push("no model equation");
      }
    }
    // Uploading fills step 1 in from the workbook.
    if (!nodes["design-source"].visibleText.includes("Filled in from")) {
      problems.push(`step 1 was not filled in: ${nodes["design-source"].visibleText}`);
    }
    if (nodes["template"].value !== t.name) {
      problems.push(`step 1 switched to ${nodes["template"].value}`);
    }

    report(label, problems, problems.length ? `\n        factors: ${factors}` : "");
  }
}

if (missing.length) {
  failures++;
  say(`  FAIL  app.js reached for ids index.html lacks: ${[...new Set(missing)].join(", ")}`);
}

app.cleanup();
server.close();
say(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
