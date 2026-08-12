// End-to-end test of the browser app's Python half, in real WebAssembly.
//
// Runs the exact bootstrap sequence app.js uses — same Pyodide version, same
// wheel, same `deps=False` install, same bootstrap.py — against a locally
// served copy of the built bundle. What it is really guarding is that the
// package still imports and works with *no* base `discopt` and *no* jax, which
// no other test can check honestly: on a developer machine and in normal CI
// both are installed.
//
//   uv run python scripts/build_web_app.py -o docs/_build/web
//   cd web && npm ci && node test-wasm.mjs ../docs/_build/web
//
// Requires network access on first run (Pyodide fetches numpy/scipy wheels from
// the CDN and openpyxl from PyPI; both are then cached in node_modules).

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { loadPyodide } from "pyodide";

const APP = path.resolve(process.argv[2] ?? "../docs/_build/web");
const wheelName = JSON.parse(fs.readFileSync(`${APP}/wheels/manifest.json`, "utf8")).wheel;

const say = (m) => console.log(m);
// Pyodide errors carry the whole asm bundle in .stack; keep only what reads.
const clean = (e) =>
  String(e?.message ?? e)
    .split("\n")
    .filter((l) => l.length < 300)
    .slice(0, 12)
    .join("\n");

for (const evt of ["uncaughtException", "unhandledRejection"]) {
  process.on(evt, (e) => {
    console.error(`FATAL:\n${clean(e)}`);
    process.exit(1);
  });
}

// Serve the bundle over HTTP, the way the browser fetches it.
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

say("booting pyodide…");
const pyodide = await loadPyodide();
say(`  ${(await pyodide.runPythonAsync("import sys; sys.version.split()[0]"))}`);

say("loading numpy, scipy, sympy, micropip…");
await pyodide.loadPackage(["micropip", "numpy", "scipy", "sympy"]);

say(`installing openpyxl + ${wheelName} (deps=False)…`);
pyodide.globals.set("_wheel_url", `${BASE}/wheels/${wheelName}`);
await pyodide.runPythonAsync(`
import micropip
await micropip.install("openpyxl")
await micropip.install(_wheel_url, deps=False)
`);
pyodide.globals.delete("_wheel_url");

say("running bootstrap.py…");
pyodide.FS.mkdirTree("/work");
await pyodide.runPythonAsync(fs.readFileSync(`${APP}/bootstrap.py`, "utf8"));

/** Call a bootstrap.py entry point and unwrap its JSON. */
const call = (fn, ...args) => {
  const py = pyodide.globals.get(fn);
  const raw = py(...args);
  py.destroy();
  const out = JSON.parse(raw);
  if (!out.ok) throw new Error(`${fn}: ${out.error}\n${out.detail ?? ""}`);
  return out;
};

let failures = 0;
const check = (label, fn) => {
  try {
    fn();
    say(`  PASS  ${label}`);
  } catch (e) {
    failures++;
    say(`  FAIL  ${label}\n        ${clean(e).split("\n").slice(0, 6).join("\n        ")}`);
  }
};

say("\n── environment ──");
const env = call("environment");
say(`  discopt-doe ${env.discopt_doe} · python ${env.python} · numpy ${env.numpy} · scipy ${env.scipy}`);

say("\n── the heavy dependencies really are absent ──");
check("jax is not importable", () => {
  const r = pyodide.runPython(
    'import importlib.util\n"present" if importlib.util.find_spec("jax") else "absent"',
  );
  if (r !== "absent") throw new Error(`jax reported ${r}`);
});
check("discopt.estimate is not importable", () => {
  const r = pyodide.runPython(`
import importlib.util
try:
    found = importlib.util.find_spec("discopt.estimate") is not None
except Exception:
    found = False
"present" if found else "absent"
`);
  if (r !== "absent") throw new Error(`discopt.estimate reported ${r}`);
});

say("\n── template catalogue ──");
const { groups } = call("describe_templates");
const names = groups.flatMap((g) => g.templates.map((t) => t.name));
check(`${names.length} templates offered`, () => {
  for (const need of [
    "symbolic",
    "latin-hypercube",
    "central-composite",
    "box-behnken",
    "factorial-2level",
    "response-surface-2d",
  ]) {
    if (!names.includes(need)) throw new Error(`missing ${need}`);
  }
});

say("\n── design generation ──");
const bounds = (...ns) => ns.map(([name, low, high]) => ({ name, low, high }));
const CASES = [
  [
    "box-behnken",
    { factors: bounds(["T", 300, 400], ["P", 1, 5], ["F", 0.1, 2]), center_points: 3,
      response: "yield" },
    15,
  ],
  [
    "central-composite",
    { factors: bounds(["a", 0, 1], ["b", 0, 1]), center_points: 4, alpha: "rotatable" },
    12,
  ],
  ["latin-hypercube", { factors: bounds(["x", 0, 10], ["z", 0, 10]), n: 12, basis: "linear" }, 12],
  [
    "response-surface-2d",
    { factors: bounds(["a", 0, 1], ["b", 0, 1]), n: 8, criterion: "determinant" },
    8,
  ],
  [
    "factorial-2level",
    {
      factors: [
        { name: "A", low: "lo", high: "hi" },
        { name: "B", low: "lo", high: "hi" },
        { name: "C", low: "lo", high: "hi" },
      ],
      replicates: 1,
    },
    8,
  ],
  [
    "latin-square",
    {
      factors: [
        { name: "row", levels: "r1,r2,r3" },
        { name: "col", levels: "c1,c2,c3" },
        { name: "treat", levels: "t1,t2,t3" },
      ],
      replicates: 1,
    },
    9,
  ],
];

for (const [template, spec, expected] of CASES) {
  check(`create ${template} (${expected} runs)`, () => {
    const out = call(
      "create_design",
      JSON.stringify({ template, response: "resp", seed: 0, ...spec }),
    );
    if (out.new_run_ids.length !== expected) {
      throw new Error(`got ${out.new_run_ids.length} runs, expected ${expected}`);
    }
    const bytes = pyodide.FS.readFile(out.file_path);
    if (bytes[0] !== 0x50 || bytes[1] !== 0x4b) throw new Error("output is not a .xlsx (zip)");
  });
}

say("\n── user-defined models (sympy) ──");

const ARRHENIUS = {
  template: "symbolic",
  expression: "k0 * exp(-Ea / (8.314 * T))",
  parameters: [
    { name: "k0", value: 2.0 },
    { name: "Ea", value: 5000.0 },
  ],
  factors: bounds(["T", 300, 500]),
  n: 6,
  response: "rate",
  error: 0.05,
  seed: 0,
};

check("check_model reports the symbolic derivatives", () => {
  const out = call("check_model", JSON.stringify(ARRHENIUS));
  if (out.parameters.join(",") !== "k0,Ea") throw new Error(`params ${out.parameters}`);
  if (out.derivatives.length !== 2) throw new Error("expected two derivatives");
  // d/dk0 of k0*exp(...) drops k0 entirely; d/dEa keeps it.
  const byName = Object.fromEntries(out.derivatives.map((d) => [d.parameter, d.expression]));
  if (byName.k0.includes("k0")) throw new Error(`dy/dk0 still mentions k0: ${byName.k0}`);
  if (!byName.Ea.includes("k0")) throw new Error(`dy/dEa lost k0: ${byName.Ea}`);
});

check("a bad expression is a message, not a crash", () => {
  const py = pyodide.globals.get("check_model");
  const out = JSON.parse(py(JSON.stringify({ ...ARRHENIUS, expression: "k0 * bogus" })));
  py.destroy();
  if (out.ok) throw new Error("expected a parse failure");
  if (!/unknown name 'bogus'/.test(out.error)) throw new Error(out.error);
});

check("a hostile expression is refused without executing", () => {
  const py = pyodide.globals.get("check_model");
  const out = JSON.parse(
    py(JSON.stringify({ ...ARRHENIUS, expression: "__import__('js').fetch('/pwned')" })),
  );
  py.destroy();
  if (out.ok) throw new Error("payload was accepted");
});

check("designs a nonlinear model at its support points", () => {
  const out = call("create_design", JSON.stringify(ARRHENIUS));
  if (out.new_run_ids.length !== 6) throw new Error(`got ${out.new_run_ids.length} runs`);
  if (out.parameter_names.join(",") !== "k0,Ea") throw new Error(`params ${out.parameter_names}`);
  // A 2-parameter D-optimal design collapses onto 2 support points.
  const temps = [...new Set(out.designs.map((d) => Math.round(d.T * 1e6) / 1e6))];
  if (temps.length !== 2) throw new Error(`expected 2 support points, got ${temps}`);
});

check("fits a nonlinear model from a distant starting guess", () => {
  // Truth is well away from the nominal guess the design was centred on.
  pyodide.runPython(`
import math, shutil, openpyxl
shutil.copy("/work/design.xlsx", "/work/upload.xlsx")
TRUE_K0, TRUE_EA = 3.7, 6200.0
book = openpyxl.load_workbook("/work/upload.xlsx")
sheet = book["runs"]
head = [c.value for c in sheet[1]]
col = head.index("rate") + 1
for row in sheet.iter_rows(min_row=2):
    vals = dict(zip(head, [c.value for c in row]))
    if vals.get("run_id") is None:
        continue
    T = float(vals["T"])
    row[col - 1].value = TRUE_K0 * math.exp(-TRUE_EA / (8.314 * T))
book.save("/work/upload.xlsx")
`);
  const { fit } = call("run_fit", "/work/upload.xlsx");
  const got = Object.fromEntries(fit.parameters.map((p) => [p.name, p.estimate]));
  if (Math.abs(got.k0 - 3.7) > 1e-4) throw new Error(`k0 = ${got.k0}, want 3.7`);
  if (Math.abs(got.Ea - 6200.0) > 1e-1) throw new Error(`Ea = ${got.Ea}, want 6200`);
  if (!fit.converged) throw new Error("solver did not converge");
});

say("\n── round trip: design → fill → fit → anova ──");
call(
  "create_design",
  JSON.stringify({
    template: "box-behnken",
    factors: bounds(["T", 300, 400], ["P", 1, 5], ["F", 0.1, 2]),
    center_points: 3,
    response: "yield",
    seed: 0,
  }),
);

// Fill the response column from a known quadratic. Box-Behnken is saturated for
// a 3-factor quadratic, so a noise-free fit must recover the coefficients exactly.
const TRUTH = { b0: 5.0, b1: 0.02, b2: -0.5, b3: 1.5, b11: -1e-4, b22: 0.1, b33: -0.3,
                b12: 1e-3, b13: 2e-3, b23: -0.05 };
pyodide.runPython(`
import shutil, openpyxl
shutil.copy("/work/design.xlsx", "/work/upload.xlsx")
TRUE = ${JSON.stringify(TRUTH)}
book = openpyxl.load_workbook("/work/upload.xlsx")
sheet = book["runs"]
head = [c.value for c in sheet[1]]
ycol = head.index("yield") + 1
for row in sheet.iter_rows(min_row=2):
    vals = dict(zip(head, [c.value for c in row]))
    if vals.get("run_id") is None:
        continue
    T, P, F = vals["T"], vals["P"], vals["F"]
    row[ycol - 1].value = (
        TRUE["b0"] + TRUE["b1"] * T + TRUE["b2"] * P + TRUE["b3"] * F
        + TRUE["b11"] * T * T + TRUE["b22"] * P * P + TRUE["b33"] * F * F
        + TRUE["b12"] * T * P + TRUE["b13"] * T * F + TRUE["b23"] * P * F
    )
book.save("/work/upload.xlsx")
`);

check("inspect_workbook reports 15 completed runs", () => {
  const out = call("inspect_workbook", "/work/upload.xlsx");
  if (out.status.n_completed !== 15) throw new Error(`n_completed=${out.status.n_completed}`);
  if (out.rows.length !== 15) throw new Error(`rows=${out.rows.length}`);
});

check("the fit describes the model it fitted, term by term", () => {
  const { model } = call("run_fit", "/work/upload.xlsx");
  if (!model) throw new Error("no model summary");
  if (model.response !== "yield") throw new Error(`response ${model.response}`);
  const terms = Object.fromEntries(model.terms.map((t) => [t.parameter, t.powers]));
  // Every coefficient in TRUTH must be described, and described correctly:
  // b0 is the intercept, b1 a main effect, b11 a square, b12 an interaction.
  for (const name of Object.keys(TRUTH)) {
    if (!(name in terms)) throw new Error(`no term for ${name}`);
  }
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  if (!same(terms.b0, {})) throw new Error(`b0 is ${JSON.stringify(terms.b0)}`);
  if (!same(terms.b1, { T: 1 })) throw new Error(`b1 is ${JSON.stringify(terms.b1)}`);
  if (!same(terms.b22, { P: 2 })) throw new Error(`b22 is ${JSON.stringify(terms.b22)}`);
  if (!same(terms.b13, { T: 1, F: 1 })) throw new Error(`b13 is ${JSON.stringify(terms.b13)}`);
});

check("a user-defined model comes back with its fitted values substituted", () => {
  // /work/design.xlsx is the box-behnken campaign by now, so rebuild the
  // Arrhenius one and fit it again — the point is the substituted expression.
  call("create_design", JSON.stringify(ARRHENIUS));
  pyodide.runPython(`
import math, shutil, openpyxl
shutil.copy("/work/design.xlsx", "/work/arrhenius.xlsx")
book = openpyxl.load_workbook("/work/arrhenius.xlsx")
sheet = book["runs"]
head = [c.value for c in sheet[1]]
col = head.index("rate") + 1
for row in sheet.iter_rows(min_row=2):
    vals = dict(zip(head, [c.value for c in row]))
    if vals.get("run_id") is None:
        continue
    row[col - 1].value = 3.7 * math.exp(-6200.0 / (8.314 * float(vals["T"])))
book.save("/work/arrhenius.xlsx")
`);
  const { model } = call("run_fit", "/work/arrhenius.xlsx");
  if (model.terms !== null) throw new Error("a symbolic model has no basis terms");
  if (!/k0/.test(model.expression)) throw new Error(`expression ${model.expression}`);
  // The fitted form must carry numbers where the parameter names were.
  if (/k0|Ea/.test(model.fitted_expression)) {
    throw new Error(`parameters not substituted: ${model.fitted_expression}`);
  }
  if (!/3\.7/.test(model.fitted_expression)) {
    throw new Error(`expected k0 = 3.7 in ${model.fitted_expression}`);
  }
});

check("fit recovers the known quadratic exactly", () => {
  const { fit } = call("run_fit", "/work/upload.xlsx");
  for (const p of fit.parameters) {
    const want = TRUTH[p.name];
    if (want === undefined) throw new Error(`unexpected parameter ${p.name}`);
    if (Math.abs(p.estimate - want) > 1e-6 * Math.max(1, Math.abs(want))) {
      throw new Error(`${p.name}: got ${p.estimate}, want ${want}`);
    }
  }
  if (!(fit.objective < 1e-15)) throw new Error(`residual ${fit.objective} too large`);
});

check("the fit reports a significance test per coefficient", () => {
  const { fit } = call("run_fit", "/work/upload.xlsx");
  if (!fit.coefficients?.length) throw new Error("no coefficient statistics");
  const byName = Object.fromEntries(fit.coefficients.map((c) => [c.name, c]));
  for (const name of Object.keys(TRUTH)) {
    const c = byName[name];
    if (!c) throw new Error(`no statistics for ${name}`);
    for (const key of ["estimate", "std_error", "t_statistic", "p_value"]) {
      if (!(key in c)) throw new Error(`${name} has no ${key}`);
    }
  }
  // Regression / Residual / Total, and a summary with R².
  const sources = fit.regression_anova.map((r) => r.source);
  for (const need of ["Regression", "Residual"]) {
    if (!sources.includes(need)) throw new Error(`no ${need} row in ${sources}`);
  }
  if (!Number.isFinite(fit.summary.R_squared)) throw new Error("no R² in the summary");
});

check("anova runs", () => {
  const { anova } = call("run_anova", "/work/upload.xlsx", "[]");
  if (anova.n_observations !== 15) throw new Error(`n=${anova.n_observations}`);
  if (!anova.rows.length) throw new Error("no anova rows");
});

say("\n── a workbook describes the design that made it ──");

for (const [template, spec, expect] of [
  [
    "central-composite",
    { factors: bounds(["T", 300, 400], ["P", 1, 5]), center_points: 4, alpha: "face" },
    { factors: [{ name: "T", low: 300, high: 400 }], options: { center_points: 4, alpha: "face" } },
  ],
  [
    "latin-square",
    {
      factors: [
        { name: "row", levels: "r1,r2,r3" },
        { name: "col", levels: "c1,c2,c3" },
        { name: "treat", levels: "t1,t2,t3" },
      ],
    },
    { factors: [{ name: "row", levels: "r1, r2, r3" }], options: {} },
  ],
  [
    "factorial-2level",
    {
      factors: [
        { name: "A", low: "lo", high: "hi" },
        { name: "B", low: "lo", high: "hi" },
      ],
      replicates: 2,
    },
    { factors: [{ name: "A", low: "lo", high: "hi" }], options: { replicates: 2 } },
  ],
]) {
  check(`${template} round-trips into the design form`, () => {
    call("create_design", JSON.stringify({ template, response: "resp", seed: 7, ...spec }));
    const out = call("design_spec_from_workbook", "/work/design.xlsx");
    if (!out.supported) throw new Error(`not supported: ${out.reason}`);
    if (out.template !== template) throw new Error(`template ${out.template}`);
    // The first factor comes back in the style this template's editor uses.
    for (const [key, want] of Object.entries(expect.factors[0])) {
      const got = out.factors[0][key];
      if (String(got) !== String(want)) throw new Error(`factor ${key}: ${got} != ${want}`);
    }
    for (const [key, want] of Object.entries(expect.options)) {
      if (String(out.options[key]) !== String(want)) {
        throw new Error(`option ${key}: ${out.options[key]} != ${want}`);
      }
    }
    if (out.options.response !== "resp") throw new Error(`response ${out.options.response}`);
    if (out.options.seed !== 7) throw new Error(`seed ${out.options.seed}`);
  });
}

check("a user-defined model round-trips with its expression and parameters", () => {
  call("create_design", JSON.stringify(ARRHENIUS));
  const out = call("design_spec_from_workbook", "/work/design.xlsx");
  if (out.expression !== ARRHENIUS.expression) throw new Error(`expression ${out.expression}`);
  // Parameter order fixes the FIM layout, so it has to survive the round trip.
  const names = out.parameters.map((p) => p.name).join(",");
  if (names !== "k0,Ea") throw new Error(`parameters ${names}`);
  if (out.parameters[1].value !== 5000) throw new Error(`Ea = ${out.parameters[1].value}`);
});

say("\n── diagnosing a workbook the analysis cannot use ──");

// Response cells the way a spreadsheet leaves them when things go wrong: a
// formula whose result was never cached (what Numbers/Sheets exports produce,
// and what shows a perfectly good number on screen), a value with a unit
// typed after it, and one left blank.
pyodide.runPython(`
import shutil, openpyxl
shutil.copy("/work/upload.xlsx", "/work/broken.xlsx")
book = openpyxl.load_workbook("/work/broken.xlsx")
sheet = book["runs"]
head = [c.value for c in sheet[1]]
ycol = head.index("yield") + 1
rows = [r for r in sheet.iter_rows(min_row=2) if r[0].value is not None]
rows[0][ycol - 1].value = "=1+2"     # formula, no cached result
rows[1][ycol - 1].value = "3.5 g"    # text, not a number
rows[2][ycol - 1].value = None       # not measured yet
book.save("/work/broken.xlsx")
`);

check("diagnose_workbook tells the three failure modes apart", () => {
  const diag = call("diagnose_workbook", "/work/broken.xlsx");
  if (diag.response !== "yield") throw new Error(`response=${diag.response}`);
  const state = (id) => diag.runs.find((r) => r.run_id === id)?.state;
  if (state(1) !== "formula") throw new Error(`run 1 is ${state(1)}, want formula`);
  if (state(2) !== "text") throw new Error(`run 2 is ${state(2)}, want text`);
  if (state(3) !== "empty") throw new Error(`run 3 is ${state(3)}, want empty`);
  if (state(4) !== "ok") throw new Error(`run 4 is ${state(4)}, want ok`);
  const shown = diag.runs.find((r) => r.run_id === 1).shown;
  if (shown !== "=1+2") throw new Error(`run 1 shows ${shown}`);
  if (diag.runs.some((r) => r.bad_inputs.length)) throw new Error("factors reported as bad");
});

check("diagnose_workbook survives a file inspect_workbook refuses", () => {
  // The uncached formula makes all_runs() raise; the diagnosis is what the
  // page falls back to, so it has to work on exactly that file.
  const py = pyodide.globals.get("inspect_workbook");
  const refused = JSON.parse(py("/work/broken.xlsx"));
  py.destroy();
  if (refused.ok) throw new Error("expected inspect_workbook to refuse the formula");
  if (!/formula/.test(refused.error)) throw new Error(`unexpected message: ${refused.error}`);
  if (!call("diagnose_workbook", "/work/broken.xlsx").runs.length) {
    throw new Error("diagnosis came back empty");
  }
});

check("failures come back as JSON, not as thrown exceptions", () => {
  const py = pyodide.globals.get("create_design");
  const out = JSON.parse(
    py(JSON.stringify({ template: "box-behnken", factors: bounds(["a", 0, 1], ["b", 0, 1]) })),
  );
  py.destroy();
  if (out.ok) throw new Error("a 2-factor Box-Behnken should have been rejected");
  if (!/3 to 5 factors/.test(out.error)) throw new Error(`unexpected message: ${out.error}`);
});

server.close();
say(`\n${failures === 0 ? "ALL CHECKS PASSED" : `${failures} CHECK(S) FAILED`}`);
process.exit(failures === 0 ? 0 : 1);
