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

say("loading numpy, scipy, micropip…");
await pyodide.loadPackage(["micropip", "numpy", "scipy"]);

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

check("anova runs", () => {
  const { anova } = call("run_anova", "/work/upload.xlsx", "[]");
  if (anova.n_observations !== 15) throw new Error(`n=${anova.n_observations}`);
  if (!anova.rows.length) throw new Error("no anova rows");
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
