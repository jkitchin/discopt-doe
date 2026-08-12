// discopt-doe in the browser: boot Pyodide, install the pure-Python wheel,
// and drive discopt.doe through web/bootstrap.py.
//
// The base `discopt` package is deliberately absent — it ships only platform
// wheels and needs jax, jaxlib, and a native solver, none of which build for
// WebAssembly. `discopt.doe` is installed with deps=false for exactly that
// reason, and the classical / linear-FIM paths are what work without them.

const PYODIDE_VERSION = "314.0.3";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// Built into the Pages artifact next to this file (see deploy-book.yml).
// Falls back to PyPI so the page still works if the local wheel is missing —
// though an older published version may predate the browser-safe imports.
const WHEEL_DIR = "wheels/";

const $ = (id) => document.getElementById(id);

const state = {
  pyodide: null,
  templates: new Map(),
  groups: [],
  current: null,
  factorRows: [],
  paramRows: [],
  analyzePath: "/work/upload.xlsx",
  analyzing: false,
};

// ─────────────────────────── boot ───────────────────────────

function setBoot(text, cls = "") {
  $("boot-text").textContent = text;
  $("boot-status").className = `boot ${cls}`;
}

async function findWheel() {
  // The deploy step writes a manifest naming the wheel it built.
  try {
    const res = await fetch(`${WHEEL_DIR}manifest.json`, { cache: "no-cache" });
    if (res.ok) {
      const { wheel } = await res.json();
      if (wheel) return WHEEL_DIR + wheel;
    }
  } catch {
    /* fall through to PyPI */
  }
  return "discopt-doe";
}

async function boot() {
  setBoot(`Loading Python ${PYODIDE_VERSION}…`);
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  setBoot("Loading numpy, scipy and sympy…");
  // sympy differentiates user-defined models; Pyodide ships a wasm build of it,
  // so take that rather than pulling the pure-Python sdist through micropip.
  await pyodide.loadPackage(["micropip", "numpy", "scipy", "sympy"]);

  setBoot("Installing discopt-doe…");
  const wheel = await findWheel();
  // Driven from Python rather than through the PyProxy: a JS object passed to a
  // PyProxy call arrives as a positional argument, so `install(url, {deps:false})`
  // silently keeps dependency resolution on — and then fails trying to find a
  // pure-Python jaxlib. Keyword arguments need `callKwargs`, and this is clearer.
  pyodide.globals.set("_wheel_url", wheel);
  await pyodide.runPythonAsync(`
import micropip
# openpyxl and et-xmlfile are pure Python, so micropip pulls them from PyPI.
await micropip.install("openpyxl")
# deps=False: this wheel's metadata declares discopt, jax, and jaxlib, none of
# which have WebAssembly builds. Everything the browser app touches is already
# satisfied by numpy, scipy, and openpyxl.
await micropip.install(_wheel_url, deps=False)
`);
  pyodide.globals.delete("_wheel_url");

  setBoot("Starting…");
  pyodide.FS.mkdirTree("/work");
  const glue = await (await fetch("bootstrap.py", { cache: "no-cache" })).text();
  await pyodide.runPythonAsync(glue);
  state.pyodide = pyodide;

  const env = call("environment");
  $("env").textContent =
    `discopt-doe ${env.discopt_doe} · Python ${env.python} · ` +
    `numpy ${env.numpy} · scipy ${env.scipy} · Pyodide ${PYODIDE_VERSION}`;

  loadTemplates();
  setBoot("Ready — everything runs locally in this tab.", "done");
  $("main").hidden = false;
}

/** Call a bootstrap.py function; unwrap its JSON, throw on a reported error. */
function call(fn, ...args) {
  const py = state.pyodide.globals.get(fn);
  const raw = py(...args);
  py.destroy();
  const out = JSON.parse(raw);
  if (!out.ok) {
    const err = new Error(out.error);
    err.detail = out.detail;
    throw err;
  }
  return out;
}

// ───────────────────────── templates ─────────────────────────

function loadTemplates() {
  const { groups } = call("describe_templates");
  state.groups = groups;
  const select = $("template");
  select.innerHTML = "";
  for (const group of groups) {
    const og = document.createElement("optgroup");
    og.label = group.label;
    for (const t of group.templates) {
      state.templates.set(t.name, t);
      const opt = document.createElement("option");
      opt.value = t.name;
      opt.textContent = t.name;
      og.appendChild(opt);
    }
    select.appendChild(og);
  }
  select.value = "central-composite";
  selectTemplate();
}

function selectTemplate() {
  const t = state.templates.get($("template").value);
  state.current = t;
  $("template-desc").textContent = t.description;

  $("model-field").hidden = !t.model_editor;
  if (t.model_editor) {
    // Seed with a worked example: an empty expression box is a blank page, and
    // the point is to show what the syntax looks like.
    const ex = t.example ?? {};
    $("expression").value = ex.expression ?? "";
    state.paramRows = (ex.parameters ?? [{ name: "a", value: 1 }]).map((p) => ({ ...p }));
    state.factorRows = (ex.factors ?? [blankFactor(0)]).map((f) => ({ ...f }));
    renderParams();
  } else {
    state.paramRows = [];
    state.factorRows = [];
    const wanted = Math.max(t.min ?? 1, Math.min(t.max ?? 3, t.min === t.max ? t.min : 2));
    for (let i = 0; i < wanted; i++) state.factorRows.push(blankFactor(i));
  }

  renderFactors();
  renderOptions();
  if (t.model_editor) checkModel();
}

// ─────────────────────── model editor ───────────────────────

function renderParams() {
  const body = $("params-body");
  body.innerHTML = "";
  state.paramRows.forEach((row, i) => {
    const tr = document.createElement("tr");
    for (const [key, type, label] of [
      ["name", "text", "Name"],
      ["value", "number", "Nominal value"],
    ]) {
      const td = document.createElement("td");
      const input = document.createElement("input");
      input.type = type;
      if (type === "number") input.step = "any";
      input.value = row[key];
      input.setAttribute("aria-label", `${label} for parameter ${i + 1}`);
      input.addEventListener("input", () => {
        row[key] = type === "number" ? Number(input.value) : input.value;
        scheduleModelCheck();
      });
      td.appendChild(input);
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    if (state.paramRows.length > 1) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ghost";
      btn.textContent = "×";
      btn.title = "Remove this parameter";
      btn.setAttribute("aria-label", `Remove parameter ${i + 1}`);
      btn.addEventListener("click", () => {
        state.paramRows.splice(i, 1);
        renderParams();
        checkModel();
      });
      td.appendChild(btn);
    }
    tr.appendChild(td);
    body.appendChild(tr);
  });
}

let modelCheckTimer = null;

function scheduleModelCheck() {
  clearTimeout(modelCheckTimer);
  modelCheckTimer = setTimeout(checkModel, 300);
}

function checkModel() {
  if (!state.current?.model_editor || !state.pyodide) return;
  const box = $("model-feedback");
  const spec = {
    expression: $("expression").value,
    parameters: state.paramRows,
    factors: state.factorRows,
    response: collectOptions().response ?? "y",
  };
  try {
    const out = call("check_model", JSON.stringify(spec));
    box.className = "feedback ok";
    box.innerHTML = "";
    const head = document.createElement("div");
    head.textContent = `Parsed: ${out.expression}`;
    const derivs = document.createElement("div");
    derivs.className = "derivs";
    derivs.textContent = out.derivatives
      .map((d) => `∂y/∂${d.parameter} = ${d.expression}`)
      .join("\n");
    box.append(head, derivs);
  } catch (err) {
    box.className = "feedback bad";
    box.textContent = err.message;
  }
}

function blankFactor(i) {
  const names = ["T", "P", "F", "c", "d", "e", "g", "h"];
  return { name: names[i] ?? `x${i + 1}`, low: 0, high: 1, levels: "low,mid,high" };
}

const FACTOR_COLUMNS = {
  bounds: [
    ["name", "Name", "text"],
    ["low", "Low", "number"],
    ["high", "High", "number"],
  ],
  levels2: [
    ["name", "Name", "text"],
    ["low", "Low level", "text"],
    ["high", "High level", "text"],
  ],
  levels: [
    ["name", "Name", "text"],
    ["levels", "Levels (comma-separated)", "text"],
  ],
};

function renderFactors() {
  const t = state.current;
  const cols = FACTOR_COLUMNS[t.factors ?? "bounds"];
  const fixed = t.min === t.max;

  $("factors-head").innerHTML =
    cols.map(([, label]) => `<th>${label}</th>`).join("") +
    (fixed ? "" : '<th class="drop-col"></th>');

  const body = $("factors-body");
  body.innerHTML = "";
  state.factorRows.forEach((row, i) => {
    const tr = document.createElement("tr");
    for (const [key, label, type] of cols) {
      const td = document.createElement("td");
      const input = document.createElement("input");
      input.type = type;
      input.value = row[key];
      input.setAttribute("aria-label", `${label} for factor ${i + 1}`);
      if (type === "number") input.step = "any";
      input.addEventListener("input", () => {
        row[key] = type === "number" ? Number(input.value) : input.value;
        // Factor names are symbols in the model expression, so renaming one
        // changes what parses.
        if (state.current?.model_editor) scheduleModelCheck();
      });
      td.appendChild(input);
      tr.appendChild(td);
    }
    if (!fixed) {
      const td = document.createElement("td");
      if (state.factorRows.length > (t.min ?? 1)) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ghost";
        btn.textContent = "×";
        btn.title = "Remove this factor";
        btn.setAttribute("aria-label", `Remove factor ${i + 1}`);
        btn.addEventListener("click", () => {
          state.factorRows.splice(i, 1);
          renderFactors();
        });
        td.appendChild(btn);
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
  });

  const min = t.min ?? 1;
  const max = t.max ?? 8;
  $("factors-hint").textContent =
    min === max
      ? `${t.name} takes exactly ${min} factor${min === 1 ? "" : "s"}.`
      : `${t.name} takes ${min} to ${max} factors.`;
  $("add-factor").hidden = state.factorRows.length >= max;
}

const OPTION_SPECS = {
  n: { label: "Runs", type: "number", value: 8, min: 1 },
  center_points: { label: "Centre points", type: "number", value: 0, min: 0 },
  replicates: { label: "Replicates", type: "number", value: 1, min: 1 },
  degree: { label: "Polynomial degree", type: "number", value: 3, min: 1 },
  mixture_total: { label: "Mixture total", type: "number", value: 1, min: 0 },
  basis: { label: "Fit model", type: "select", value: "linear", options: ["linear", "quadratic"] },
  alpha: {
    label: "Axial distance",
    type: "select",
    value: "rotatable",
    options: ["rotatable", "face"],
  },
  criterion: {
    label: "Criterion",
    type: "select",
    value: "determinant",
    options: ["determinant", "trace", "min_eigenvalue", "condition_number"],
  },
  outside_bounds: { label: "Axial points outside bounds", type: "check", value: false },
};

function renderOptions() {
  const box = $("opts");
  box.innerHTML = "";
  const opts = [...(state.current.options ?? []), "response", "seed"];

  for (const key of opts) {
    const spec =
      OPTION_SPECS[key] ??
      (key === "response"
        ? { label: "Response column", type: "text", value: "y" }
        : { label: "Seed", type: "number", value: 42, min: 0 });

    const div = document.createElement("div");
    div.className = spec.type === "check" ? "opt check" : "opt";
    const id = `opt-${key}`;

    let input;
    if (spec.type === "select") {
      input = document.createElement("select");
      for (const o of spec.options) {
        const opt = document.createElement("option");
        opt.value = opt.textContent = o;
        input.appendChild(opt);
      }
      // Latin hypercubes default to a linear fit; the RSM designs to quadratic.
      input.value =
        key === "basis" && state.current.name !== "latin-hypercube" ? "quadratic" : spec.value;
    } else {
      input = document.createElement("input");
      input.type = spec.type === "check" ? "checkbox" : spec.type;
      if (spec.type === "check") input.checked = spec.value;
      else input.value = spec.value;
      if (spec.min !== undefined) input.min = spec.min;
      if (spec.type === "number") input.step = key === "mixture_total" ? "any" : "1";
    }
    input.id = id;
    input.dataset.optKey = key;

    const label = document.createElement("label");
    label.htmlFor = id;
    label.textContent = spec.label;

    if (spec.type === "check") {
      div.append(input, label);
    } else {
      div.append(label, input);
    }
    box.appendChild(div);
  }
}

function collectOptions() {
  const out = {};
  for (const el of $("opts").querySelectorAll("[data-opt-key]")) {
    out[el.dataset.optKey] = el.type === "checkbox" ? el.checked : el.value;
  }
  return out;
}

// ─────────────────────── generate design ───────────────────────

function showError(box, err) {
  box.hidden = false;
  box.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = err.message;
  box.appendChild(p);
  if (err.detail) {
    const pre = document.createElement("pre");
    pre.textContent = err.detail.trim().split("\n").slice(-4).join("\n");
    box.appendChild(pre);
  }
}

function download(path, filename) {
  const bytes = state.pyodide.FS.readFile(path);
  const url = URL.createObjectURL(
    new Blob([bytes], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** `cellClass(value, column)` styles individual cells — used for the ANOVA verdicts. */
function table(el, columns, rows, format = (v) => v, cellClass = () => "") {
  const cell = (r, c) => {
    const cls = cellClass(r[c], c);
    return `<td${cls ? ` class="${cls}"` : ""}>${escapeHtml(format(r[c], c))}</td>`;
  };
  el.innerHTML =
    `<thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${columns.map((c) => cell(r, c)).join("")}</tr>`).join("")}</tbody>`;
}

function escapeHtml(v) {
  if (v === null || v === undefined) return "";
  return String(v).replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

const num = (v) =>
  typeof v === "number" && Number.isFinite(v)
    ? Math.abs(v) >= 1e-4 && Math.abs(v) < 1e6
      ? Number(v.toFixed(4)).toString()
      : v.toExponential(3)
    : (v ?? "");

// ─────────────────────── writing the model down ───────────────────────

const SUPERSCRIPT = { 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };

/** A basis term as text: {T: 1, P: 2} → "T·P²". The intercept is "". */
function termText(powers) {
  return Object.entries(powers)
    .map(([factor, p]) => (Number(p) === 1 ? factor : `${factor}${SUPERSCRIPT[p] ?? `^${p}`}`))
    .join("·");
}

/** sympy's output, in the notation the rest of the page uses. */
const prettyExpr = (s) => s.replace(/\*\*/g, "^").replace(/\*/g, "·");

/**
 * The model as an equation, in terms of its parameter names:
 * "yield = b0 + b1·T + b11·T² + b12·T·P", or the user's own expression.
 * Returns null for a campaign with no model (the Latin-square family).
 */
function modelEquation(model) {
  if (!model) return null;
  if (model.expression) return `${model.response} = ${prettyExpr(model.expression)}`;
  if (!model.terms?.length) return null;
  const rhs = model.terms
    .map(({ parameter, powers }) => {
      const term = termText(powers);
      return term ? `${parameter}·${term}` : parameter;
    })
    .join(" + ");
  return `${model.response} = ${rhs}`;
}

/** The same equation with the fitted numbers in place of the parameter names. */
function fittedEquation(model, parameters) {
  if (!model) return null;
  if (model.fitted_expression) {
    return `${model.response} = ${prettyExpr(model.fitted_expression)}`;
  }
  if (!model.terms?.length) return null;
  const estimate = Object.fromEntries(parameters.map((p) => [p.name, p.estimate]));
  let rhs = "";
  for (const [i, { parameter, powers }] of model.terms.entries()) {
    const v = estimate[parameter];
    if (typeof v !== "number" || !Number.isFinite(v)) return null;
    const term = termText(powers);
    const piece = term ? `${num(Math.abs(v))}·${term}` : num(Math.abs(v));
    if (i === 0) rhs = v < 0 ? `−${piece}` : piece;
    else rhs += v < 0 ? ` − ${piece}` : ` + ${piece}`;
  }
  return `${model.response} = ${rhs}`;
}

// ───────────────────────── significance ─────────────────────────

// Templates that compare level means rather than fitting a model; `do_fit`
// rejects them, and that is the expected outcome, not a failure.
const COMBINATORIAL = ["latin-square", "graeco-latin", "hyper-graeco-latin"];

const ALPHA = 0.05;
const YES = "✓ yes";
const NO = "✗ no";
const VERDICT_CLASS = { [YES]: "yes", [NO]: "no" };

/** The verdict for one p-value. Rows with no F-ratio have nothing to test,
 *  and get an em dash rather than a red cross. */
const verdict = (p) => (typeof p === "number" && Number.isFinite(p) ? (p < ALPHA ? YES : NO) : "—");

/** The verdict for one coefficient: does its 95% interval exclude zero?
 *  Equivalent to the t-test at α = 0.05 — the interval is built from the same
 *  critical value — and it is the one rule that works for both fit paths,
 *  since the nonlinear fit reports intervals but no p-values. */
function coefficientVerdict(row) {
  const lo = row.ci_lower_95;
  const hi = row.ci_upper_95;
  if (![lo, hi].every((v) => typeof v === "number" && Number.isFinite(v))) return "—";
  return lo > 0 || hi < 0 ? YES : NO;
}

/**
 * One row per coefficient, from whichever shape the fit returned.
 *
 * The linear path returns `coefficients` with t-statistics and p-values; the
 * symbolic path returns only `parameters`. Everything common to both is kept,
 * and the verdict comes from the interval so the column means the same thing
 * either way.
 */
function coefficientRows(fit) {
  const stats = Object.fromEntries((fit.coefficients ?? []).map((c) => [c.name, c]));
  return fit.parameters.map((p) => {
    const c = stats[p.name];
    const row = { ...p };
    if (c) {
      row.t = c.t_statistic;
      row.p = c.p_value;
    }
    row.significant = coefficientVerdict(row);
    return row;
  });
}

/** A fit with no residual left: every p-value below is then degenerate. */
function isExact(fit, summary) {
  if (Number.isFinite(summary.R_squared) && summary.R_squared < 1 - 1e-12) return false;
  return Number.isFinite(fit.objective) && Math.abs(fit.objective) < 1e-20;
}

/** Render one or more equations as their own block of monospace lines. */
function equationBlock(...lines) {
  const div = document.createElement("div");
  div.className = "equation";
  for (const line of lines.filter(Boolean)) {
    const p = document.createElement("p");
    p.textContent = line;
    div.appendChild(p);
  }
  return div.children.length ? div : null;
}

let lastDesign = null;

async function generate() {
  const btn = $("generate");
  btn.disabled = true;
  $("design-error").hidden = true;
  try {
    const spec = {
      template: state.current.name,
      factors: state.factorRows,
      ...collectOptions(),
    };
    if (state.current.model_editor) {
      spec.expression = $("expression").value;
      spec.parameters = state.paramRows;
    }
    const out = call("create_design", JSON.stringify(spec));
    lastDesign = out;

    const factorNames = Object.keys(out.designs[0] ?? {}).filter((k) => k !== "run_id");
    $("design-summary").innerHTML =
      `<strong>${out.new_run_ids.length}</strong> runs · model <strong>${escapeHtml(
        out.template,
      )}</strong>` +
      (out.n_parameters
        ? ` · <strong>${out.n_parameters}</strong> parameters (${out.parameter_names.join(", ")})`
        : "");
    table($("design-table"), ["run_id", ...factorNames], out.designs, num);
    // The equation the design is built to estimate — the same one the fit in
    // step 2 reports coefficients for.
    const equation = modelEquation(out.model);
    $("design-equation").innerHTML = "";
    $("design-equation").hidden = !equation;
    if (equation) {
      $("design-equation").appendChild(equationBlock(equation));
      if (out.nominal_parameters && Object.keys(out.nominal_parameters).length) {
        const p = document.createElement("p");
        p.className = "hint";
        p.textContent = `centred on ${Object.entries(out.nominal_parameters)
          .map(([k, v]) => `${k} = ${num(v)}`)
          .join(", ")}`;
        $("design-equation").appendChild(p);
      }
    }
    $("design-result").hidden = false;
    $("download-design").hidden = false;
    download(out.file_path, out.download_name);
  } catch (err) {
    showError($("design-error"), err);
    $("design-result").hidden = true;
    $("download-design").hidden = true;
  } finally {
    btn.disabled = false;
  }
}

// ───────────────────────── analyze ─────────────────────────

const isBlank = (v) => v === null || v === undefined || (typeof v === "string" && !v.trim());

/** A response the analysis can use: something that reads as a finite number.
 *  A cell showing "3.5 g" or "n/a" is not blank but is not usable either. */
const isUsable = (v) => !isBlank(v) && Number.isFinite(Number(v));

/**
 * Ask Python to classify every response and factor cell.
 *
 * `inspect_workbook` reports a run as pending whether the cell is empty, holds
 * text, or holds a formula whose result Excel never wrote to the file — three
 * different mistakes with three different fixes, and the last one looks like a
 * perfectly good number on screen. This is what lets the message say which.
 */
function diagnose() {
  try {
    return call("diagnose_workbook", state.analyzePath);
  } catch (err) {
    console.error(err);
    return null;
  }
}

/** Checklist lines describing every distinct reason a run is unusable. */
function fixesFor(diag, response, pending) {
  if (!diag) {
    return [
      `Fill in '${response}' for run_id ${listRuns(pending)} with one measured number per run.`,
    ];
  }
  const lines = [];
  const runsWith = (s) => diag.runs.filter((r) => r.state === s);
  const idsOf = (rows) => listRuns(rows.map((r) => r.run_id));

  const formula = runsWith("formula");
  if (formula.length) {
    lines.push(
      `Run ${idsOf(formula)}: the '${diag.response}' cell holds a formula ` +
        `(${formula[0].shown}) whose computed result is not stored in the file. Excel shows you ` +
        "the number, but the file only carries it if Excel itself saved the workbook — so the " +
        "analysis sees an empty cell.",
      "Either re-open the workbook in Excel or LibreOffice and save it again, which caches the " +
        "results, or select the response column, copy it, and paste it back with Paste Special → " +
        "Values so the numbers are stored literally. Exports from Numbers, Google Sheets and " +
        "some scripts drop the cached results.",
    );
  }
  const text = runsWith("text");
  if (text.length) {
    lines.push(
      `Run ${idsOf(text)}: the '${diag.response}' cell is text rather than a number ` +
        `(it reads “${text[0].shown}”). Drop units and stray spaces, use a dot for the decimal ` +
        "point, and leave a run empty rather than writing “n/a” if it was not measured.",
    );
  }
  const empty = runsWith("empty");
  if (empty.length) {
    lines.push(
      `Run ${idsOf(empty)}: '${diag.response}' is empty. Type one measured number per run.`,
    );
  }
  const badInputs = diag.runs.filter((r) => r.bad_inputs.length);
  if (badInputs.length) {
    const first = badInputs[0].bad_inputs[0];
    lines.push(
      `Run ${idsOf(badInputs)}: the factor column '${first.column}' is ` +
        `${first.state === "formula" ? "a formula with no stored result" : "empty"} too. The ` +
        "factor columns are the design itself — restore them from the workbook step 1 generated " +
        "rather than retyping them.",
    );
  }
  return lines;
}

/** "1, 2, 3", truncated once the list stops being something you can act on. */
function listRuns(ids) {
  const shown = ids.slice(0, 12).join(", ");
  return ids.length > 12 ? `${shown} … (${ids.length} in all)` : shown;
}

function setStatus(text) {
  $("analyze-status").hidden = !text;
  $("analyze-status-text").textContent = text ?? "";
}

// `call()` runs Python synchronously and blocks the main thread, so anything
// written to the DOM just before it would not be painted until it returns.
//
// Raced against a timer rather than left to requestAnimationFrame alone:
// browsers stop firing frame callbacks for a tab that is hidden, backgrounded
// or fully occluded, and the obvious thing to do after dropping a workbook is
// to switch back to Excel. Waiting for a frame that will not come until you
// return is indistinguishable from a hang.
const PAINT_TIMEOUT_MS = 50;
const paint = () =>
  new Promise((resolve) => {
    let done = false;
    const go = () => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(go, PAINT_TIMEOUT_MS);
    requestAnimationFrame(() => setTimeout(go, 0));
  });

/**
 * Report a blocked analysis: what stopped, why, and the specific edits that
 * would let it run. `fixes` is a checklist; `err`, when the failure came from
 * Python, supplies the underlying message and the tail of its traceback.
 */
function showProblem(title, fixes = [], err = null) {
  const box = $("analyze-error");
  box.hidden = false;
  box.innerHTML = "";

  const h = document.createElement("p");
  h.className = "alert-title";
  h.textContent = title;
  box.appendChild(h);

  if (err?.message) {
    const why = document.createElement("p");
    why.className = "alert-why";
    why.textContent = err.message;
    box.appendChild(why);
  }
  if (fixes.length) {
    const ul = document.createElement("ul");
    for (const fix of fixes) {
      const li = document.createElement("li");
      li.textContent = fix;
      ul.appendChild(li);
    }
    box.appendChild(ul);
  }
  if (err?.detail) {
    const pre = document.createElement("pre");
    pre.textContent = err.detail.trim().split("\n").slice(-4).join("\n");
    box.appendChild(pre);
  }
}

/** A non-blocking caveat, shown inline with the results it qualifies. */
function note(title, lines) {
  const div = document.createElement("div");
  div.className = "note";
  const h = document.createElement("p");
  h.className = "note-title";
  h.textContent = title;
  div.appendChild(h);
  for (const line of [].concat(lines)) {
    const p = document.createElement("p");
    p.textContent = line;
    div.appendChild(p);
  }
  return div;
}

function block(title, hint, build) {
  const section = document.createElement("div");
  section.className = "result-block";
  const h = document.createElement("h3");
  h.textContent = title;
  const p = document.createElement("p");
  p.className = "hint";
  p.textContent = hint;
  section.append(h, p);
  build(section);
  return section;
}

function dataTable(el, columns, rows, { scroll = false, cellClass } = {}) {
  const t = document.createElement("table");
  t.className = "grid data";
  table(t, columns, rows, num, cellClass);
  if (!scroll) {
    el.appendChild(t);
    return;
  }
  const div = document.createElement("div");
  div.className = "scroll";
  div.appendChild(t);
  el.appendChild(div);
}

/**
 * Fill step 1 in from an uploaded campaign.
 *
 * A workbook carries its whole design, so showing it beats leaving the form on
 * the page defaults: you can see what produced the file you are analyzing, and
 * the form is then set up to build the next design like it. Whatever was typed
 * before is replaced, which is why it says so out loud.
 */
function syncDesignFrom(file) {
  const box = $("design-source");
  box.innerHTML = "";
  box.hidden = true;

  let spec;
  try {
    spec = call("design_spec_from_workbook", state.analyzePath);
  } catch (err) {
    console.error(err);
    return;
  }

  if (!spec.supported || !state.templates.has(spec.template)) {
    box.appendChild(
      note(`Not filled in from ${file.name}`, [
        spec.reason ?? `'${spec.template}' is not a design this page can build.`,
        "The analysis in step 2 is unaffected — it reads the workbook directly.",
      ]),
    );
    box.hidden = false;
    return;
  }

  $("template").value = spec.template;
  selectTemplate(); // rebuilds the editors this template needs

  if (spec.factors?.length) state.factorRows = spec.factors.map((row) => ({ ...row }));
  if (typeof spec.expression === "string") $("expression").value = spec.expression;
  if (spec.parameters?.length) state.paramRows = spec.parameters.map((p) => ({ ...p }));
  renderFactors();
  renderParams();

  for (const [key, value] of Object.entries(spec.options ?? {})) {
    const input = $(`opt-${key}`);
    if (!input) continue;
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value;
  }
  if (state.current?.model_editor) checkModel();

  box.appendChild(
    note(`Filled in from ${file.name}`, [
      "This is the design that workbook was built with. Generating from here makes a new " +
        "workbook rather than adding runs to that one — for another batch of the same " +
        "campaign, use `discopt doe extend` at the command line.",
    ]),
  );
  box.hidden = false;
}

/**
 * Drop or pick a workbook and the whole analysis runs: read it, fit, ANOVA.
 * Anything that stops it short reports what is wrong with the file and what
 * to change, because the fix is always an edit to the workbook.
 */
async function acceptFile(file) {
  if (!file) return;
  // A second drop landing mid-analysis would interleave its output with the
  // run in flight: `call()` blocks the thread, but the awaits between steps
  // let queued events through. Say so rather than dropping it on the floor —
  // an ignored drop looks like the page is broken.
  if (state.analyzing) {
    setStatus(`Still working — ${file.name} will not be read until this finishes.`);
    return;
  }
  state.analyzing = true;
  try {
    await analyzeFile(file);
  } catch (err) {
    showProblem("The analysis stopped unexpectedly", [
      "This is a bug rather than something wrong with your workbook — the browser console has " +
        "the full trace.",
      "Reloading the page and dropping the file again usually clears it.",
    ], err);
    console.error(err);
  } finally {
    state.analyzing = false;
    setStatus("");
  }
}

async function analyzeFile(file) {
  const box = $("analyze-result");
  box.innerHTML = "";
  box.hidden = true;
  $("analyze-error").hidden = true;
  $("download-analyzed").hidden = true;
  $("upload-name").hidden = true;
  setStatus("");

  if (!/\.xlsx$/i.test(file.name)) {
    showProblem(`“${file.name}” is not an .xlsx workbook`, [
      "Step 1 downloads a .xlsx campaign workbook — drop that same file back here once you have " +
        "filled in the response column.",
      "A .csv or .xls file has to be re-saved from Excel first, as “Excel Workbook (.xlsx)”. " +
        "Saving as CSV throws away the design metadata the analysis needs.",
    ]);
    return;
  }
  if (file.size === 0) {
    showProblem(`“${file.name}” is empty`, [
      "The file is 0 bytes — the save probably did not finish, or the download was interrupted.",
      "Re-save it from Excel, or regenerate the design in step 1, and drop it here again.",
    ]);
    return;
  }

  $("upload-name").textContent = `Loaded ${file.name} (${(file.size / 1024).toFixed(0)} kB)`;
  $("upload-name").hidden = false;
  setStatus(`Reading ${file.name}…`);
  await paint();

  let out;
  try {
    state.pyodide.FS.writeFile(state.analyzePath, new Uint8Array(await file.arrayBuffer()));
    out = call("inspect_workbook", state.analyzePath);
  } catch (err) {
    setStatus("");
    // The cell-level diagnosis works on files `inspect_workbook` refuses, so
    // an unreadable response column is described rather than just rejected.
    const diag = diagnose();
    const cells = diag ? fixesFor(diag, "the response column", []) : [];
    showProblem("Could not read that workbook", [
      ...cells,
      "It has to be a workbook generated in step 1: the analysis reads the design, the model " +
        "and the factor metadata that discopt-doe writes into its hidden sheets.",
      "Keep the sheet names and the header row exactly as generated — renaming a sheet or a " +
        "column, or deleting rows, removes what the fit needs.",
      "Type your measurements into the response column only, and leave the run_id and factor " +
        "columns untouched.",
      "If another tool re-exported the file, generate a fresh design and copy your numbers in.",
    ], err);
    return;
  }

  const s = out.status;
  box.appendChild(
    block(
      "Campaign",
      `${s.template || "custom"} · ${s.n_completed} of ${s.n_total} runs completed`,
      (el) => dataTable(el, out.columns, out.rows, { scroll: true }),
    ),
  );
  box.hidden = false;
  // Step 1 now shows the design this workbook came from.
  syncDesignFrom(file);
  setStatus("");

  const pending = out.rows.filter((r) => !isUsable(r[out.response])).map((r) => r.run_id);
  const diag = pending.length ? diagnose() : null;

  if (!s.n_completed) {
    showProblem(`Nothing to analyze yet — no run has a usable '${out.response}' value`, [
      `'${out.response}' is the column the fit and the ANOVA read.`,
      ...fixesFor(diag, out.response, pending),
      "Partly filled is fine: the analysis runs on whatever rows do have a response, and says " +
        "which are still missing.",
      "Save as .xlsx and drop the file here again — the fit and the ANOVA re-run on their own.",
    ]);
    return;
  }

  await runAnalyses(out, pending, diag);
}

/** Fit and ANOVA, one after the other. They are independent — one failing
 *  does not stop the other, so a failure is reported next to the results
 *  that did come through rather than replacing them. */
async function runAnalyses(out, pending, diag) {
  const box = $("analyze-result");
  const s = out.status;

  if (pending.length) {
    box.appendChild(
      note(`${pending.length} of ${s.n_total} runs have no usable response`, [
        `Analyzing the ${s.n_completed} completed run${s.n_completed === 1 ? "" : "s"} only. ` +
          "Estimates from a partial design are less precise than the design was built for, and " +
          "the ANOVA can be unbalanced.",
        ...fixesFor(diag, out.response, pending),
        "Fix those, save, and drop the file here again for the complete analysis.",
      ]),
    );
  }

  setStatus("Fitting the model…");
  await paint();
  let fitted = false;
  try {
    const { fit, model } = call("run_fit", state.analyzePath);
    fitted = true;
    // `coefficients` carries the t-tests; the symbolic path returns only
    // `parameters`. Both carry the 95% interval, which is what the verdict
    // reads, so the two shapes are merged into one table.
    const rows = coefficientRows(fit);
    const summary = fit.summary ?? {};
    box.appendChild(
      block(
        "Fitted coefficients",
        [
          `${fit.n_observations} observations`,
          `${rows.length} parameters`,
          Number.isFinite(summary.degrees_of_freedom)
            ? `${summary.degrees_of_freedom} residual df`
            : null,
          Number.isFinite(summary.R_squared) ? `R² = ${num(summary.R_squared)}` : null,
          `residual sum of squares ${num(fit.objective)}`,
          `significance at α = ${ALPHA}`,
        ]
          .filter(Boolean)
          .join(" · "),
        (el) => {
          // The model, then the same model with the fitted numbers in it —
          // which is the thing you actually want to take away from a fit.
          const equations = equationBlock(
            modelEquation(model),
            fittedEquation(model, fit.parameters),
          );
          if (equations) el.appendChild(equations);

          const columns = ["name", "estimate", "std_error", "ci_lower_95", "ci_upper_95"];
          if (rows.some((r) => r.p !== undefined)) columns.push("t", "p");
          columns.push("significant");
          dataTable(el, columns, rows, {
            scroll: true,
            cellClass: (v, c) => (c === "significant" ? (VERDICT_CLASS[v] ?? "") : ""),
          });

          const legend = document.createElement("p");
          legend.className = "hint";
          legend.textContent =
            `${YES} marks a coefficient whose 95% interval excludes zero — the data resolve that ` +
            `term. ${NO} means this experiment could not tell it from zero, which is not the same ` +
            "as showing it is zero.";
          el.appendChild(legend);

          // Fewer observations than parameters: the estimates come from a
          // least-norm solution and the covariance is undefined, so the blank
          // std_error / CI cells need explaining rather than looking broken.
          const n_p = rows.length;
          if (fit.n_observations < n_p) {
            el.appendChild(
              note("Under-determined fit — standard errors are not available", [
                `${n_p} parameters were estimated from ${fit.n_observations} observation` +
                  `${fit.n_observations === 1 ? "" : "s"}, so there is no residual degree of ` +
                  "freedom and the blank columns above cannot be computed.",
                `Fill in at least ${n_p + 1} runs of '${out.response}' for confidence intervals.`,
              ]),
            );
          } else if (isExact(fit, summary)) {
            el.appendChild(
              note("This fit is exact — the residual is numerically zero", [
                `The model reproduces every observation to ${num(fit.objective)}, so there is no ` +
                  "scatter left to measure a coefficient against. Simulated or formula-generated " +
                  "responses do this.",
                "The verdicts below still separate the terms that carry the response from the " +
                  "ones that came out at rounding error, but read them as “non-zero”, not as a " +
                  "statistical result — with real measurement noise the intervals would be wider.",
              ]),
            );
          }

          // The model's own ANOVA: how much of the response the fit explains,
          // against what it leaves over.
          if (fit.regression_anova?.length) {
            const anovaRows = fit.regression_anova.map((r) => ({
              source: r.source,
              ss: r.ss,
              df: r.df,
              ms: r.ms,
              f: r.f_statistic,
              p: r.p_value,
              significant: verdict(r.p_value),
            }));
            const h = document.createElement("h4");
            h.textContent = "Regression ANOVA";
            const hint = document.createElement("p");
            hint.className = "hint";
            hint.textContent =
              "Does the model as a whole explain the response better than its mean does?";
            el.append(h, hint);
            dataTable(el, ["source", "ss", "df", "ms", "f", "p", "significant"], anovaRows, {
              cellClass: (v, c) => (c === "significant" ? (VERDICT_CLASS[v] ?? "") : ""),
            });
          }
        },
      ),
    );
    // fit writes the parameters/FIM/ANOVA sheets back into the workbook.
    $("download-analyzed").hidden = false;
  } catch (err) {
    box.appendChild(
      note("The model fit did not run", [
        err.message,
        !s.template
          ? "The workbook carries no template metadata, so there is no model to fit. Generate " +
            "the design in step 1 and the fit will have something to work with."
          : COMBINATORIAL.includes(s.template)
            ? `A '${s.template}' design has no model to fit — it compares level means, so the ` +
              "ANOVA below is its analysis and nothing is missing here."
            : "A fit usually fails because a factor cell was blanked out or edited to something " +
              "that is not a number; the message above names the run when that is the cause.",
      ]),
    );
  }

  // The factor-level ANOVA compares the mean response across each factor's
  // levels. That is the analysis for a design built out of levels; on a
  // continuous design every distinct value becomes its own "level", so it
  // decomposes nothing and reports F-ratios against an aliased residual. Those
  // designs get the regression ANOVA above instead. It still runs as a
  // fallback when the fit did not, since then it is the only analysis left.
  if (!out.combinatorial && fitted) {
    setStatus("");
    return;
  }

  setStatus("Running the ANOVA…");
  await paint();
  try {
    const { anova } = call("run_anova", state.analyzePath, "[]");
    box.appendChild(
      block(
        "ANOVA over factor levels",
        `${anova.n_observations} observations · grand mean ${num(anova.grand_mean)}` +
          (anova.balanced ? " · balanced" : " · unbalanced (marginal SS)") +
          ` · significance at α = ${ALPHA}`,
        (el) => {
          const rows = anova.rows.map((r) => ({ ...r, significant: verdict(r.p) }));
          dataTable(el, ["source", "ss", "df", "ms", "f", "p", "significant"], rows, {
            cellClass: (v, c) => (c === "significant" ? (VERDICT_CLASS[v] ?? "") : ""),
          });
          const legend = document.createElement("p");
          legend.className = "hint";
          legend.textContent =
            `${YES} marks a factor whose level means differ by more than the residual scatter ` +
            `explains. ${NO} means this experiment did not resolve a difference, which is not ` +
            "the same as showing there is none. Rows without an F-ratio (residual, total) have " +
            "nothing to test.";
          el.appendChild(legend);
          if (!out.combinatorial) {
            el.appendChild(
              note("Read this one with care", [
                "This design's factors are continuous, so each distinct value is being treated " +
                  "as its own level. It is shown because the fit did not run; the coefficient " +
                  "tests, not this table, are the analysis this design was built for.",
              ]),
            );
          }
        },
      ),
    );
  } catch (err) {
    box.appendChild(
      note("The ANOVA did not run", [
        err.message,
        "A factor-level ANOVA needs factors that repeat across runs. A space-filling design (a " +
          "Latin hypercube, say) gives every run its own factor values, so there is nothing to " +
          "pool — the fitted coefficients are its analysis.",
      ]),
    );
  }

  setStatus("");
}

// ─────────────────────────── wiring ───────────────────────────

$("template").addEventListener("change", selectTemplate);
$("add-factor").addEventListener("click", () => {
  state.factorRows.push(blankFactor(state.factorRows.length));
  renderFactors();
});
$("expression").addEventListener("input", scheduleModelCheck);
$("add-param").addEventListener("click", () => {
  state.paramRows.push({ name: `p${state.paramRows.length + 1}`, value: 1 });
  renderParams();
  checkModel();
});
$("generate").addEventListener("click", generate);
$("download-design").addEventListener("click", () => {
  if (lastDesign) download(lastDesign.file_path, lastDesign.download_name);
});
$("upload").addEventListener("change", (e) => {
  const file = e.target.files[0];
  // Clear the input so re-picking the same file (after editing it) fires
  // `change` again and re-runs the analysis.
  e.target.value = "";
  acceptFile(file);
});
$("download-analyzed").addEventListener("click", () =>
  download(state.analyzePath, "analyzed-campaign.xlsx"),
);

const drop = $("drop");
for (const evt of ["dragenter", "dragover"]) {
  drop.addEventListener(evt, (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
}
for (const evt of ["dragleave", "drop"]) {
  drop.addEventListener(evt, (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (evt !== "drop") return;
    const files = e.dataTransfer.files;
    if (!files.length) {
      // Dragged text, a link, or a folder: Chrome and Safari report no file
      // for all three, and the drop looks like it simply did nothing.
      showProblem("That drop carried no file", [
        "Drag the .xlsx file itself — a folder, a link, or selected text cannot be read.",
        "From Excel, save the workbook to disk first, then drag it from Finder or Explorer.",
      ]);
      return;
    }
    acceptFile(files[0]);
  });
}

boot().catch((err) => {
  setBoot(`Could not start: ${err.message}`, "failed");
  console.error(err);
});
