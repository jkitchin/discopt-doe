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

function table(el, columns, rows, format = (v) => v) {
  el.innerHTML =
    `<thead><tr>${columns.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows
      .map(
        (r) =>
          `<tr>${columns.map((c) => `<td>${escapeHtml(format(r[c], c))}</td>`).join("")}</tr>`,
      )
      .join("")}</tbody>`;
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
    if (out.expression) {
      $("design-summary").innerHTML +=
        `<br><span class="hint">model <code>${escapeHtml(out.expression)}</code>` +
        ` centred on ${Object.entries(out.nominal_parameters ?? {})
          .map(([k, v]) => `${escapeHtml(k)}=${num(v)}`)
          .join(", ")}</span>`;
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

async function acceptFile(file) {
  if (!file) return;
  const buf = new Uint8Array(await file.arrayBuffer());
  state.pyodide.FS.writeFile(state.analyzePath, buf);
  $("upload-name").textContent = `Loaded ${file.name} (${(file.size / 1024).toFixed(0)} kB)`;
  $("upload-name").hidden = false;
  $("analyze-error").hidden = true;
  $("analyze-result").hidden = true;
  $("download-analyzed").hidden = true;

  try {
    const out = call("inspect_workbook", state.analyzePath);
    const s = out.status;
    const box = $("analyze-result");
    box.innerHTML = "";
    box.appendChild(
      block(
        "Campaign",
        `${s.template || "custom"} · ${s.n_completed} of ${s.n_total} runs completed`,
        (el) => {
          const t = document.createElement("table");
          t.className = "grid data";
          table(t, out.columns, out.rows, num);
          const scroll = document.createElement("div");
          scroll.className = "scroll";
          scroll.appendChild(t);
          el.appendChild(scroll);
        },
      ),
    );
    box.hidden = false;
    const ready = s.n_completed > 0;
    $("fit").disabled = !ready;
    $("anova").disabled = !ready;
    if (!ready) {
      showError($("analyze-error"), {
        message: `No completed runs yet — fill in the '${out.response}' column and re-upload.`,
      });
    }
  } catch (err) {
    showError($("analyze-error"), err);
    $("fit").disabled = true;
    $("anova").disabled = true;
  }
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

function runAnalysis(kind) {
  $("analyze-error").hidden = true;
  try {
    const box = $("analyze-result");
    if (kind === "fit") {
      const { fit } = call("run_fit", state.analyzePath);
      box.appendChild(
        block(
          "Fitted coefficients",
          `${fit.n_observations} observations · residual sum of squares ${num(fit.objective)}`,
          (el) => {
            const t = document.createElement("table");
            t.className = "grid data";
            table(
              t,
              ["name", "estimate", "std_error", "ci_lower_95", "ci_upper_95"],
              fit.parameters,
              num,
            );
            el.appendChild(t);
          },
        ),
      );
    } else {
      const { anova } = call("run_anova", state.analyzePath, "[]");
      box.appendChild(
        block(
          "ANOVA",
          `${anova.n_observations} observations · grand mean ${num(anova.grand_mean)}` +
            (anova.balanced ? " · balanced" : " · unbalanced (marginal SS)"),
          (el) => {
            const t = document.createElement("table");
            t.className = "grid data";
            table(t, ["source", "ss", "df", "ms", "f", "p"], anova.rows, num);
            el.appendChild(t);
          },
        ),
      );
    }
    box.hidden = false;
    // fit writes parameters/FIM/ANOVA sheets back into the workbook.
    $("download-analyzed").hidden = kind !== "fit";
  } catch (err) {
    showError($("analyze-error"), err);
  }
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
$("upload").addEventListener("change", (e) => acceptFile(e.target.files[0]));
$("fit").addEventListener("click", () => runAnalysis("fit"));
$("anova").addEventListener("click", () => runAnalysis("anova"));
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
    if (evt === "drop") acceptFile(e.dataTransfer.files[0]);
  });
}

boot().catch((err) => {
  setBoot(`Could not start: ${err.message}`, "failed");
  console.error(err);
});
