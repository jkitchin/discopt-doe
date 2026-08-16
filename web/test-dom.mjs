// The browser app's front half: drop a workbook, get an analysis.
//
// `test-wasm.mjs` covers the Python side in real WebAssembly. This covers what
// happens around it — that dropping a file runs the fit and the ANOVA without
// anyone pressing a button, and that every way a workbook can be unusable
// produces a message naming the runs and the edit that fixes them.
//
//   cd web && node test-dom.mjs
//
// No dependencies and no browser: app.js is loaded against the smallest DOM it
// will accept and a Pyodide stub that answers with bootstrap.py's JSON
// contract, so the assertions are about the text a user actually sees.

import fs from "node:fs";
import path from "node:path";

import { WEB, installFakeDom, loadApp } from "./fake-dom.mjs";

const { nodes, missing, html } = installFakeDom();

// ── a Pyodide that answers only in bootstrap.py's JSON contract ─────

let RESPONSES = {};
const pyodide = {
  globals: {
    get(fn) {
      const f = (...args) =>
        JSON.stringify(RESPONSES[fn] ? RESPONSES[fn](...args) : { ok: false, error: `no stub: ${fn}` });
      f.destroy = () => {};
      return f;
    },
    set() {},
    delete() {},
  },
  FS: { writeFile() {}, readFile: () => new Uint8Array([1]), mkdirTree() {} },
};

// ── load app.js: skip boot(), expose what the test drives ───────────

const consoleError = console.error;
console.error = () => {}; // the app logs diagnosis failures it then recovers from
const { acceptFile, state, loadTemplates, cleanup } = await loadApp();
state.pyodide = pyodide;

// The template catalogue boot() would have loaded. Shapes mirror TEMPLATE_UI
// in bootstrap.py; test-wasm.mjs checks the real thing.
RESPONSES = {
  describe_templates: () => ({
    ok: true,
    groups: [
      {
        label: "Space-filling & response surface",
        hint: "",
        templates: [
          {
            name: "central-composite",
            description: "A rotatable response-surface design.",
            factors: "bounds",
            options: ["center_points", "alpha", "outside_bounds"],
            min: 2,
            max: 6,
          },
          {
            name: "box-behnken",
            description: "A 3-level response-surface design.",
            factors: "bounds",
            options: ["center_points"],
            min: 3,
            max: 5,
          },
        ],
      },
    ],
  }),
};
loadTemplates();

// ── fixtures ────────────────────────────────────────────────────────

const file = (name, size = 1024) => ({ name, size, arrayBuffer: async () => new ArrayBuffer(size) });
const usable = (v) => v !== null && v !== "" && Number.isFinite(Number(v));
const COMBINATORIAL = ["latin-square", "graeco-latin", "hyper-graeco-latin", "factorial-2level"];
/** `inspect_workbook`'s answer for a campaign whose responses are `resp`. */
const inspected = (resp, template = "box-behnken") => () => ({
  ok: true,
  status: { template, n_total: resp.length, n_completed: resp.filter(usable).length },
  columns: ["run_id", "T", "y"],
  rows: resp.map((y, i) => ({ run_id: i + 1, T: 300 + i, y })),
  response: "y",
  combinatorial: COMBINATORIAL.includes(template),
});
const diagnosed = (states) => () => ({
  ok: true,
  response: "y",
  runs: states.map(([state, shown], i) => ({ run_id: i + 1, state, shown, bad_inputs: [] })),
});
const FIT = {
  ok: true,
  fit: {
    n_observations: 8,
    objective: 0.01,
    parameters: [
      { name: "b0", estimate: 1, std_error: 0.1, ci_lower_95: 0.8, ci_upper_95: 1.2 },
      { name: "b1", estimate: 2.5, std_error: 0.1, ci_lower_95: 2.3, ci_upper_95: 2.7 },
      // b11's interval straddles zero: this experiment cannot tell it from 0.
      { name: "b11", estimate: -0.75, std_error: 0.9, ci_lower_95: -2.6, ci_upper_95: 1.1 },
    ],
    coefficients: [
      { name: "b0", t_statistic: 10, p_value: 1e-4 },
      { name: "b1", t_statistic: 25, p_value: 1e-7 },
      { name: "b11", t_statistic: -0.83, p_value: 0.44 },
    ],
    regression_anova: [
      { source: "Regression", ss: 30, df: 2, ms: 15, f_statistic: 45, p_value: 0.0002 },
      { source: "Residual", ss: 1.7, df: 5, ms: 0.34, f_statistic: null, p_value: null },
      { source: "Total (corrected)", ss: 31.7, df: 7, ms: null, f_statistic: null, p_value: null },
    ],
    summary: { n_observations: 8, n_parameters: 3, degrees_of_freedom: 5, R_squared: 0.947 },
  },
  model: {
    response: "yield",
    expression: null,
    terms: [
      { parameter: "b0", powers: {} },
      { parameter: "b1", powers: { T: 1 } },
      { parameter: "b11", powers: { T: 2 } },
    ],
  },
};
const ANOVA = {
  ok: true,
  anova: {
    n_observations: 8,
    grand_mean: 2,
    balanced: true,
    rows: [
      { source: "T", ss: 12, df: 1, ms: 12, f: 40, p: 0.001 },
      { source: "P", ss: 0.2, df: 1, ms: 0.2, f: 0.7, p: 0.44 },
      { source: "residual", ss: 1.5, df: 5, ms: 0.3, f: null, p: null },
    ],
  },
};

// ── scenarios ───────────────────────────────────────────────────────

let failures = 0;

/** Drop `file` with `stubs` in place and assert on what the page then shows. */
async function scenario(label, stubs, want) {
  RESPONSES = stubs;
  await acceptFile(want.file ?? file("campaign.xlsx"));
  const shown = `${nodes["analyze-error"].visibleText}\n${nodes["analyze-result"].visibleText}`;
  const markup = nodes["analyze-result"].markup;
  const problems = [
    ...(want.contains ?? []).filter((t) => !shown.includes(t)).map((t) => `missing: ${t}`),
    ...(want.absent ?? []).filter((t) => shown.includes(t)).map((t) => `should not say: ${t}`),
    ...(want.markup ?? []).filter((t) => !markup.includes(t)).map((t) => `missing markup: ${t}`),
    ...(want.markupAbsent ?? [])
      .filter((t) => markup.includes(t))
      .map((t) => `markup should not contain: ${t}`),
  ];
  if (want.download !== undefined && nodes["download-analyzed"].hidden !== !want.download) {
    problems.push(`download button ${want.download ? "hidden" : "shown"}`);
  }
  if (!nodes["analyze-status"].hidden) problems.push("the progress line was left showing");
  if (problems.length) {
    failures++;
    console.log(`FAIL  ${label}\n      ${problems.join("\n      ")}\n----- shown -----\n${shown}\n`);
  } else {
    console.log(`PASS  ${label}`);
  }
}

await scenario(
  "dropping a filled-in workbook fits it and judges each coefficient, with no clicks",
  { inspect_workbook: inspected([1, 2, 3, 4]), run_fit: () => FIT, run_anova: () => ANOVA },
  {
    contains: [
      "Campaign",
      "Fitted coefficients",
      // the model, then the model with its fitted numbers in it
      "yield = b0 + b1·T + b11·T²",
      "yield = 1 + 2.5·T − 0.75·T²",
      "R² = 0.947",
      "5 residual df",
      "α = 0.05",
      // a resolved coefficient, and one whose interval straddles zero
      "✓ yes",
      "✗ no",
      // the model's own decomposition, not a factor-level one
      "Regression ANOVA",
      "Total (corrected)",
    ],
    // the verdict cells are coloured, and the ✓/✗ carries it without colour
    markup: ['class="yes"', 'class="no"'],
    // a continuous design gets no factor-level ANOVA: every value is its own level
    absent: ["did not run", "ANOVA over factor levels", "grand mean"],
    download: true,
  },
);

await scenario(
  "a level-based design gets the factor ANOVA, since that is its analysis",
  {
    inspect_workbook: inspected([1, 2, 3, 4], "latin-square"),
    run_fit: () => ({ ok: false, error: "fit is not defined for latin-square" }),
    run_anova: () => ANOVA,
  },
  {
    contains: ["ANOVA over factor levels", "grand mean", "✓ yes", "✗ no", "level means differ"],
    absent: ["Read this one with care"],
    download: false,
  },
);

await scenario(
  "a continuous design falls back to the factor ANOVA only if the fit failed, with a warning",
  {
    inspect_workbook: inspected([1, 2, 3, 4]),
    run_fit: () => ({ ok: false, error: "run 3 has a blank value for input 'T'" }),
    run_anova: () => ANOVA,
  },
  {
    contains: ["The model fit did not run", "ANOVA over factor levels", "Read this one with care"],
    download: false,
  },
);

await scenario(
  "an exact fit says so, instead of presenting degenerate p-values as a result",
  {
    inspect_workbook: inspected([1, 2, 3, 4]),
    run_fit: () => ({
      ok: true,
      fit: {
        ...FIT.fit,
        objective: 1.738e-30,
        summary: { ...FIT.fit.summary, R_squared: 1 },
      },
      model: FIT.model,
    }),
  },
  { contains: ["This fit is exact", "no scatter left"], download: true },
);

await scenario(
  "a user-defined model is shown as its own expression, fitted values and all",
  {
    inspect_workbook: inspected([1, 2, 3, 4]),
    run_fit: () => ({
      ok: true,
      fit: {
        n_observations: 6,
        objective: 1e-20,
        parameters: [
          { name: "k0", estimate: 3.7, std_error: 0.01, ci_lower_95: 3.6, ci_upper_95: 3.8 },
          { name: "Ea", estimate: 6200, std_error: 5, ci_lower_95: 6190, ci_upper_95: 6210 },
        ],
      },
      model: {
        response: "rate",
        expression: "k0*exp(-0.1203*Ea/T)",
        terms: null,
        fitted_expression: "3.700*exp(-745.9/T)",
      },
    }),
    run_anova: () => ANOVA,
  },
  { contains: ["rate = k0·exp(-0.1203·Ea/T)", "rate = 3.700·exp(-745.9/T)"], download: true },
);

await scenario(
  // A Latin square: `_model_summary` returns null, there is no equation to
  // write, and the missing fit is the expected outcome rather than a failure.
  "a design with no model shows no equation, and says the ANOVA is the analysis",
  {
    inspect_workbook: inspected([1, 2, 3, 4], "latin-square"),
    run_fit: () => ({ ok: false, error: "fit is not defined for latin-square" }),
    run_anova: () => ANOVA,
  },
  {
    contains: ["ANOVA", "✓ yes", "has no model to fit", "nothing is missing here"],
    absent: ["Fitted coefficients"],
    markupAbsent: ['class="equation"'],
    download: false,
  },
);

await scenario("a .csv is refused with the reason", {}, {
  file: file("data.csv"),
  contains: ["not an .xlsx workbook", "Excel Workbook"],
  download: false,
});

await scenario("a 0-byte file is refused with the reason", {}, {
  file: file("campaign.xlsx", 0),
  contains: ["is empty", "0 bytes"],
  download: false,
});

await scenario(
  // The failure that prompted this: Excel shows the computed number, but the
  // file carries only the formula, so the analysis sees an empty cell.
  "a formula with no stored result is explained rather than just rejected",
  {
    inspect_workbook: () => ({
      ok: false,
      error: "run 1: response cell 'y' contains the formula '=B2*2' but no computed value is stored.",
      detail: "Traceback (most recent call last):\n  ...\nValueError: ...",
    }),
    diagnose_workbook: diagnosed([["formula", "=B2*2"], ["ok", "3"]]),
  },
  {
    contains: ["Could not read that workbook", "holds a formula", "=B2*2", "Paste Special"],
    download: false,
  },
);

await scenario(
  "an untouched workbook names the runs that are still blank",
  {
    inspect_workbook: inspected([null, null, null]),
    diagnose_workbook: diagnosed([["empty", ""], ["empty", ""], ["empty", ""]]),
  },
  { contains: ["Nothing to analyze yet", "Run 1, 2, 3", "is empty"], download: false },
);

await scenario(
  "a partly filled workbook is analyzed, and the rest is explained",
  {
    inspect_workbook: inspected([1, "3.5 g", null, 4]),
    diagnose_workbook: diagnosed([["ok", "1"], ["text", "3.5 g"], ["empty", ""], ["ok", "4"]]),
    run_fit: () => FIT,
    run_anova: () => ANOVA,
  },
  {
    contains: ["have no usable response", "text rather than a number", "3.5 g", "Fitted coefficients", "ANOVA"],
    download: true,
  },
);

await scenario(
  "a fit that does not apply still leaves the ANOVA",
  {
    inspect_workbook: inspected([1, 2, 3, 4]),
    run_fit: () => ({ ok: false, error: "fit is not defined for latin-square" }),
    run_anova: () => ANOVA,
  },
  { contains: ["The model fit did not run", "latin-square", "grand mean"], download: false },
);

await scenario(
  // Both analyses failing is still a readable page: two notes and the campaign.
  "when neither analysis runs, both say why and the campaign table survives",
  {
    inspect_workbook: inspected([1, 2, 3, 4], "latin-square"),
    run_fit: () => ({ ok: false, error: "fit is not defined for latin-square" }),
    run_anova: () => ({ ok: false, error: "every run has distinct factor values" }),
  },
  {
    contains: [
      "Campaign",
      "The model fit did not run",
      "The ANOVA did not run",
      "distinct factor values",
    ],
    download: false,
  },
);

await scenario(
  "an under-determined fit says why the standard errors are blank",
  {
    inspect_workbook: inspected([1, 2]),
    run_fit: () => ({
      ok: true,
      fit: {
        n_observations: 2,
        objective: 0,
        parameters: ["b0", "b1", "b2"].map((name) => ({
          name, estimate: 1, std_error: null, ci_lower_95: null, ci_upper_95: null,
        })),
      },
    }),
    run_anova: () => ANOVA,
  },
  { contains: ["Under-determined fit", "at least 4 runs"], download: true },
);

// ── a hidden element must actually be hidden ────────────────────────

{
  // `hidden` is honoured by a UA-stylesheet rule, so any author rule setting
  // `display` outranks it: `.boot { display: flex }` alone left the analysis
  // spinner on screen forever, spinning after the work was done. The fake DOM
  // here cannot see CSS, so check the stylesheet itself.
  const css = fs.readFileSync(path.join(WEB, "style.css"), "utf8");
  const guarded = /\[hidden\]\s*\{[^}]*display:\s*none\s*!important/.test(css);
  const hiddenWithClass = [...html.matchAll(/<[^>]*\bhidden\b[^>]*>/g)].filter((m) =>
    /class="/.test(m[0]),
  );
  if (hiddenWithClass.length && !guarded) {
    failures++;
    console.log(
      "FAIL  elements carry both `hidden` and a class, but style.css has no " +
        "[hidden] { display: none !important } guard",
    );
  } else {
    console.log("PASS  the `hidden` attribute cannot be overridden by a class");
  }
}

// ── step 1 reflects an uploaded campaign ────────────────────────────

{
  // The workbook carries its whole design; uploading it should show that
  // design rather than leave the form on the page defaults.
  RESPONSES = {
    inspect_workbook: inspected([1, 2, 3, 4], "central-composite"),
    run_fit: () => FIT,
    design_spec_from_workbook: () => ({
      ok: true,
      supported: true,
      template: "central-composite",
      factors: [
        { name: "T", low: 300, high: 400 },
        { name: "P", low: 1, high: 5 },
      ],
      options: { response: "yield", seed: 7, center_points: 4, alpha: "face" },
    }),
  };
  // Start from a different template, so "it was already right" cannot pass.
  nodes["template"].value = "box-behnken";
  await acceptFile(file("central-composite-campaign.xlsx"));

  const source = nodes["design-source"];
  const problems = [];
  if (source.hidden) problems.push("the step 1 note is hidden");
  if (!source.visibleText.includes("central-composite-campaign.xlsx")) {
    problems.push("the note does not name the workbook");
  }
  if (nodes["template"].value !== "central-composite") {
    problems.push(`template is ${nodes["template"].value}`);
  }
  if (state.factorRows?.length !== 2 || state.factorRows[1]?.name !== "P") {
    problems.push(`factor rows are ${JSON.stringify(state.factorRows)}`);
  }
  if (Number(state.factorRows?.[0]?.high) !== 400) problems.push("factor bounds were not applied");
  const centerPoints = document.getElementById("opt-center_points").value;
  if (String(centerPoints) !== "4") problems.push(`center_points is ${centerPoints}`);
  if (document.getElementById("opt-response").value !== "yield") {
    problems.push("the response name was not applied");
  }
  if (problems.length) {
    failures++;
    console.log(`FAIL  uploading a workbook fills step 1 in\n      ${problems.join("\n      ")}`);
  } else {
    console.log("PASS  uploading a workbook fills step 1 in");
  }
}

{
  // A campaign this page cannot rebuild says so instead of silently leaving
  // a form that describes something else.
  RESPONSES = {
    inspect_workbook: inspected([1, 2, 3, 4]),
    run_fit: () => FIT,
    design_spec_from_workbook: () => ({
      ok: true,
      supported: false,
      reason: "this campaign was built from a Python module rather than a template",
    }),
  };
  await acceptFile(file("module-campaign.xlsx"));
  const shown = nodes["design-source"].visibleText;
  if (!shown.includes("Not filled in") || !shown.includes("Python module")) {
    failures++;
    console.log(`FAIL  an unsupported campaign explains itself (shown: "${shown}")`);
  } else {
    console.log("PASS  an unsupported campaign explains itself");
  }
}

// ── the analysis must not depend on a frame ever being painted ──────

{
  // A hidden, backgrounded or occluded tab gets no requestAnimationFrame
  // callbacks at all — and switching back to Excel right after dropping a
  // workbook is the normal thing to do. If progress waits on a frame, the page
  // simply stops, which reads as a hang.
  const realRaf = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = () => {}; // never fires, like a hidden tab
  RESPONSES = { inspect_workbook: inspected([1, 2, 3, 4]), run_fit: () => FIT };
  const finished = await Promise.race([
    acceptFile(file("campaign.xlsx")).then(() => true),
    new Promise((r) => setTimeout(() => r(false), 3000)),
  ]);
  globalThis.requestAnimationFrame = realRaf;
  const shown = nodes["analyze-result"].visibleText;
  if (!finished || !shown.includes("Fitted coefficients")) {
    failures++;
    console.log("FAIL  the analysis runs in a tab that never paints a frame");
  } else if (!nodes["analyze-status"].hidden || state.analyzing) {
    failures++;
    console.log("FAIL  the analysis left the page marked busy");
  } else {
    console.log("PASS  the analysis runs in a tab that never paints a frame");
  }
}

{
  // A drop landing mid-analysis is refused, but visibly: silently ignoring it
  // is indistinguishable from a broken page.
  RESPONSES = { inspect_workbook: inspected([1, 2, 3, 4]), run_fit: () => FIT };
  const first = acceptFile(file("first.xlsx"));
  await acceptFile(file("second.xlsx"));
  const busy = nodes["analyze-status-text"].textContent;
  await first;
  if (!busy.includes("second.xlsx") || !busy.includes("Still working")) {
    failures++;
    console.log(`FAIL  a drop during the analysis says nothing (status: "${busy}")`);
  } else {
    console.log("PASS  a drop during the analysis is refused out loud");
  }
}

// ── analytics events, when there is anything listening ──────────────

{
  // Everything above ran with no `gtag` on globalThis, which is the no-op path
  // — that it got this far is the check that a blocked or unconfigured tag
  // cannot take the app down with it. Now the other side: the events have to
  // carry the design type, or they answer nothing.
  const sent = [];
  globalThis.gtag = (kind, name, params) => sent.push([kind, name, params?.template]);

  RESPONSES = { inspect_workbook: inspected([1, 2, 3, 4], "latin-hypercube"), run_fit: () => FIT };
  await acceptFile(file("done.xlsx"));

  RESPONSES = { inspect_workbook: () => ({ ok: false, error: "not a campaign workbook" }) };
  await acceptFile(file("stranger.xlsx"));

  delete globalThis.gtag;

  const seen = sent.map((e) => e.filter(Boolean).join(":"));
  // GA4 rejects an event name with a hyphen in it, so this is not cosmetic.
  const badName = sent.find(([, name]) => !/^[a-z][a-z0-9_]{0,39}$/.test(name));
  if (!seen.includes("event:analyze:latin-hypercube") || !seen.includes("event:analyze_rejected")) {
    failures++;
    console.log(`FAIL  the analytics events are wrong (got: ${seen.join(", ") || "none"})`);
  } else if (badName) {
    failures++;
    console.log(`FAIL  "${badName[1]}" is not a legal GA4 event name`);
  } else {
    console.log("PASS  an analysis and a refusal are each counted, by design type");
  }
}

console.error = consoleError;
cleanup();
if (missing.length) {
  failures++;
  console.log(`FAIL  app.js reaches for ids index.html does not have: ${[...new Set(missing)].join(", ")}`);
}
console.log(failures ? `\n${failures} CHECK(S) FAILED` : "\nALL CHECKS PASSED");
process.exit(failures === 0 ? 0 : 1);
