// The smallest DOM app.js will accept, plus a loader for app.js itself.
//
// Shared by test-dom.mjs (which pairs it with a stubbed Pyodide, for speed and
// for failure modes that are hard to provoke for real) and test-fuzz.mjs
// (which pairs it with real WebAssembly). Neither needs a browser.
//
// It is deliberately thin: enough of the DOM for the app to run and for a test
// to read back what a user would see, and no more. Where it does model
// something exactly, there is a reason in a comment.

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const WEB = path.dirname(fileURLToPath(import.meta.url));

export function installFakeDom() {
  const nodes = {};
  const missing = [];

  class El {
    constructor(tag = "div") {
      this.tagName = tag;
      this.children = [];
      this._text = "";
      this._html = "";
      this.classList = {
        _s: new Set(),
        add(c) { this._s.add(c); },
        remove(c) { this._s.delete(c); },
        contains(c) { return this._s.has(c); },
      };
      this.style = {};
      this.dataset = {};
      this.hidden = false;
      this.disabled = false;
      this.className = "";
      this.value = "";
      this.type = "";
      this.checked = false;
    }
    // An element becomes findable by id once it has one — the option inputs
    // step 1 builds are created this way, not written into index.html.
    set id(v) {
      this._id = v;
      nodes[v] = this;
    }
    get id() {
      return this._id ?? "";
    }
    set textContent(v) { this._text = String(v); this.children = []; }
    get textContent() { return this._text; }
    set innerHTML(v) { this._html = String(v); this.children = []; this._text = ""; }
    get innerHTML() { return this._html; }
    appendChild(c) { this.children.push(c); return c; }
    append(...cs) { this.children.push(...cs); }
    addEventListener() {}
    setAttribute() {}
    click() {} // the download anchor
    /** Attribute selectors only — enough for `[data-opt-key]`, which is how
     *  the design step reads its options back out of the form. */
    querySelectorAll(selector) {
      const m = /^\[([a-z-]+)\]$/i.exec(selector);
      if (!m) return [];
      const key = m[1].replace(/^data-/, "").replace(/-([a-z])/g, (_, c) => c.toUpperCase());
      const found = [];
      const walk = (el) => {
        for (const child of el.children) {
          if (child.dataset?.[key] !== undefined) found.push(child);
          walk(child);
        }
      };
      walk(this);
      return found;
    }
    /** Everything this subtree would render as text, tags stripped. */
    get visibleText() {
      if (this.hidden) return "";
      return [
        this._text,
        this._html.replace(/<[^>]+>/g, " "),
        ...this.children.map((c) => c.visibleText),
      ]
        .filter(Boolean)
        .join("\n");
    }
    /** The subtree's raw markup, for assertions about cell classes. */
    get markup() {
      if (this.hidden) return "";
      return [this._html, ...this.children.map((c) => c.markup)].filter(Boolean).join("\n");
    }
  }

  // Every id in index.html exists up front, so a lookup that lands in
  // `missing` means app.js reached for something the page does not have.
  const html = fs.readFileSync(path.join(WEB, "index.html"), "utf8");
  for (const [, id] of html.matchAll(/id="([^"]+)"/g)) nodes[id] = new El();

  globalThis.document = {
    getElementById(id) {
      if (!nodes[id]) {
        missing.push(id);
        nodes[id] = new El();
      }
      return nodes[id];
    },
    createElement: (tag) => new El(tag),
  };
  globalThis.requestAnimationFrame = (fn) => setTimeout(fn, 0);
  // The design step hands the finished workbook to the browser to download.
  globalThis.URL = globalThis.URL ?? {};
  globalThis.URL.createObjectURL = () => "blob:test";
  globalThis.URL.revokeObjectURL = () => {};

  return { nodes, missing, html, El };
}

/**
 * Load app.js with its boot() call removed, returning what a test drives.
 *
 * boot() downloads Pyodide from a CDN and installs the wheel; the tests supply
 * their own runtime instead. Everything else is the real module.
 */
export async function loadApp() {
  const src =
    fs.readFileSync(path.join(WEB, "app.js"), "utf8").replace(/boot\(\)\.catch\([\s\S]*$/, "") +
    "\nglobalThis.__app = { acceptFile, state, loadTemplates, generate, selectTemplate };\n";
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "discopt-doe-app-"));
  const shim = path.join(dir, "app.mjs");
  fs.writeFileSync(shim, src);
  await import(`file://${shim}`);
  return { ...globalThis.__app, cleanup: () => fs.rmSync(dir, { recursive: true, force: true }) };
}
