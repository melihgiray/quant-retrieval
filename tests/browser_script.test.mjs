import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../src/quant_retrieval/serve/static/app.js", import.meta.url), "utf8");

function browserFixture(search = "") {
  const requests = [];
  const status = { textContent: "" };
  const results = {
    children: [],
    replaceChildren(...children) { this.children = children; },
  };
  const input = { value: "" };
  const submit = { disabled: false };
  const attributes = new Map();
  const form = {
    addEventListener() {},
    querySelector: () => submit,
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: (name) => attributes.delete(name),
  };
  const listeners = {};
  const elements = { "#search-form": form, "#query": input, "#status": status, "#results": results };
  const document = {
    querySelector: (selector) => elements[selector],
    querySelectorAll: () => [],
    createElement: () => ({
      children: [],
      append(...children) { this.children.push(...children); },
    }),
  };
  const fetch = (url, options) => new Promise((resolve) => requests.push({ url, options, resolve }));
  const window = {
    location: new URL(`https://demo.example/${search}`),
    history: {
      pushState(state, title, url) { window.location = new URL(url); },
    },
    addEventListener: (name, listener) => { listeners[name] = listener; },
  };
  const context = vm.createContext({ document, fetch, window, AbortController, URL, URLSearchParams });
  vm.runInContext(source, context);
  return { context, requests, status, results, input, window, submit, attributes, listeners };
}

function response(query) {
  return {
    ok: true,
    json: async () => ({
      query,
      elapsed_ms: 1,
      results: [{ answer_id: 10, text: "answer", url: "https://example.com/10" }],
    }),
  };
}

test("an older response cannot replace newer search results", async () => {
  const browser = browserFixture();
  const first = vm.runInContext('search("first")', browser.context);
  const second = vm.runInContext('search("second")', browser.context);
  assert.equal(browser.requests[0].options.signal.aborted, true);

  browser.requests[1].resolve(response("second"));
  await second;
  browser.requests[0].resolve(response("first"));
  await first;

  assert.match(browser.status.textContent, /second/);
  assert.equal(browser.results.children.length, 1);
});

test("a rejected question is not reported as a service outage", async () => {
  const browser = browserFixture();
  const pending = vm.runInContext('search("invalid")', browser.context);
  browser.requests[0].resolve({
    ok: false,
    status: 422,
    json: async () => ({ detail: "query must not be empty" }),
  });
  await pending;

  assert.equal(browser.status.textContent, "query must not be empty");
  assert.equal(browser.results.children.length, 0);
});

test("an empty result gets a useful status", async () => {
  const browser = browserFixture();
  const pending = vm.runInContext('search("unknown")', browser.context);
  browser.requests[0].resolve({
    ok: true,
    json: async () => ({ query: "unknown", elapsed_ms: 1, results: [] }),
  });
  await pending;

  assert.match(browser.status.textContent, /No answers found/);
  assert.equal(browser.results.children.length, 0);
});

test("a successful search is stored in the page URL", async () => {
  const browser = browserFixture();
  const pending = vm.runInContext('search("risk neutral pricing")', browser.context);
  browser.requests[0].resolve(response("risk neutral pricing"));
  await pending;

  assert.equal(browser.window.location.search, "?q=risk+neutral+pricing");
});

test("a query in the page URL runs on load", async () => {
  const browser = browserFixture("?q=delta+hedging");
  assert.equal(browser.input.value, "delta hedging");
  assert.match(browser.requests[0].url, /q=delta\+hedging/);
  browser.requests[0].resolve(response("delta hedging"));
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(browser.status.textContent, /delta hedging/);
});

test("the search form exposes its busy state", async () => {
  const browser = browserFixture();
  const pending = vm.runInContext('search("gamma")', browser.context);
  assert.equal(browser.submit.disabled, true);
  assert.equal(browser.attributes.get("aria-busy"), "true");

  browser.requests[0].resolve(response("gamma"));
  await pending;
  assert.equal(browser.submit.disabled, false);
  assert.equal(browser.attributes.has("aria-busy"), false);
});

test("answer links cannot control the search page", () => {
  const browser = browserFixture();
  const card = vm.runInContext(
    'resultCard({ answer_id: 10, text: "answer", url: "https://example.com/10" }, 0)',
    browser.context,
  );
  const link = card.children[1].children[1];

  assert.equal(link.target, "_blank");
  assert.equal(link.rel, "noopener noreferrer");
});

test("history navigation restores its query", async () => {
  const browser = browserFixture();
  const first = vm.runInContext('search("delta")', browser.context);
  browser.requests[0].resolve(response("delta"));
  await first;

  browser.window.location = new URL("https://demo.example/?q=gamma");
  browser.listeners.popstate();
  assert.equal(browser.input.value, "gamma");
  assert.match(browser.requests[1].url, /q=gamma/);
  browser.requests[1].resolve(response("gamma"));
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(browser.status.textContent, /gamma/);
});
