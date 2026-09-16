import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../src/quant_retrieval/serve/static/app.js", import.meta.url), "utf8");

function browserFixture() {
  const requests = [];
  const status = { textContent: "" };
  const results = {
    children: [],
    replaceChildren(...children) { this.children = children; },
  };
  const input = { value: "" };
  const form = { addEventListener() {} };
  const elements = { "#search-form": form, "#query": input, "#status": status, "#results": results };
  const document = {
    querySelector: (selector) => elements[selector],
    querySelectorAll: () => [],
    createElement: () => ({ append() {} }),
  };
  const fetch = (url, options) => new Promise((resolve) => requests.push({ url, options, resolve }));
  const context = vm.createContext({ document, fetch, AbortController, URLSearchParams });
  vm.runInContext(source, context);
  return { context, requests, status, results };
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
