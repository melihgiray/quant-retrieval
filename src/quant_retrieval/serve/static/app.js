const form = document.querySelector("#search-form");
const input = document.querySelector("#query");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const submit = form.querySelector('button[type="submit"]');
let activeRequest = null;

function resultCard(hit, index) {
  const article = document.createElement("article");

  const rank = document.createElement("span");
  rank.className = "rank";
  rank.textContent = String(index + 1).padStart(2, "0");

  const content = document.createElement("div");
  const answer = document.createElement("p");
  answer.textContent = hit.text;

  const link = document.createElement("a");
  link.href = hit.url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = `Read answer ${hit.answer_id} on Quant Stack Exchange`;

  content.append(answer, link);
  article.append(rank, content);
  return article;
}

function rememberQuery(query) {
  const url = new URL(window.location.href);
  if (url.searchParams.get("q") === query) return;
  url.searchParams.set("q", query);
  window.history.pushState(null, "", url);
}

async function search(query) {
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  form.setAttribute("aria-busy", "true");
  submit.disabled = true;
  status.textContent = "Searching the full corpus…";
  results.replaceChildren();

  try {
    const parameters = new URLSearchParams({ q: query, k: "10" });
    const response = await fetch(`/search?${parameters}`, { signal: controller.signal });
    if (controller.signal.aborted) return;
    if (response.status === 422) {
      const problem = await response.json();
      if (controller.signal.aborted) return;
      status.textContent = typeof problem.detail === "string"
        ? problem.detail
        : "Check your question and try again.";
      return;
    }
    if (!response.ok) throw new Error("Search failed");
    const payload = await response.json();
    if (controller.signal.aborted) return;

    if (payload.results.length === 0) {
      rememberQuery(payload.query);
      status.textContent = `No answers found for “${payload.query}”. Try a different question.`;
      return;
    }
    rememberQuery(payload.query);
    const noun = payload.results.length === 1 ? "answer" : "answers";
    status.textContent = `${payload.results.length} ${noun} in ${payload.elapsed_ms} ms for “${payload.query}”`;
    results.replaceChildren(...payload.results.map(resultCard));
  } catch (error) {
    if (!controller.signal.aborted) {
      status.textContent = "Search is unavailable. Try again in a moment.";
    }
  } finally {
    if (activeRequest === controller) {
      activeRequest = null;
      form.removeAttribute("aria-busy");
      submit.disabled = false;
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const query = input.value.trim();
  if (query) search(query);
});

document.querySelectorAll("[data-query]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.query;
    search(input.value);
  });
});

function restoreQueryFromLocation() {
  const query = new URLSearchParams(window.location.search).get("q")?.trim();
  if (query) {
    input.value = query;
    search(query);
  } else {
    activeRequest?.abort();
    input.value = "";
    status.textContent = "";
    results.replaceChildren();
  }
}

window.addEventListener("popstate", restoreQueryFromLocation);
restoreQueryFromLocation();
