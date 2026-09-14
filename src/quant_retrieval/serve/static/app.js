const form = document.querySelector("#search-form");
const input = document.querySelector("#query");
const status = document.querySelector("#status");
const results = document.querySelector("#results");

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
  link.rel = "noreferrer";
  link.textContent = `Read answer ${hit.answer_id} on Quant Stack Exchange`;

  content.append(answer, link);
  article.append(rank, content);
  return article;
}

async function search(query) {
  status.textContent = "Searching the full corpus…";
  results.replaceChildren();

  try {
    const parameters = new URLSearchParams({ q: query, k: "10" });
    const response = await fetch(`/search?${parameters}`);
    if (!response.ok) throw new Error("Search failed");
    const payload = await response.json();

    status.textContent = `${payload.results.length} answers in ${payload.elapsed_ms} ms for “${payload.query}”`;
    results.replaceChildren(...payload.results.map(resultCard));
  } catch (error) {
    status.textContent = "Search is unavailable. Try again in a moment.";
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
