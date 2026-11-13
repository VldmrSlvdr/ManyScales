"""Write the questionnaire link page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ManyScales questionnaire codebook</title>
<style>
  :root { color-scheme: light; --ink: #1c1917; --muted: #57534e; --line: #e7e5e4; --paper: #fafaf9; --mark: #9a3412; }
  body { margin: 0; font: 15px/1.45 "Iowan Old Style", Palatino, Georgia, serif; color: var(--ink); background: var(--paper); }
  header { position: sticky; top: 0; background: #fff; border-bottom: 1px solid var(--line); padding: 12px 20px; z-index: 2; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  header p { margin: 0; color: var(--muted); max-width: 78rem; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; align-items: center; }
  button, input { font: inherit; }
  button { border: 1px solid var(--line); background: #fff; padding: 4px 10px; cursor: pointer; }
  button.on { background: #1c1917; color: #fff; }
  input { padding: 4px 8px; min-width: 16rem; border: 1px solid var(--line); }
  main { display: grid; grid-template-columns: 220px 1fr; min-height: calc(100vh - 110px); }
  nav { border-right: 1px solid var(--line); padding: 8px; position: sticky; top: 108px; align-self: start; max-height: calc(100vh - 120px); overflow: auto; }
  nav button { display: block; width: 100%; text-align: left; margin: 0 0 4px; }
  section { padding: 16px 22px 48px; }
  article { border-top: 1px solid var(--line); padding: 12px 0; }
  h2 { font-size: 18px; margin: 0 0 8px; }
  h3 { font-size: 16px; margin: 0 0 6px; }
  .meta { color: var(--muted); font-size: 13px; }
  code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin: 8px 0; }
  .cell { background: #fff; border: 1px solid var(--line); padding: 8px; }
  .cell b { display: block; font-size: 12px; color: var(--muted); font-weight: 600; }
  .ask { margin: 8px 0; }
  .mismatch { color: var(--mark); }
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; }
    nav { position: static; max-height: none; display: flex; gap: 4px; overflow: auto; }
    nav button { width: auto; }
    .grid { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>Questionnaire codebook</h1>
  <p>Each column in the survey export is tied to the wording stored in the Qualtrics file. Item rows also show the workbook original, the manual Japanese, and the LLM Japanese. A marked line means the text on the questionnaire is not the same string as that workbook cell.</p>
  <div class="controls">
    <button type="button" id="citizen" class="on">Citizens</button>
    <button type="button" id="expert">Experts</button>
    <input id="q" type="search" placeholder="Column, scale, or wording">
  </div>
</header>
<main>
  <nav id="scales"></nav>
  <section id="body"></section>
</main>
<script>
const DATA = __DATA__;
const state = { survey: "citizen", scale: "", query: "" };

function scales() {
  const rows = DATA[state.survey];
  const seen = new Map();
  for (const row of rows) {
    if (!row.scale_id) continue;
    if (!seen.has(row.scale_id)) seen.set(row.scale_id, row.scale_name || row.scale_id);
  }
  return [...seen.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

function renderNav() {
  const nav = document.getElementById("scales");
  nav.innerHTML = "";
  for (const [scaleId, name] of scales()) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = scaleId + " " + name;
    button.className = scaleId === state.scale ? "on" : "";
    button.addEventListener("click", () => { state.scale = scaleId; render(); });
    nav.appendChild(button);
  }
}

function matches(row) {
  if (state.scale && row.scale_id !== state.scale) return false;
  if (!state.query) return true;
  const hay = [row.column, row.scale_name, row.stem, row.shown_text, row.original_en, row.human_ja, row.llm_ja, row.dimension].join(" ").toLowerCase();
  return hay.includes(state.query);
}

function block(title, text) {
  const cell = document.createElement("div");
  cell.className = "cell";
  const label = document.createElement("b");
  label.textContent = title;
  cell.appendChild(label);
  cell.appendChild(document.createTextNode(text || "—"));
  return cell;
}

function ask(row) {
  const div = document.createElement("div");
  div.className = "ask";
  const code = document.createElement("code");
  code.textContent = row.column;
  div.appendChild(code);
  const note = document.createElement("div");
  note.textContent = row.stem || "";
  div.appendChild(note);
  if (row.shown_text && row.shown_text !== row.stem) {
    const shown = document.createElement("div");
    shown.textContent = "Item as shown: " + row.shown_text;
    div.appendChild(shown);
  }
  if (row.role === "item_problem" && row.dimension) {
    const dim = document.createElement("div");
    dim.textContent = "Checkbox: " + row.dimension;
    div.appendChild(dim);
  } else if (row.dimension) {
    const dim = document.createElement("div");
    dim.className = "meta";
    dim.textContent = row.dimension;
    div.appendChild(dim);
  }
  if (row.choices) {
    const choices = document.createElement("div");
    choices.className = "meta";
    choices.textContent = row.choices;
    div.appendChild(choices);
  }
  const column = row.column || "";
  if (column.endsWith("_preference") || column.endsWith("_llm_guess") || column.endsWith("_comparison_table")) {
    const slot = document.createElement("div");
    slot.className = "meta";
    slot.textContent = "In this comparison table, 翻訳1 is the human Japanese column and 翻訳2 is the LLM Japanese column. trans_cond (human or ai) records which version the respondent answered first.";
    div.appendChild(slot);
  }
  if (row.item_id && row.matches_workbook === false && (row.role === "item_response" || row.role === "item_problem")) {
    const mark = document.createElement("div");
    mark.className = "mismatch";
    mark.textContent = "Questionnaire text differs from the workbook cell for this condition.";
    div.appendChild(mark);
  }
  return div;
}

function render() {
  if (!state.scale) {
    const first = scales()[0];
    state.scale = first ? first[0] : "";
  }
  renderNav();
  const body = document.getElementById("body");
  body.innerHTML = "";
  const rows = DATA[state.survey].filter(matches);
  const title = document.createElement("h2");
  const named = rows.find(row => row.scale_name);
  title.textContent = (state.scale || "Survey") + (named ? " " + named.scale_name : "");
  body.appendChild(title);
  const lead = document.createElement("p");
  lead.className = "meta";
  lead.textContent = rows.length + " columns on this view.";
  body.appendChild(lead);
  const scaleRows = rows.filter(row => !row.item_id);
  if (scaleRows.length) {
    const heading = document.createElement("h3");
    heading.textContent = "Scale-level questions";
    body.appendChild(heading);
    for (const row of scaleRows) body.appendChild(ask(row));
  }
  const items = new Map();
  for (const row of rows) {
    if (!row.item_id) continue;
    if (!items.has(row.item_id)) items.set(row.item_id, []);
    items.get(row.item_id).push(row);
  }
  for (const itemId of [...items.keys()].sort()) {
    const group = items.get(itemId);
    const article = document.createElement("article");
    const heading = document.createElement("h3");
    heading.textContent = "Item " + itemId;
    article.appendChild(heading);
    const grid = document.createElement("div");
    grid.className = "grid";
    const sample = group[0];
    grid.appendChild(block("Original English", sample.original_en));
    grid.appendChild(block("Manual Japanese", sample.human_ja));
    grid.appendChild(block("LLM Japanese", sample.llm_ja));
    article.appendChild(grid);
    for (const row of group) article.appendChild(ask(row));
    body.appendChild(article);
  }
}

document.getElementById("citizen").addEventListener("click", () => {
  state.survey = "citizen"; state.scale = "";
  document.getElementById("citizen").className = "on";
  document.getElementById("expert").className = "";
  render();
});
document.getElementById("expert").addEventListener("click", () => {
  state.survey = "expert"; state.scale = "";
  document.getElementById("expert").className = "on";
  document.getElementById("citizen").className = "";
  render();
});
document.getElementById("q").addEventListener("input", (event) => {
  state.query = event.target.value.trim().toLowerCase();
  render();
});
render();
</script>
</body>
</html>
"""


def write_codebook_page(records: dict[str, list[dict[str, Any]]], path: Path) -> None:
    payload = json.dumps(records, ensure_ascii=False).replace("<", "\\u003c")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PAGE.replace("__DATA__", payload), encoding="utf-8")
