"use strict";
// Everything from a document is untrusted, so text is only ever set with textContent, never as HTML.

const main = document.getElementById("main");
const userArea = document.getElementById("user-area");
let refreshTimer = null;

const FIELD_NAMES = {
  invoice_number: "Invoice number", invoice_date: "Invoice date", due_date: "Due date",
  vendor_name: "Vendor", bill_to: "Bill to", currency: "Currency",
  subtotal: "Subtotal", tax: "Tax", cgst: "CGST", sgst: "SGST", igst: "IGST",
  shipping: "Shipping", discount: "Discount", rounding: "Rounding", deposit: "Deposit paid",
  grand_total: "Grand total", line_items: "Line items",
};
const STATUS_NAMES = {
  queued: "Waiting", running: "Processing", accepted: "Accepted",
  needs_review: "Needs review", reviewed: "Reviewed", failed: "Failed",
};

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child);
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (response.status === 401 && path !== "/api/login") {
    showLogin();
    throw new Error("Please log in.");
  }
  if (!response.ok) {
    let message = "Something went wrong. Please try again.";
    try {
      const body = await response.json();
      if (typeof body.detail === "string") message = body.detail;
    } catch { /* The server's reply wasn't JSON, so the general message stands. */ }
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

function clearRefresh() {
  if (refreshTimer) clearTimeout(refreshTimer);
  refreshTimer = null;
}

function statusTag(status) {
  return el("span", { class: `tag ${status}` }, STATUS_NAMES[status] || status);
}

function showLogin() {
  clearRefresh();
  userArea.replaceChildren();
  const username = el("input", { autocomplete: "username", required: true, autofocus: true });
  const password = el("input", { type: "password", autocomplete: "current-password", required: true });
  const message = el("p", { class: "error", role: "alert" });
  const form = el("form", {
    class: "panel login",
    onsubmit: async (event) => {
      event.preventDefault();
      message.textContent = "";
      try {
        await api("/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username: username.value, password: password.value }),
        });
        start();
      } catch (error) {
        message.textContent = error.message;
      }
    },
  },
    el("h2", {}, "Log in"),
    el("label", {}, el("span", {}, "Username"), username),
    el("label", {}, el("span", {}, "Password"), password),
    el("button", { class: "primary", type: "submit" }, "Log in"),
    message,
  );
  main.replaceChildren(form);
}

async function uploadFiles(files, message) {
  for (const file of files) {
    message.textContent = `Uploading ${file.name}...`;
    const body = new FormData();
    body.append("file", file);
    try {
      await api("/api/jobs", { method: "POST", body });
      message.textContent = "";
    } catch (error) {
      message.textContent = `${file.name}: ${error.message}`;
      return;
    }
  }
}

// Searches the person's own accepted and reviewed invoices. The totals are added up by the server, not by a language model.
function searchPanel() {
  const vendor = el("input", { placeholder: "Vendor, for example Acme", "aria-label": "Vendor" });
  const from = el("input", { type: "date", "aria-label": "From date" });
  const to = el("input", { type: "date", "aria-label": "To date" });
  const question = el("input", { placeholder: "Or ask: what did Acme invoice us last quarter?", "aria-label": "Question" });
  const message = el("p", { class: "error", role: "alert" });
  const summary = el("p", { class: "small" });
  const rows = el("tbody");
  const askRow = el("div", { class: "row", style: "margin-top:10px;display:none" },
    el("div", { style: "flex:1;min-width:220px" }, question),
    el("button", { type: "button", onclick: () => run(true) }, "Ask"),
  );
  api("/api/me").then((me) => { if (me.can_ask) askRow.style.display = "flex"; }).catch(() => {});

  async function run(useQuestion) {
    message.textContent = "";
    try {
      let result;
      if (useQuestion) {
        result = await api("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question.value }),
        });
        const f = result.filters;
        summary.textContent = `${result.summary} (Understood as: vendor ${f.vendor || "any"}, from ${f.date_from || "any date"}, to ${f.date_to || "any date"}.)`;
      } else {
        const query = new URLSearchParams();
        if (vendor.value) query.set("vendor", vendor.value);
        if (from.value) query.set("date_from", from.value);
        if (to.value) query.set("date_to", to.value);
        result = await api(`/api/invoices?${query}`);
        summary.textContent = result.summary;
      }
      rows.replaceChildren(...result.invoices.map((invoice) => el("tr", { class: "link", onclick: () => showJob(invoice.job_id) },
        el("td", {}, invoice.vendor_name),
        el("td", {}, invoice.invoice_number ?? ""),
        el("td", {}, invoice.invoice_date ?? ""),
        el("td", { class: "num" }, `${invoice.grand_total ?? ""} ${invoice.currency ?? ""}`),
      )));
    } catch (error) {
      message.textContent = error.message;
    }
  }

  return el("div", { class: "panel" },
    el("div", { class: "row" },
      el("div", { style: "flex:2;min-width:180px" }, vendor),
      el("div", { style: "flex:1;min-width:140px" }, from),
      el("div", { style: "flex:1;min-width:140px" }, to),
      el("button", { class: "primary", type: "button", onclick: () => run(false) }, "Search"),
    ),
    askRow, message, summary,
    el("div", { class: "scroll" }, el("table", {}, rows)),
  );
}

async function showList() {
  clearRefresh();
  const input = el("input", { type: "file", accept: "application/pdf,.pdf", multiple: true, style: "display:none" });
  const message = el("p", { class: "error", role: "alert" });
  const rows = el("tbody");
  const drop = el("div", {
    class: "drop",
    ondragover: (event) => { event.preventDefault(); drop.classList.add("over"); },
    ondragleave: () => drop.classList.remove("over"),
    ondrop: async (event) => {
      event.preventDefault();
      drop.classList.remove("over");
      await uploadFiles(event.dataTransfer.files, message);
      refresh();
    },
  },
    el("button", { class: "primary", type: "button", onclick: () => input.click() }, "Upload PDFs"),
    el("span", { class: "muted" }, " or drop them here"),
  );
  input.addEventListener("change", async () => {
    await uploadFiles(input.files, message);
    input.value = "";
    refresh();
  });

  main.replaceChildren(
    el("h2", {}, "Upload"),
    el("div", { class: "panel" }, drop, input, message),
    el("h2", {}, "Search your invoices"),
    searchPanel(),
    el("h2", {}, "Your documents"),
    el("div", { class: "panel scroll" },
      el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "File"), el("th", {}, "Status"), el("th", { class: "num" }, "Pages"),
          el("th", { class: "hide-narrow" }, "Uploaded"),
        )),
        rows,
      ),
    ),
  );

  async function refresh() {
    clearRefresh();
    try {
      const jobs = await api("/api/jobs");
      rows.replaceChildren(...(jobs.length ? jobs.map((job) => el("tr", { class: "link", onclick: () => showJob(job.id) },
        el("td", {}, job.filename, job.error ? el("div", { class: "small muted" }, job.error) : null),
        el("td", {}, statusTag(job.status), job.reused ? el("div", { class: "small muted" }, "Earlier result reused") : null),
        el("td", { class: "num" }, String(job.page_count)),
        el("td", { class: "hide-narrow muted small" }, new Date(job.created_at).toLocaleString()),
      )) : [el("tr", {}, el("td", { colspan: 4, class: "muted" }, "Nothing uploaded yet."))]));
      if (jobs.some((job) => job.status === "queued" || job.status === "running")) {
        refreshTimer = setTimeout(refresh, 2000);
      }
    } catch (error) {
      message.textContent = error.message;
    }
  }
  refresh();
}

function displayValue(value) {
  if (value === null || value === undefined || value === "") return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

// An editable list of line items. The server checks every value again, so this only collects what the person typed.
function itemsEditor(items) {
  const keys = ["description", "quantity", "unit_price", "amount"];
  const lines = (items || []).map((item) => ({
    description: item.description ?? "", quantity: item.quantity ?? "", unit_price: item.unit_price ?? "", amount: item.amount ?? "",
  }));
  const original = JSON.stringify(lines);
  const body = el("tbody");

  function draw() {
    body.replaceChildren(...lines.map((line, index) => el("tr", {},
      ...keys.map((key) => el("td", { class: key === "description" ? null : "num" },
        el("input", {
          value: line[key],
          "aria-label": `${key.replace("_", " ")} on line ${index + 1}`,
          oninput: (event) => { line[key] = event.target.value; },
        }))),
      el("td", {}, el("button", {
        type: "button", "aria-label": `Remove line ${index + 1}`,
        onclick: () => { lines.splice(index, 1); draw(); },
      }, "Remove")),
    )));
  }
  draw();

  const header = ["Description", "Qty", "Unit price", "Amount", ""];
  const node = el("div", {},
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, header.map((name, index) => el("th", { class: index > 0 && index < 4 ? "num" : null }, name)))),
      body,
    )),
    el("button", {
      type: "button", style: "margin-top:8px",
      onclick: () => { lines.push({ description: "", quantity: "", unit_price: "", amount: "" }); draw(); },
    }, "Add line"),
  );
  return {
    node,
    changed: () => JSON.stringify(lines) !== original,
    // A line the person left completely empty is not a line.
    value: () => lines.filter((line) => keys.some((key) => String(line[key]).trim() !== "")).map((line) => ({ ...line })),
  };
}

function gridTable(rows) {
  return el("div", { class: "scroll" }, el("table", {},
    el("tbody", {}, rows.map((row) => el("tr", {}, row.map((cell) => el("td", {}, cell))))),
  ));
}

function fieldsPanel(job, fields, message) {
  const inputs = {};
  let itemEditor = null;
  const rows = Object.entries(FIELD_NAMES).map(([name, title]) => {
    const field = fields[name];
    if (!field) return null;
    const isItems = name === "line_items";
    let value = null;
    if (isItems) {
      itemEditor = itemsEditor(field.value);
      value = itemEditor.node;
    } else {
      inputs[name] = el("input", { value: displayValue(field.value), "aria-label": title, placeholder: "Not found" });
    }
    return el("tr", { class: field.needs_review ? "field-review" : null },
      el("td", {}, title),
      el("td", {},
        value || inputs[name],
        field.suggested_value !== null && field.suggested_value !== undefined
          ? el("div", { class: "suggest" }, `The model suggests: ${displayValue(field.suggested_value)} `,
              isItems ? null : el("button", { type: "button", onclick: () => { inputs[name].value = displayValue(field.suggested_value); } }, "Use this"))
          : null,
        (field.review_reasons || []).map((reason) => el("div", { class: "reason" }, reason)),
      ),
      el("td", { class: "small muted hide-narrow" },
        field.page ? `Page ${field.page}` : "",
        field.source ? el("div", {}, ({ text_layer: "PDF text", ocr: "OCR", mixed: "PDF text and OCR", llm: "Language model", person: "Person" })[field.source] || field.source) : null,
      ),
      el("td", { class: "num small" }, field.confidence ? field.confidence.toFixed(2) : ""),
    );
  });

  async function save(finish) {
    message.textContent = "";
    const edits = [];
    const approve = [];
    for (const [name, input] of Object.entries(inputs)) {
      const field = fields[name];
      if (input.value !== displayValue(field.value)) {
        edits.push({ name, value: input.value === "" ? null : input.value });
      } else if (field.needs_review) {
        approve.push(name);
      }
    }
    // Line items are sent only if the person changed them. Otherwise, a flagged list is confirmed as it stands.
    if (itemEditor && itemEditor.changed()) {
      edits.push({ name: "line_items", value: itemEditor.value() });
    } else if (fields.line_items && fields.line_items.needs_review) {
      approve.push("line_items");
    }
    try {
      await api(`/api/jobs/${job.id}/review`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ edits, approve, finish }),
      });
      showJob(job.id);
    } catch (error) {
      message.textContent = error.message;
    }
  }

  return el("div", { class: "panel" },
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "Field"), el("th", {}, "Value"), el("th", { class: "hide-narrow" }, "Source"), el("th", { class: "num" }, "Confidence"))),
      el("tbody", {}, rows),
    )),
    el("div", { class: "row", style: "margin-top:14px" },
      el("button", { class: "primary", type: "button", onclick: () => save(true) }, "Confirm all and finish"),
      el("span", { class: "small muted" }, "Your corrections are logged. The original machine result is kept."),
    ),
  );
}

function pagesPanel(job, pages) {
  return pages.map((page) => el("div", {},
    el("h2", {}, `Page ${page.number}`),
    el("div", { class: "panel" },
      el("p", { class: "small muted", style: "margin-top:0" }, (page.notes || []).join(". ")),
      page.regions.map((region) => {
        if (region.type === "table" && region.table) {
          return el("div", { style: "margin-bottom:14px" }, el("div", { class: "kind" }, "Table"), gridTable(region.table.rows));
        }
        if (region.type === "chart") {
          const name = region.image ? region.image.split("/").pop() : null;
          return el("div", { style: "margin-bottom:14px" },
            el("div", { class: "kind" }, "Chart: saved as an image, values not extracted"),
            name ? el("img", { class: "chart", alt: "Chart from this page", src: `/api/jobs/${job.id}/charts/${encodeURIComponent(name)}` }) : null,
            region.labels && region.labels.length ? el("div", { class: "small muted" }, `Labels read: ${region.labels.join(", ")}`) : null,
          );
        }
        if (!region.text) return null;
        return el("div", {}, el("div", { class: "kind" }, region.type), el("div", { class: "region-text" }, region.text));
      }),
      page.flagged_words && page.flagged_words.length
        ? el("div", {}, el("div", { class: "kind" }, "Words outside any region"), el("div", { class: "region-text" }, page.flagged_words.map((word) => word.text).join(" ")))
        : null,
    ),
  ));
}

async function showJob(id) {
  clearRefresh();
  const message = el("p", { class: "error", role: "alert" });
  let job;
  try {
    job = await api(`/api/jobs/${id}`);
  } catch (error) {
    main.replaceChildren(el("button", { onclick: showList }, "Back"), el("p", { class: "error" }, error.message));
    return;
  }
  const back = el("button", { type: "button", onclick: showList }, "Back");
  const head = el("div", { class: "row spread" },
    el("div", { class: "row" }, back, el("strong", {}, job.filename), statusTag(job.status)),
    el("div", { class: "row" },
      job.result ? el("a", { class: "button", href: `/api/jobs/${job.id}/result.json`, download: `${job.filename}.json` }, "Download JSON") : null,
      job.status !== "running" ? el("button", {
        type: "button",
        onclick: async () => {
          if (!confirm(`Delete ${job.filename} and its result? This can't be undone.`)) return;
          try { await api(`/api/jobs/${job.id}`, { method: "DELETE" }); showList(); }
          catch (error) { message.textContent = error.message; }
        },
      }, "Delete") : null,
    ),
  );

  if (!job.result) {
    const text = job.status === "failed" ? (job.error || "This file couldn't be processed.") : "This document is still being processed.";
    main.replaceChildren(head, el("div", { class: "panel", style: "margin-top:14px" }, text), message);
    if (job.status === "queued" || job.status === "running") refreshTimer = setTimeout(() => showJob(id), 2000);
    return;
  }

  const machine = job.result.machine;
  const fields = job.result.reviewed ? job.result.reviewed.fields : machine.fields;
  // replaceChildren would print a null or an array as text, so the parts are wrapped by el() first.
  main.replaceChildren(el("div", {},
    head,
    message,
    el("h2", {}, "Fields"),
    fieldsPanel(job, fields, message),
    machine.llm && machine.llm.calls
      ? el("p", { class: "small muted" }, `The language model was used ${machine.llm.calls} times, estimated cost $${machine.llm.estimated_cost_usd.toFixed(4)}.`)
      : null,
    job.history && job.history.length
      ? el("div", {}, el("h2", {}, "Change history"), el("div", { class: "panel small" }, job.history.map((entry) => {
          const when = new Date(entry.at).toLocaleString();
          const title = FIELD_NAMES[entry.field] || entry.field;
          const before = displayValue(entry.old_value);
          const after = displayValue(entry.new_value);
          return el("div", {}, before === after
            ? `${when}: ${entry.username} confirmed ${title}${entry.field === "line_items" ? "" : ` as ${after || "empty"}`}`
            : `${when}: ${entry.username} changed ${title} from ${before || "empty"} to ${after || "empty"}`);
        })))
      : null,
    pagesPanel(job, machine.pages),
  ));
}

async function start() {
  try {
    const me = await api("/api/me");
    userArea.replaceChildren(
      el("span", { class: "muted" }, me.username),
      el("button", { type: "button", onclick: async () => { await api("/api/logout", { method: "POST" }); showLogin(); } }, "Log out"),
    );
    showList();
  } catch { /* api() has already shown the login page. */ }
}

start();
