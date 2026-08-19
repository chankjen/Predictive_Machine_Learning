const fmtPct = (value) => `${(value * 100).toFixed(1)}%`;

function bandClass(band) {
  return `band-${String(band).replace(" ", "-")}`;
}

function addChatLine(text, cls) {
  const log = document.getElementById("chatLog");
  const node = document.createElement("div");
  node.className = cls;
  node.textContent = text;
  log.appendChild(node);
  log.scrollTop = log.scrollHeight;
}

async function explainAsset(assetId) {
  const response = await fetch(`/api/explain/${assetId}`);
  const data = await response.json();
  document.getElementById("assetExplanation").textContent = data.narrative || data.error;
}

async function loadDashboard() {
  const response = await fetch("/api/summary");
  const data = await response.json();

  document.getElementById("failureRate").textContent = fmtPct(data.failureRate);
  document.getElementById("highRiskAsset").textContent = `Asset ${data.highRiskAsset}`;
  document.getElementById("highRiskProbability").textContent = fmtPct(data.highRiskProbability);

  const benchmark = data.metrics.filter((row) => row.failure_recall !== null);
  Plotly.newPlot(
    "metricsChart",
    [
      {
        x: benchmark.map((row) => row.model),
        y: benchmark.map((row) => row.failure_recall),
        name: "Failure recall",
        type: "bar",
        marker: { color: "#2563eb" },
      },
      {
        x: benchmark.map((row) => row.model),
        y: benchmark.map((row) => row.failure_precision),
        name: "Failure precision",
        type: "bar",
        marker: { color: "#1b7f5a" },
      },
    ],
    {
      margin: { l: 42, r: 12, t: 8, b: 96 },
      yaxis: { tickformat: ".0%", range: [0, 1.05] },
      legend: { orientation: "h", y: 1.12 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    },
    { displayModeBar: false, responsive: true }
  );

  const rows = document.getElementById("fleetRows");
  rows.innerHTML = "";
  data.fleet.slice(0, 10).forEach((asset) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${asset.asset_id}</td>
      <td>${asset.cycle}</td>
      <td>${fmtPct(asset.risk_probability)}</td>
      <td class="${bandClass(asset.risk_band)}">${asset.risk_band}</td>
      <td>${asset.primary_driver}</td>
      <td><button class="mini-btn" data-asset="${asset.asset_id}">Explain</button></td>
    `;
    rows.appendChild(tr);
  });

  document.querySelectorAll(".mini-btn").forEach((button) => {
    button.addEventListener("click", () => explainAsset(button.dataset.asset));
  });

  if (data.fleet.length) {
    explainAsset(data.fleet[0].asset_id);
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    button.classList.add("active");
    document.getElementById("evidenceImage").src = button.dataset.image;
  });
});

document.getElementById("chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;
  addChatLine(message, "user");
  input.value = "";
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  const data = await response.json();
  addChatLine(data.answer, "bot");
});

loadDashboard();
