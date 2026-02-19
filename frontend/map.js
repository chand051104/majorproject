import { renderParcelDetails, setStatus } from "./sidebar.js";

const API_BASE = window.URBANGUARD_API_BASE || "http://localhost:8000";
const DEFAULT_CENTER = [17.385, 78.4867];
const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView(DEFAULT_CENTER, 11);
const canvasRenderer = L.canvas({ padding: 0.5 });

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

let violationsLayer = null;

function colorForRisk(category) {
  if (category === "High") return "#da3a1b";
  if (category === "Medium") return "#f0a202";
  return "#2f9e44";
}

function styleForFeature(feature) {
  const category = feature.properties?.risk_category;
  return {
    color: colorForRisk(category),
    fillColor: colorForRisk(category),
    weight: 1.2,
    fillOpacity: 0.35,
  };
}

function radiusForScore(score) {
  const value = Number(score || 0);
  if (value >= 70) return 8;
  if (value >= 50) return 6.5;
  if (value >= 35) return 5.5;
  return 4.5;
}

async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(`API ${response.status}: ${await response.text()}`);
  }
  return response.json();
}

async function loadParcelDetails(parcelId) {
  setStatus(`Loading parcel ${parcelId}...`, "neutral");
  const payload = await fetchJson(`${API_BASE}/parcel/${encodeURIComponent(parcelId)}`);
  renderParcelDetails(payload);
  const tone = payload.risk_category === "High" ? "high" : payload.risk_category === "Medium" ? "medium" : "low";
  setStatus(`Parcel ${parcelId} loaded`, tone);
}

function bindFeatureEvents(arg1, arg2) {
  const feature = arg1?.properties ? arg1 : arg2;
  const layer = (arg1 && (typeof arg1.on === "function" || typeof arg1.bindPopup === "function")) ? arg1 : arg2;
  const props = feature?.properties || {};
  const parcelId = props.parcel_id ?? props.building_id;
  const score = Number(props.final_violation_probability || 0).toFixed(1);
  const type = props.encroachment_type || props.satellite_evidence || "none";
  if (layer && typeof layer.bindPopup === "function") {
    layer.bindPopup(
      `<strong>${parcelId ?? "Parcel"}</strong><br/>Violation Probability: ${score}%<br/>Signals: ${type}`
    );
  }

  if (!layer || typeof layer.on !== "function") {
    return;
  }

  layer.on("click", async () => {
    if (parcelId === null || parcelId === undefined) {
      setStatus("Parcel identifier missing in feature properties", "medium");
      return;
    }
    try {
      await loadParcelDetails(parcelId);
    } catch (error) {
      setStatus(`Failed to load parcel: ${error.message}`, "high");
    }
  });
}

async function loadViolations() {
  setStatus("Loading complete city results...", "neutral");
  if (violationsLayer) {
    violationsLayer.remove();
  }
  violationsLayer = L.layerGroup().addTo(map);

  const pageSize = 10000;
  let offset = 0;
  let total = null;
  let loaded = 0;
  let bounds = null;

  while (total === null || offset < total) {
    const payload = await fetchJson(
      `${API_BASE}/violations?limit=${pageSize}&offset=${offset}` +
        `&sort_by=final_violation_probability&descending=true&prioritize_satellite=true` +
        `&as_points=true&compact=true&min_probability=30`
    );
    total = payload.total_rows;
    const features = payload.features || [];
    const geojson = { type: "FeatureCollection", features };

    const batchLayer = L.geoJSON(geojson, {
      pointToLayer: (feature, latlng) => {
        const props = feature.properties || {};
        const category = props.risk_category;
        const score = props.final_violation_probability;
        return L.circleMarker(latlng, {
          radius: radiusForScore(score),
          color: colorForRisk(category),
          fillColor: colorForRisk(category),
          weight: 1.0,
          fillOpacity: 0.72,
          renderer: canvasRenderer,
        });
      },
      onEachFeature: (feature, layer) => bindFeatureEvents(feature, layer),
    });
    batchLayer.addTo(violationsLayer);
    if (features.length > 0) {
      const b = batchLayer.getBounds();
      bounds = bounds ? bounds.extend(b) : b;
    }

    loaded += features.length;
    offset += features.length;
    setStatus(`Loaded ${loaded.toLocaleString()} / ${total.toLocaleString()} parcels`, "neutral");
    if (features.length === 0) {
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  if (bounds && bounds.isValid()) {
    map.fitBounds(bounds, { padding: [20, 20] });
  }
  setStatus(`Loaded entire city: ${loaded.toLocaleString()} parcels`, "low");
}

async function runPipeline() {
  setStatus("Running city-scale fusion pipeline...", "neutral");
  await fetchJson(`${API_BASE}/pipeline/run?max_buildings=500000&async_run=true`, { method: "POST" });

  let running = true;
  while (running) {
    const status = await fetchJson(`${API_BASE}/pipeline/status`);
    running = Boolean(status.running);
    if (status.last_error) {
      throw new Error(status.last_error);
    }
    if (running) {
      setStatus("Pipeline in progress...", "neutral");
      await new Promise((resolve) => setTimeout(resolve, 3000));
    } else {
      const rows = status.last_report?.rows ?? 0;
      setStatus(`Pipeline completed (${rows} rows)`, "low");
    }
  }
  await loadViolations();
}

document.getElementById("runPipelineBtn").addEventListener("click", async () => {
  try {
    await runPipeline();
  } catch (error) {
    setStatus(`Pipeline failed: ${error.message}`, "high");
  }
});

loadViolations().catch((error) => {
  setStatus(`No results yet. Run pipeline first. ${error.message}`, "medium");
});
