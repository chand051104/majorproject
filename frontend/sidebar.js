const statusBadge = document.getElementById("statusBadge");
const parcelSubtextEl = document.querySelector(".parcel-header .subtext");
const temporalMetricLabelEl = document.getElementById("temporalMetricLabel");
const FETCH_WARNING_TOKENS = new Set(["sentinel_fetch_failed", "tile_fetch_failed", "no_imagery_source"]);

const COVERAGE_LIMIT = 60;
const COVERAGE_CRITICAL = 80;

function asNumber(value) {
  const parsed = Number(value);
  if (value === null || value === undefined || Number.isNaN(parsed)) return null;
  return parsed;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function score100(value) {
  const n = asNumber(value);
  if (n === null) return "--";
  const bounded = clamp(n, 0, 99.0);
  return `${bounded.toFixed(1)}/100`;
}

function formatPercent(value, { ratio = false, digits = 1, fallback = "--" } = {}) {
  const n = asNumber(value);
  if (n === null) return fallback;
  const pct = ratio ? n * 100 : n;
  return `${pct.toFixed(digits)}%`;
}

function formatPercentAuto(value, { digits = 1, fallback = "--" } = {}) {
  const n = asNumber(value);
  if (n === null) return fallback;
  const ratio = Math.abs(n) <= 1.2;
  return formatPercent(n, { ratio, digits, fallback });
}

function temporalDisplay(data) {
  const growthPct = asNumber(data.growth_rate_pct);
  if (growthPct !== null && Math.abs(growthPct) > 0.01) {
    return { label: "Temporal Change", value: `${growthPct.toFixed(1)}%` };
  }

  const growthRatio = asNumber(data.footprint_growth_ratio ?? data.growth_ratio);
  if (growthRatio !== null && Math.abs(growthRatio) > 0.0001) {
    return { label: "Temporal Change", value: `${(growthRatio * 100).toFixed(1)}%` };
  }

  const imageDelta = asNumber(data.temporal_image_delta);
  if (imageDelta !== null && imageDelta > 0) {
    return { label: "Temporal Change", value: `${(imageDelta * 100).toFixed(1)}%` };
  }

  const modelScore = asNumber(data.temporal_model_score ?? data.temporal_growth_score);
  if (modelScore !== null && modelScore > 0) {
    return { label: "Temporal Signal", value: `${modelScore.toFixed(0)} pts` };
  }

  return { label: "Temporal Change", value: "0.0%" };
}

function formatMeters(value, digits = 2) {
  const n = asNumber(value);
  if (n === null) return "--";
  return `${n.toFixed(digits)}m`;
}

function yesNo(value) {
  const n = asNumber(value);
  if (n === null) return value ? "Yes" : "No";
  return n > 0 ? "Yes" : "No";
}

function prettyToken(raw) {
  return String(raw || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (m) => m.toUpperCase());
}

function parseEvidence(value) {
  return String(value || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .filter((x) => x !== "none" && x !== "stable");
}

function hasFetchWarning(value) {
  return String(value || "")
    .split(",")
    .map((x) => x.trim())
    .some((x) => FETCH_WARNING_TOKENS.has(x));
}

function readableEvidence(value, fallback = "none") {
  const raw = String(value || "")
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .filter((x) => x !== "none" && x !== "stable")
    .filter((x) => !FETCH_WARNING_TOKENS.has(x))
    .map((x) => x.replaceAll("_", " "));
  if (!raw.length) {
    return hasFetchWarning(value) ? "partial satellite coverage" : fallback;
  }
  if (hasFetchWarning(value)) raw.push("partial satellite coverage");
  return raw.join(", ");
}

function centroidFromGeometry(geometry) {
  if (!geometry || !geometry.type || !geometry.coordinates) return null;
  const type = geometry.type;
  const coords = geometry.coordinates;

  if (type === "Point" && Array.isArray(coords) && coords.length >= 2) {
    return { lon: Number(coords[0]), lat: Number(coords[1]) };
  }

  const points = [];
  const pushPoint = (pt) => {
    if (Array.isArray(pt) && pt.length >= 2) {
      const lon = Number(pt[0]);
      const lat = Number(pt[1]);
      if (Number.isFinite(lon) && Number.isFinite(lat)) points.push([lon, lat]);
    }
  };

  if (type === "Polygon" && Array.isArray(coords)) {
    coords.forEach((ring) => Array.isArray(ring) && ring.forEach(pushPoint));
  } else if (type === "MultiPolygon" && Array.isArray(coords)) {
    coords.forEach((poly) => Array.isArray(poly) && poly.forEach((ring) => Array.isArray(ring) && ring.forEach(pushPoint)));
  } else if (type === "LineString" && Array.isArray(coords)) {
    coords.forEach(pushPoint);
  } else if (type === "MultiLineString" && Array.isArray(coords)) {
    coords.forEach((line) => Array.isArray(line) && line.forEach(pushPoint));
  }

  if (!points.length) return null;
  const sum = points.reduce((acc, [lon, lat]) => ({ lon: acc.lon + lon, lat: acc.lat + lat }), { lon: 0, lat: 0 });
  return { lon: sum.lon / points.length, lat: sum.lat / points.length };
}

function formatBasicAddress(data) {
  const explicitAddress = [
    data.full_address,
    data.address,
    data.street,
    data.road_name,
    data.locality,
    data.area_name,
    data.ward,
    data.city,
    data.state,
  ]
    .map((x) => (x ?? "").toString().trim())
    .filter(Boolean)
    .filter((v, i, arr) => arr.indexOf(v) === i);

  if (explicitAddress.length) return explicitAddress.join(", ");

  const center = centroidFromGeometry(data.geometry);
  if (center && Number.isFinite(center.lat) && Number.isFinite(center.lon)) {
    const lat = Math.abs(center.lat).toFixed(5);
    const lon = Math.abs(center.lon).toFixed(5);
    const latDir = center.lat >= 0 ? "N" : "S";
    const lonDir = center.lon >= 0 ? "E" : "W";
    return `Hyderabad, Telangana (${lat}°${latDir}, ${lon}°${lonDir})`;
  }

  return "Hyderabad, Telangana";
}

function buildSubtext(data) {
  const locality = [data.locality, data.area_name, data.ward].find((v) => String(v || "").trim());
  const adminBits = [locality, data.sector, data.block]
    .map((v) => (v ?? "").toString().trim())
    .filter(Boolean);
  const idBits = [
    data.parcel_id ? `Parcel ${data.parcel_id}` : null,
    data.building_id ? `Building ${data.building_id}` : null,
  ].filter(Boolean);
  const bits = [...adminBits, ...idBits];
  return bits.length ? bits.join(" • ") : "Violation Assessment Unit";
}

function violationsFromRecord(data) {
  const vector = parseEvidence(data.vector_triggers).map(prettyToken);
  const sat = parseEvidence(data.satellite_evidence)
    .filter((x) => !FETCH_WARNING_TOKENS.has(x))
    .map(prettyToken);
  const temporal = parseEvidence(data.temporal_evidence)
    .filter((x) => !FETCH_WARNING_TOKENS.has(x))
    .map(prettyToken);

  const parts = [];
  if (vector.length) parts.push(`Vector rules: ${vector.join(", ")}`);
  if (sat.length) parts.push(`Satellite indicators: ${sat.join(", ")}`);
  if (temporal.length) parts.push(`Temporal indicators: ${temporal.join(", ")}`);
  if (hasFetchWarning(data.temporal_evidence) || hasFetchWarning(data.satellite_evidence)) {
    parts.push("Satellite fetch had partial coverage; cached/fallback tiles were used for some locations");
  }

  if (!parts.length) return "No active violations detected for the selected location.";
  return parts.join(" | ");
}

function buildViolationAnalysis(data) {
  const riskCategory = (data.risk_category || "Unknown").toUpperCase();
  const finalRisk = score100(data.final_risk ?? data.final_violation_probability);
  const mlScore = asNumber(data.ml_score);

  const roadDistance = formatMeters(data.dist_road_m);
  const roadRequired = formatMeters(data.road_setback_required_m);
  const roadDeficit = formatMeters(data.road_setback_deficit_m);

  const lakeIntrusion = formatMeters(data.lake_buffer_intrusion_m);
  const nalaIntrusion = formatMeters(data.nala_buffer_intrusion_m);
  const canalIntrusion = formatMeters(data.canal_buffer_intrusion_m);

  const spillover = formatPercent(data.parcel_spillover_pct, { ratio: true });
  const coverage = formatPercent(data.parcel_coverage_ratio ?? data.built_up_ratio, { ratio: true });
  const footpathBuiltup = formatPercent(data.footpath_buffer_builtup_pct, { ratio: false });

  const temporal = temporalDisplay(data);

  const vectorScore = score100(data.vector_risk_score);
  const satScore = score100(data.satellite_encroachment_score);
  const temporalScore = score100(data.temporal_growth_score);

  const summaryLine = `Overall risk is ${riskCategory} (${finalRisk}).`;
  const confidenceLine = mlScore === null ? "Model confidence is not available." : `ML confidence (0–1): ${mlScore.toFixed(3)}.`;

  const lines = [
    "LOCATION SUMMARY",
    `Address: ${formatBasicAddress(data)}`,
    `Context: ${buildSubtext(data)}`,
    "",
    "RISK SNAPSHOT",
    `${summaryLine}`,
    `${confidenceLine}`,
    `Component scores: Vector ${vectorScore}, Satellite ${satScore}, Temporal ${temporalScore}`,
    "",
    "KEY VIOLATIONS DETECTED",
    `${violationsFromRecord(data)}`,
    "",
    "MEASUREMENTS (VERIFIABLE)",
    `Road distance: ${roadDistance} | Required setback: ${roadRequired} | Deficit: ${roadDeficit}`,
    `Water intrusion: Lake ${lakeIntrusion}, Nala ${nalaIntrusion}, Canal/Drain ${canalIntrusion}`,
    `Parcel spillover: ${spillover}`,
    `Parcel coverage: ${coverage} (limit ${COVERAGE_LIMIT}%, critical ${COVERAGE_CRITICAL}%)`,
    `Footpath built-up in buffer: ${footpathBuiltup}`,
    `Observed temporal metric: ${temporal.value}`,
    "",
    "INTERPRETATION",
    "Large setback deficits or parcel spillover indicate likely boundary/setback violations.",
    "Any lake/nala/canal intrusion increases legal and flood-risk severity.",
    "High coverage or multi-parcel overlap suggests overdevelopment pressure.",
    "Consistent temporal growth signals potential unauthorized expansion.",
    "",
    "HUMAN-READABLE REPORT",
    data.human_readable_violation_report || data.legal_narrative || "No legal narrative generated.",
  ];

  return lines.join("\n");
}

export function setStatus(text, tone = "neutral") {
  statusBadge.textContent = text;
  statusBadge.dataset.tone = tone;
}

export function renderParcelDetails(data) {
  const category = (data.risk_category || "Unknown").toLowerCase();

  document.getElementById("parcelId").textContent = formatBasicAddress(data);
  if (parcelSubtextEl) {
    parcelSubtextEl.textContent = buildSubtext(data);
  }

  document.getElementById("riskValue").textContent = score100(data.final_risk ?? data.final_violation_probability);
  document.getElementById("riskValue").dataset.tone = category;

  const chip = document.getElementById("riskCategory");
  chip.textContent = `${category.toUpperCase()} RISK`;
  chip.dataset.tone = category;

  document.getElementById("scoreVector").textContent = score100(data.vector_risk_score);
  document.getElementById("scoreSatellite").textContent = score100(data.satellite_encroachment_score);
  const temporal = temporalDisplay(data);
  if (temporalMetricLabelEl) temporalMetricLabelEl.textContent = temporal.label;
  document.getElementById("scoreTemporal").textContent = temporal.value;

  document.getElementById("encroachmentType").textContent = (data.encroachment_type || "none").replaceAll("_", " ");
  document.getElementById("satEvidence").textContent = readableEvidence(data.satellite_evidence, "none");
  document.getElementById("beforeAfter").textContent = readableEvidence(data.temporal_evidence, "stable");

  document.getElementById("ruleNarrative").textContent = violationsFromRecord(data);
  document.getElementById("legalNarrative").textContent =
    data.human_readable_violation_report || data.legal_narrative || "No legal narrative generated.";

  const analysisEl = document.getElementById("generatedViolationAnalysis");
  if (analysisEl) {
    analysisEl.textContent = buildViolationAnalysis(data);
  }
}
