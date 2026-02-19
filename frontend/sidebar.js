const statusBadge = document.getElementById("statusBadge");

function fmtScore(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "--";
  }
  return `${Number(value).toFixed(1)}%`;
}

export function setStatus(text, tone = "neutral") {
  statusBadge.textContent = text;
  statusBadge.dataset.tone = tone;
}

export function renderParcelDetails(data) {
  const category = data.risk_category || "Unknown";
  document.getElementById("parcelId").textContent = String(data.parcel_id ?? data.building_id ?? "--");
  document.getElementById("riskValue").textContent = fmtScore(data.final_violation_probability);
  document.getElementById("riskCategory").textContent = category;
  document.getElementById("riskCategory").dataset.tone = category.toLowerCase();

  document.getElementById("scoreVector").textContent = fmtScore(data.vector_risk_score);
  document.getElementById("scoreSatellite").textContent = fmtScore(data.satellite_encroachment_score);
  document.getElementById("scoreTemporal").textContent = fmtScore(data.temporal_growth_score);

  document.getElementById("encroachmentType").textContent = data.encroachment_type || data.vector_triggers || "none";
  document.getElementById("satEvidence").textContent = data.satellite_evidence || "none";
  document.getElementById("beforeAfter").textContent = data.before_after_image || data.temporal_evidence || "stable";
  document.getElementById("ruleNarrative").textContent = data.rule_narrative || "No specific rule narrative.";
  document.getElementById("legalNarrative").textContent = data.legal_narrative || "No legal narrative generated.";
}

