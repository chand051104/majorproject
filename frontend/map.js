import { renderParcelDetails, setStatus } from "./sidebar.js?v=20260220c";

const API_BASE =
  window.URBANGUARD_API_BASE ||
  (() => {
    const { protocol, hostname, port, origin } = window.location;
    if (!hostname) return "http://127.0.0.1:8000";
    if (port === "8000") return origin;
    return `${protocol}//${hostname}:8000`;
  })();

const DEFAULT_CENTER = [17.385, 78.4867];
const STATIC_DATA_URL = "./data/violations_payload.json";
const AUTH_TOKEN_KEY = "urbanguard_auth_token";

const map = L.map("map", { zoomControl: true, preferCanvas: true }).setView(DEFAULT_CENTER, 11);
const canvasRenderer = L.canvas({ padding: 0.5 });
const PAGE_SIZE = Number(window.URBANGUARD_PAGE_SIZE || 2000);
const MIN_PROBABILITY = Number(window.URBANGUARD_MIN_PROBABILITY || 0);
const MAX_RENDERED = Number(window.URBANGUARD_MAX_RENDERED_PARCELS || 10000);

const runPipelineBtn = document.getElementById("runPipelineBtn");
const logoutBtn = document.getElementById("logoutBtn");
const userScopeBadge = document.getElementById("userScopeBadge");
const loginOverlay = document.getElementById("loginOverlay");
const loginForm = document.getElementById("loginForm");
const usernameInput = document.getElementById("usernameInput");
const passwordInput = document.getElementById("passwordInput");
const loginError = document.getElementById("loginError");
const toggleFlagBtn = document.getElementById("toggleFlagBtn");
const flagMeta = document.getElementById("flagMeta");

L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

let violationsLayer = null;
let authToken = localStorage.getItem(AUTH_TOKEN_KEY) || "";
let currentUser = null;
let selectedParcelId = null;
let selectedIsUnflagged = false;
let staticMode = false;
let staticPayload = null;
let staticFeatureById = new Map();

function colorForRisk(category) {
  if (category === "High") return "#da3a1b";
  if (category === "Medium") return "#f0a202";
  return "#2f9e44";
}

function riskTone(category) {
  if (category === "High") return "high";
  if (category === "Medium") return "medium";
  return "low";
}

function setRunButtonLoading(loading) {
  if (!runPipelineBtn) return;
  runPipelineBtn.disabled = Boolean(loading);
  runPipelineBtn.classList.toggle("is-loading", Boolean(loading));
  runPipelineBtn.textContent = loading ? "Running…" : "Run Pipeline";
}

function radiusForScore(score) {
  const value = Number(score || 0);
  if (value >= 70) return 8;
  if (value >= 50) return 6.5;
  if (value >= 35) return 5.5;
  return 4.5;
}

function setAuthToken(token) {
  authToken = token || "";
  if (authToken) {
    localStorage.setItem(AUTH_TOKEN_KEY, authToken);
  } else {
    localStorage.removeItem(AUTH_TOKEN_KEY);
  }
}

function authHeaders(initHeaders = {}) {
  const headers = new Headers(initHeaders || {});
  if (authToken) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }
  return headers;
}

function showLogin(errorMessage = "") {
  if (loginOverlay) loginOverlay.classList.remove("hidden");
  if (loginError) loginError.textContent = errorMessage;
  if (usernameInput) usernameInput.focus();
}

function hideLogin() {
  if (loginOverlay) loginOverlay.classList.add("hidden");
  if (loginError) loginError.textContent = "";
}

function applyUserState() {
  if (staticMode) {
    if (userScopeBadge) {
      userScopeBadge.hidden = false;
      userScopeBadge.textContent = "Static demo mode • precomputed output";
    }
    if (logoutBtn) logoutBtn.hidden = true;
    if (runPipelineBtn) {
      runPipelineBtn.disabled = true;
      runPipelineBtn.title = "Disabled in static mode";
    }
    return;
  }
  const role = currentUser?.role || "surveyor";
  const areaName = currentUser?.area_name || "Assigned area";
  const display = currentUser?.display_name || currentUser?.username || "User";
  if (userScopeBadge) {
    userScopeBadge.hidden = !currentUser;
    userScopeBadge.textContent = currentUser ? `${display} • ${areaName}` : "Not signed in";
  }
  if (logoutBtn) logoutBtn.hidden = !currentUser;
  if (runPipelineBtn) {
    const canRun = currentUser && role === "admin";
    runPipelineBtn.disabled = !canRun;
    runPipelineBtn.title = canRun ? "" : "Only admin can run full city pipeline";
  }
}

function updateFlagControls(detailPayload = null) {
  if (!toggleFlagBtn || !flagMeta) return;
  if (!detailPayload) {
    selectedParcelId = null;
    selectedIsUnflagged = false;
    toggleFlagBtn.disabled = true;
    toggleFlagBtn.dataset.mode = "unflag";
    toggleFlagBtn.textContent = "Unflag Violation";
    flagMeta.textContent = "Select a location to flag or unflag.";
    return;
  }

  selectedParcelId = String(detailPayload.parcel_id ?? detailPayload.building_id ?? "");
  selectedIsUnflagged = Boolean(detailPayload.is_unflagged);
  toggleFlagBtn.disabled = !selectedParcelId;
  toggleFlagBtn.dataset.mode = selectedIsUnflagged ? "reflag" : "unflag";
  toggleFlagBtn.textContent = selectedIsUnflagged ? "Re-flag Violation" : "Unflag Violation";
  const rec = detailPayload.unflag_record || null;
  if (selectedIsUnflagged && rec) {
    const who = rec.unflagged_by || "unknown";
    const reason = rec.reason || "Reviewed";
    flagMeta.textContent = `Currently unflagged by ${who}. Reason: ${reason}`;
  } else {
    flagMeta.textContent = "This parcel is currently active in violation view.";
  }
}

async function fetchJson(url, init = {}, options = {}) {
  const { auth = true } = options;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 20000);
  const headers = auth ? authHeaders(init.headers) : new Headers(init.headers || {});
  const response = await fetch(url, { ...init, headers, signal: controller.signal }).finally(() => clearTimeout(timeout));
  if (!response.ok) {
    let plain = "";
    try {
      const raw = await response.text();
      plain = String(raw || "")
        .replace(/<[^>]*>/g, " ")
        .replace(/\s+/g, " ")
        .trim();
    } catch {
      plain = "";
    }
    const short = plain.length > 220 ? `${plain.slice(0, 220)}...` : plain;
    const err = new Error(`API ${response.status}${short ? `: ${short}` : ""}`);
    err.status = response.status;
    throw err;
  }
  return response.json();
}

async function login(username, password) {
  const payload = await fetchJson(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }, { auth: false });
  setAuthToken(payload.access_token || "");
  currentUser = payload.user || null;
  applyUserState();
  return payload;
}

async function loadCurrentUser() {
  if (!authToken) throw new Error("No auth token");
  const payload = await fetchJson(`${API_BASE}/auth/me`);
  currentUser = payload.user || null;
  applyUserState();
  return payload;
}

async function logout() {
  try {
    await fetchJson(`${API_BASE}/auth/logout`, { method: "POST" });
  } catch {
    // Ignore network/auth errors while clearing local session.
  }
  setAuthToken("");
  currentUser = null;
  applyUserState();
  updateFlagControls(null);
  if (violationsLayer) violationsLayer.clearLayers();
  showLogin("Signed out. Please login.");
}

async function loadParcelDetails(parcelId) {
  if (staticMode) {
    const key = String(parcelId ?? "");
    const feature = staticFeatureById.get(key);
    if (!feature) {
      setStatus(`Parcel ${key} not found in static data`, "medium");
      return;
    }
    const props = { ...(feature.properties || {}), geometry: feature.geometry || null };
    renderParcelDetails(props);
    updateFlagControls(props);
    const tone = props.risk_category === "High" ? "high" : props.risk_category === "Medium" ? "medium" : "low";
    setStatus(`Parcel ${key} loaded`, tone);
    return;
  }
  setStatus(`Loading parcel ${parcelId}...`, "neutral");
  const payload = await fetchJson(`${API_BASE}/parcel/${encodeURIComponent(parcelId)}`);
  renderParcelDetails(payload);
  updateFlagControls(payload);
  const tone = payload.risk_category === "High" ? "high" : payload.risk_category === "Medium" ? "medium" : "low";
  setStatus(`Parcel ${parcelId} loaded`, tone);
}

function popupHtmlForFeature(feature) {
  const props = feature?.properties || {};
  const parcelId = props.parcel_id ?? props.building_id;
  const rawScore = Number(props.final_risk ?? props.final_violation_probability ?? 0);
  const score = Math.min(99.0, Number.isFinite(rawScore) ? rawScore : 0).toFixed(1);
  const type = props.encroachment_type || props.satellite_evidence || "none";
  const category = props.risk_category || "Low";
  const tone = riskTone(category);
  return `
    <div class="map-popup tone-${tone}">
      <div class="map-popup-title">${parcelId ?? "Location"}</div>
      <div class="map-popup-row"><span>Risk:</span><strong class="risk-${tone}">${category}</strong></div>
      <div class="map-popup-row"><span>Probability:</span><strong>${score}%</strong></div>
      <div class="map-popup-row"><span>Signals:</span><strong>${String(type).replaceAll("_", " ")}</strong></div>
    </div>
  `;
}

async function handleFeatureClick(feature) {
  const props = feature?.properties || {};
  const parcelId = props.parcel_id ?? props.building_id;
  if (parcelId === null || parcelId === undefined) {
    setStatus("Parcel identifier missing in feature properties", "medium");
    return;
  }
  try {
    await loadParcelDetails(parcelId);
  } catch (error) {
    setStatus(`Failed to load parcel: ${error.message}`, "high");
  }
}

function applyScopeFromResponse(scope) {
  if (!scope || !Array.isArray(scope.area_bbox) || scope.area_bbox.length !== 4) return false;
  const [minLon, minLat, maxLon, maxLat] = scope.area_bbox.map(Number);
  if (![minLon, minLat, maxLon, maxLat].every(Number.isFinite)) return false;
  map.fitBounds(
    [
      [minLat, minLon],
      [maxLat, maxLon],
    ],
    { padding: [18, 18] }
  );
  return true;
}

async function loadViolations() {
  if (staticMode) {
    setStatus("Loading violations from static output...", "neutral");
    updateFlagControls(null);
    if (violationsLayer) {
      violationsLayer.remove();
    }
    violationsLayer = L.layerGroup().addTo(map);
    const payload = staticPayload || { features: [], total_rows: 0, scope: null };
    const allFeatures = payload.features || [];
    const total = payload.total_rows ?? allFeatures.length;
    if (!allFeatures.length) {
      setStatus("No active violations found in static output.", "low");
      return;
    }

    staticFeatureById = new Map();
    for (const f of allFeatures) {
      const props = f?.properties || {};
      const id = props.parcel_id ?? props.building_id;
      if (id !== null && id !== undefined) staticFeatureById.set(String(id), f);
    }

    const renderCount = Math.min(allFeatures.length, MAX_RENDERED);
    const features = allFeatures.slice(0, renderCount);
    const geojson = { type: "FeatureCollection", features };
    const batchLayer = L.geoJSON(geojson, {
      pointToLayer: (feature, latlng) => {
        const props = feature.properties || {};
        const category = props.risk_category;
        const score = props.final_risk ?? props.final_violation_probability;
        return L.circleMarker(latlng, {
          radius: radiusForScore(score),
          color: colorForRisk(category),
          fillColor: colorForRisk(category),
          weight: 1.0,
          fillOpacity: 0.72,
          renderer: canvasRenderer,
        });
      },
    });
    batchLayer.on("click", async (event) => {
      const clickedLayer = event?.layer;
      const feature = clickedLayer?.feature;
      if (!feature) return;
      if (clickedLayer && typeof clickedLayer.getLatLng === "function") {
        L.popup()
          .setLatLng(clickedLayer.getLatLng())
          .setContent(popupHtmlForFeature(feature))
          .openOn(map);
      }
      await handleFeatureClick(feature);
    });
    batchLayer.addTo(violationsLayer);
    const b = batchLayer.getBounds();
    if (b && b.isValid()) {
      map.fitBounds(b, { padding: [20, 20] });
    }

    if (renderCount < total) {
      setStatus(
        `Loaded ${renderCount.toLocaleString()} highest-priority parcels (of ${total.toLocaleString()}) from static output`,
        "medium"
      );
    } else {
      setStatus(`Loaded static set: ${renderCount.toLocaleString()} parcels`, "low");
    }
    return;
  }

  setStatus("Loading violations (performance mode)...", "neutral");
  updateFlagControls(null);
  if (violationsLayer) {
    violationsLayer.remove();
  }
  violationsLayer = L.layerGroup().addTo(map);

  let offset = 0;
  let total = null;
  let loaded = 0;
  let bounds = null;
  let scopeApplied = false;

  while ((total === null || offset < total) && loaded < MAX_RENDERED) {
    const remaining = Math.max(0, MAX_RENDERED - loaded);
    const nextLimit = Math.min(PAGE_SIZE, remaining || PAGE_SIZE);
    const payload = await fetchJson(
      `${API_BASE}/violations?limit=${nextLimit}&offset=${offset}` +
        `&sort_by=final_violation_probability&descending=true&prioritize_satellite=true` +
        `&as_points=true&compact=true&min_probability=${MIN_PROBABILITY}`
    );
    if (!scopeApplied) {
      scopeApplied = applyScopeFromResponse(payload.scope);
    }
    total = payload.total_rows;
    if (total === 0) {
      setStatus("No active violations found in your area.", "low");
      return;
    }
    const features = payload.features || [];
    const geojson = { type: "FeatureCollection", features };

    const batchLayer = L.geoJSON(geojson, {
      pointToLayer: (feature, latlng) => {
        const props = feature.properties || {};
        const category = props.risk_category;
        const score = props.final_risk ?? props.final_violation_probability;
        return L.circleMarker(latlng, {
          radius: radiusForScore(score),
          color: colorForRisk(category),
          fillColor: colorForRisk(category),
          weight: 1.0,
          fillOpacity: 0.72,
          renderer: canvasRenderer,
        });
      },
    });
    batchLayer.on("click", async (event) => {
      const clickedLayer = event?.layer;
      const feature = clickedLayer?.feature;
      if (!feature) return;
      if (clickedLayer && typeof clickedLayer.getLatLng === "function") {
        L.popup()
          .setLatLng(clickedLayer.getLatLng())
          .setContent(popupHtmlForFeature(feature))
          .openOn(map);
      }
      await handleFeatureClick(feature);
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
    await new Promise((resolve) => requestAnimationFrame(resolve));
  }

  if (!scopeApplied && bounds && bounds.isValid()) {
    map.fitBounds(bounds, { padding: [20, 20] });
  }
  if (total !== null && loaded < total) {
    setStatus(
      `Loaded ${loaded.toLocaleString()} highest-priority parcels (of ${total.toLocaleString()}) for faster map performance`,
      "medium"
    );
    return;
  }
  setStatus(`Loaded full set: ${loaded.toLocaleString()} parcels`, "low");
}

async function runPipeline() {
  if (staticMode) {
    setStatus("Pipeline is disabled in static mode.", "medium");
    return;
  }
  if (!currentUser || currentUser.role !== "admin") {
    setStatus("Only admin can run the full city pipeline.", "medium");
    return;
  }
  setRunButtonLoading(true);
  try {
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
  } finally {
    setRunButtonLoading(false);
    applyUserState();
  }
}

async function toggleFlagState() {
  if (staticMode) {
    setStatus("Flag/unflag is disabled in static mode.", "medium");
    return;
  }
  if (!selectedParcelId) {
    setStatus("Select a parcel first.", "medium");
    return;
  }
  const mode = selectedIsUnflagged ? "reflag" : "unflag";
  const endpoint =
    mode === "unflag"
      ? `${API_BASE}/violations/${encodeURIComponent(selectedParcelId)}/unflag`
      : `${API_BASE}/violations/${encodeURIComponent(selectedParcelId)}/reflag`;
  const payload =
    mode === "unflag"
      ? {
          reason:
            currentUser?.role === "surveyor"
              ? "Reviewed by assigned surveyor in field validation."
              : "Reviewed by admin.",
        }
      : null;
  setStatus(mode === "unflag" ? "Unflagging violation..." : "Restoring violation flag...", "neutral");
  await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: payload ? JSON.stringify(payload) : undefined,
  });
  setStatus(mode === "unflag" ? "Violation unflagged." : "Violation re-flagged.", "low");
  await loadViolations();
}

if (runPipelineBtn) {
  runPipelineBtn.addEventListener("click", async () => {
    try {
      await runPipeline();
    } catch (error) {
      if (error?.status === 401) {
        await logout();
        return;
      }
      setStatus(`Pipeline failed: ${error.message}`, "high");
      setRunButtonLoading(false);
      applyUserState();
    }
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", async () => {
    await logout();
  });
}

if (toggleFlagBtn) {
  toggleFlagBtn.addEventListener("click", async () => {
    try {
      await toggleFlagState();
    } catch (error) {
      if (error?.status === 401) {
        await logout();
        return;
      }
      setStatus(`Flag update failed: ${error.message}`, "high");
    }
  });
}

if (loginForm) {
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = String(usernameInput?.value || "").trim();
    const password = String(passwordInput?.value || "").trim();
    if (!username || !password) {
      if (loginError) loginError.textContent = "Enter both username and password.";
      return;
    }
    if (loginError) loginError.textContent = "";
    try {
      await login(username, password);
      hideLogin();
      await loadViolations();
    } catch (error) {
      if (loginError) loginError.textContent = "Invalid credentials. Try a demo user.";
      setAuthToken("");
      currentUser = null;
      applyUserState();
    }
  });
}

async function bootstrap() {
  try {
    const res = await fetch(STATIC_DATA_URL, { cache: "no-store" });
    if (res.ok) {
      staticPayload = await res.json();
      staticMode = true;
      if (loginOverlay) loginOverlay.classList.add("hidden");
      applyUserState();
      await loadViolations();
      return;
    }
  } catch {
    // Fallback to API mode.
  }

  if (!authToken) {
    showLogin();
    setStatus("Login required.", "medium");
    applyUserState();
    return;
  }
  try {
    await loadCurrentUser();
    hideLogin();
    await loadViolations();
  } catch (error) {
    setAuthToken("");
    currentUser = null;
    applyUserState();
    showLogin("Session expired. Please login again.");
    setStatus("Login required.", "medium");
  }
}

bootstrap();
