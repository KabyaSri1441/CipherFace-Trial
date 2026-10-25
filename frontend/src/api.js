export const API_BASE = "";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (error) {
    throw new Error("Local client API unavailable on port 8001");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    let detail = payload.detail || `Request failed (${response.status})`;
    if (typeof detail === "string" && detail.includes("authentication is not configured")) {
      detail += " The browser must talk to the local client (port 8001), which holds the token; " +
        "the ciphertext server (port 8002) must share its API token. Start both with: python -m scripts.run_stack";
    }
    throw new Error(detail);
  }
  return payload;
}

export function getStatus() {
  return request("/api/status");
}

export function getTemplates() {
  return request("/api/templates");
}

export function getAuditEvents() {
  return Promise.resolve([]);
}

export function getResearchSummary() {
  return Promise.resolve(null);
}

export function deleteTemplate(userId) {
  const body = new FormData();
  body.append("user_id", userId);
  return request("/api/templates", { method: "DELETE", body });
}

export function verifyImage(file, userId) {
  const body = new FormData();
  body.append("image", file);
  body.append("user_id", userId);
  return request("/api/verify", { method: "POST", body });
}

export function verifyEncryptedEmbedding(encryptedEmbedding, userId) {
  const body = new FormData();
  body.append("encrypted_embedding", encryptedEmbedding);
  body.append("user_id", userId);
  return request("/api/verify-encrypted", { method: "POST", body });
}

export function enrollImage(name, file) {
  const body = new FormData();
  body.append("name", name);
  body.append("image", file);
  return request("/api/enroll", { method: "POST", body });
}

export function startEvaluation() {
  return request("/api/evaluation/start", { method: "POST" });
}

export function getEvaluation(jobId) {
  return request(`/api/evaluation/${jobId}`);
}

export function getLfwSummary() {
  return request("/api/lfw/summary");
}

export function getLfwIdentities(search = "") {
  return request(`/api/lfw/identities?search=${encodeURIComponent(search)}`);
}

export function getLfwIdentity(name) {
  return request(`/api/lfw/identity/${encodeURIComponent(name)}`);
}

export function demoLogin(password) {
  const body = new FormData();
  body.append("password", password);
  return request("/api/demo-login", { method: "POST", body });
}
