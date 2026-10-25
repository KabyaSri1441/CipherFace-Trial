import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  Camera,
  Database,
  Fingerprint,
  LockKeyhole,
  LogIn,
  LogOut,
  Network,
  RefreshCw,
  ScanFace,
  Server,
  ShieldCheck,
  Upload,
  UserRoundPlus,
  X,
  Zap,
} from "lucide-react";

import {
  API_BASE,
  enrollImage,
  getEvaluation,
  getLfwIdentities,
  getLfwIdentity,
  getStatus,
  getTemplates,
  getAuditEvents,
  getResearchSummary,
  deleteTemplate,
  startEvaluation,
  verifyEncryptedEmbedding,
  verifyImage,
  demoLogin,
} from "./api";

const pipeline = [
  ["Face detection", "MTCNN", "client"],
  ["Face alignment", "5-point landmarks", "client"],
  ["Feature extraction", "FaceNet + ArcFace", "client"],
  ["Feature fusion", "512-D weighted sum", "client"],
  ["Normalization", "L2 normalized", "client"],
  ["Encryption", "CKKS ciphertext", "encryption"],
  ["Send encrypted probe", "Ciphertext only", "boundary"],
  ["Encrypted template database", "Already stored on Server", "server"],
  ["Homomorphic matching", "Encrypted dot product", "server"],
  ["Encrypted similarity", "Return ciphertext", "boundary"],
  ["Client decryption", "Secret key stays here", "client"],
  ["Decision", "ACCEPT / REJECT", "client"],
];

const initialStatus = {
  system: "OFFLINE",
  models: {
    mtcnn: false,
    facenet: false,
    arcface: false,
  },
  encryption: {
    scheme: "CKKS",
    server_mode: "CIPHERTEXT_ONLY",
  },
  database: {
    status: false,
    enrolled_users: 0,
  },
  verification: {
    threshold: 0.142078993,
  },
};

function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [loginPassword, setLoginPassword] = useState("");
  const [loginError, setLoginError] = useState("");
  const [activeView, setActiveView] = useState("overview");
  const [clientAction, setClientAction] = useState("verify");
  const [status, setStatus] = useState(initialStatus);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [templates, setTemplates] = useState([]);
  const [auditEvents, setAuditEvents] = useState([]);
  const [research, setResearch] = useState(null);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [encryptedEmbedding, setEncryptedEmbedding] = useState("");
  const [verifyFile, setVerifyFile] = useState(null);
  const [enrollFile, setEnrollFile] = useState(null);
  const [name, setName] = useState("");
  const [claimedUser, setClaimedUser] = useState("");
  const [conferenceMode, setConferenceMode] = useState(false);

  const navigation = [
    ["overview", "Command Center", Activity],
    ["client", "Client Console", Fingerprint],
    ["server", "Server Console", Server],
    ["verification", "Verification", ScanFace],
    ["enrollment", "Enrollment", UserRoundPlus],
    ["privacy", "Privacy Center", LockKeyhole],
    ["security", "Threat Model", ShieldCheck],
    ["research", "Research & Evaluation", Zap],
    ["templates", "Template Registry", Database],
  ];


  async function submitLogin(event) {
    event.preventDefault();
    setLoginError("");
    try {
      await demoLogin(loginPassword);
      setAuthenticated(true);
      setActiveView("overview");
      setConferenceMode(false);
      setLoginPassword("");
      refresh();
    } catch (error) {
      setLoginError(error.message || "Login failed.");
    }
  }

  async function refresh() {
    try {
      const [nextStatus, nextTemplates, nextAudit] = await Promise.all([
        getStatus(), getTemplates(), getAuditEvents(),
      ]);
      setStatus(nextStatus);
      setTemplates(nextTemplates);
      setClaimedUser((current) => current || nextTemplates[0]?.name || "");
      setAuditEvents(nextAudit);
      setStatusLoaded(true);
      setNotice("");
    } catch (error) {
      setStatusLoaded(false);
      setNotice(error.message);
    }
  }

  useEffect(() => { refresh(); }, []);

  useEffect(() => {
    if (activeView !== "research" || research) return;
    getResearchSummary().then(setResearch).catch((error) => setNotice(error.message));
  }, [activeView, research]);

  async function submitVerify(event) {
    event.preventDefault();
    if (!verifyFile) return setNotice("Choose a face image or capture one locally.");
    if (!verifyFile.type.startsWith("image/") || verifyFile.size > 10 * 1024 * 1024) {
      return setNotice("Use a valid image file up to 10 MB.");
    }
    if (!claimedUser) return setNotice("Choose the identity you are claiming.");
    setBusy(true);
    setNotice("");
    setResult(null);
    setInspection(null);
    setEncryptedEmbedding("");
    try {
      const response = await verifyImage(verifyFile, claimedUser);
      setInspection(response);
      setEncryptedEmbedding(response.encrypted_embedding || "");
      setResult(response);
      if (!conferenceMode) setVerifyFile(null);
      setAuditEvents(response.privacy_audit || await getAuditEvents());
      await refresh();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }
  
  async function submitManualVerification(event, ciphertext) {
    event.preventDefault();

    if (!ciphertext || !ciphertext.trim()) {
      setNotice("Paste the encrypted ciphertext before verifying.");
      return;
    }

    if (!claimedUser) {
      setNotice("Choose the identity you are claiming.");
      return;
    }

    setBusy(true);
    setNotice("");
    setResult(null);

    try {
      const response = await verifyEncryptedEmbedding(
        ciphertext.trim(),
        claimedUser
      );

      setResult(response);
      setNotice(
        `Verification completed: ${response.decision || "Result received"}.`
      );

      await refresh();
    } catch (error) {
      setNotice(
        error.message || "Manual ciphertext verification failed."
      );
    } finally {
      setBusy(false);
    }
  }

  async function submitEnroll(event) {
    event.preventDefault();
    if (!name.trim()) return setNotice("Enter an enrollment identifier.");
    if (!enrollFile) return setNotice("Choose or capture an enrollment image.");
    if (!enrollFile.type.startsWith("image/") || enrollFile.size > 10 * 1024 * 1024) {
      return setNotice("Use a valid image file up to 10 MB.");
    }
    setBusy(true);
    setNotice("");
    try {
      const response = await enrollImage(name, enrollFile);
      setAuditEvents(response.privacy_audit || await getAuditEvents());
      setNotice(`${response.name} enrolled as an encrypted template.`);
      setName("");
      setEnrollFile(null);
      await refresh();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function removeTemplate(userId) {
    if (!window.confirm(`Delete the encrypted template for ${userId}?`)) return;
    try {
      await deleteTemplate(userId);
      setTemplates((current) => current.filter((template) => template.name !== userId));
      setStatus((current) => ({
        ...current,
        database: { ...current.database, enrolled_users: Math.max(0, current.database.enrolled_users - 1) },
      }));
      setAuditEvents(await getAuditEvents());
      setNotice("Encrypted template and metadata deleted.");
    } catch (error) {
      setNotice(error.message);
    }
  }

  const posture = status.privacy_posture || {};
  const operational = statusLoaded && status.system === "ONLINE";
  const trustVerified = operational
    && posture.server_context_secret_free === true
    && posture.secret_context_outside_project === true
    && posture.secret_context_permissions_owner_only === true
    && posture.server_has_image_route === false
    && posture.server_has_plaintext_embedding_route === false;
  const viewTitle = navigation.find(([id]) => id === activeView)?.[1] || "Command Center";
  const workspaceProps = {
    action: activeView === "enrollment" ? "enroll" : clientAction,
    setAction: (action) => {
      setClientAction(action);
      setActiveView(action === "enroll" ? "enrollment" : "verification");
    },
    busy, verifyFile, setVerifyFile, enrollFile, setEnrollFile, name, setName,
    claimedUser, setClaimedUser, result, inspection,
    submitVerify, submitEnroll, status, templates, setNotice, setResult,
    setInspection, conferenceMode,
    hideActionSelector: true,
  };


  if (!authenticated) {
    return (
      <div className="cf-login-screen">
        <form className="cf-login-card" onSubmit={submitLogin}>
          <div className="cf-login-mark"><Fingerprint size={28} /></div>
          <div className="eyebrow">CIPHERFACE / AUTHENTICATION NODE</div>
          <h1>Sign in</h1>
          <p>Access the local client console and ciphertext server console.</p>
          <label htmlFor="cf-login-password">Demo password</label>
          <input id="cf-login-password" type="password" autoComplete="current-password" value={loginPassword} onChange={(e) => setLoginPassword(e.target.value)} required placeholder="Enter demo password" />
          {loginError && <div className="cf-login-error" role="alert">{loginError}</div>}
          <button className="cf-login-submit" type="submit"><LogIn size={16} /> SIGN IN</button>
          <small>Development demo gate. Default password: <code>cipherface-demo</code>. Set CIPHERFACE_DEMO_PASSWORD before launch to change it. This is not production authentication.</small>
        </form>
      </div>
    );
  }

  return (
    <div className="app-shell research-shell">
      <header className="topbar research-topbar">
        <a className="brand" href="#overview" onClick={(event) => { event.preventDefault(); setActiveView("overview"); }}>
          <span className="brand-mark"><Fingerprint size={20} /></span>
          <span className="brand-wordmark">CIPHERFACE</span>
          <span className="brand-subtitle">PRIVACY-PRESERVING BIOMETRIC AUTHENTICATION</span>
        </a>
        <div className="topbar-actions">
          <div className={`system-pill ${operational ? "is-online" : "is-offline"}`}>
            <span /> {operational ? "SYSTEM OPERATIONAL" : "SYSTEM UNAVAILABLE"}
          </div>
          <div className={`posture-pill ${trustVerified ? "is-verified" : "is-partial"}`}>
            <LockKeyhole size={13} /> {trustVerified ? "KEY BOUNDARY VERIFIED" : "PRIVACY POSTURE PARTIAL"}
          </div>
          <button className="demo-button" onClick={() => { setConferenceMode(true); setClientAction("verify"); setActiveView("verification"); }}>
            <Activity size={14} /> CONFERENCE DEMO
          </button>
          <button className="demo-button" onClick={() => { setAuthenticated(false); setActiveView("overview"); setResult(null); setEncryptedEmbedding(""); }}>
            <LogOut size={14} /> SIGN OUT
          </button>
        </div>
      </header>

      <div className="product-nav-wrap">
        <nav className="product-nav" aria-label="CipherFace sections">
          {navigation.map(([id, label, Icon]) => (
            <button key={id} className={activeView === id ? "product-nav-item active" : "product-nav-item"} onClick={() => setActiveView(id)}>
              <Icon size={15} /> {label}
            </button>
          ))}
        </nav>
        <span className="nav-build-id">RESEARCH SYSTEM / CKKS-512 / LOCAL CLIENT</span>
      </div>

      <main className="research-content">
        <div className="page-heading research-heading">
          <div>
            <div className="eyebrow">CIPHERFACE / {viewTitle.toUpperCase()}</div>
            <h1>{activeView === "overview" ? "Privacy-preserving authentication" : viewTitle}</h1>
            <p>Trusted local biometric processing · ciphertext-only server matching · local decision</p>
          </div>
          <button className="icon-button" onClick={refresh} title="Refresh verified system state"><RefreshCw size={16} /></button>
        </div>

        {notice && <div className="notice" role="status"><X size={15} />{notice}</div>}

        {activeView === "overview" && (
          <CommandCenter
            status={status} operational={operational} trustVerified={trustVerified}
            onNavigate={setActiveView} auditEvents={auditEvents}
            workspace={<ClientWorkspace {...workspaceProps} encryptedEmbedding={encryptedEmbedding} hideActionSelector />}
          />
        )}
        {(activeView === "verification" || activeView === "enrollment" || activeView === "client") && (
          <>
            {activeView === "client" && <div className="console-intro"><div className="eyebrow">TRUSTED LOCAL PROCESS / PORT 8001</div><h2>Client Console</h2><p>Image processing and plaintext embeddings stay local. The serialized CKKS ciphertext below is the actual payload returned by the client after transmission.</p></div>}
            <ClientWorkspace {...workspaceProps} encryptedEmbedding={encryptedEmbedding} hideActionSelector />
          </>
        )}
        {activeView === "server" && <ServerConsole status={status} templates={templates} result={result} encryptedEmbedding={encryptedEmbedding} claimedUser={claimedUser} setClaimedUser={setClaimedUser} busy={busy} onVerify={submitManualVerification} />}
        {activeView === "privacy" && <PrivacyCenter status={status} auditEvents={auditEvents} />}
        {activeView === "security" && <ThreatModel status={status} />}
        {activeView === "research" && <ResearchView research={research} />}
        {activeView === "templates" && <DatabaseView templates={templates} onDelete={removeTemplate} />}
      </main>
      <footer className="research-footer"><span>CIPHERFACE RESEARCH PROTOTYPE</span><span>Local client trusted · remote verifier untrusted · no production-security claim</span></footer>
    </div>
  );
}


function ServerConsole({
  status,
  templates,
  result,
  encryptedEmbedding,
  claimedUser,
  setClaimedUser,
  busy,
  onVerify,
}) {
  const [copied, setCopied] = useState(false);
  const [ciphertext, setCiphertext] = useState("");
  const [useGeneratedCiphertext, setUseGeneratedCiphertext] = useState(false);

  const copy = async () => {
    const value = encryptedEmbedding || "";

    if (!value) return;

    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  };

  const handleSubmit = (event) => {
    onVerify(event, ciphertext);
  };

  return (
    <div className="server-console-grid">
      <section className="server-console-card">
        <div className="eyebrow">UNTRUSTED VERIFIER / PORT 8002</div>

        <h2>Ciphertext Server Console</h2>

        <p>
          The server receives ciphertext only when you explicitly submit it.
          Homomorphic matching is performed server-side, and the encrypted
          similarity is returned for decryption by the local client.
        </p>

        <div className="server-metrics">
          <div>
            <span>Server mode</span>
            <strong>
              {status.encryption?.server_mode || "UNKNOWN"}
            </strong>
          </div>

          <div>
            <span>Secret key loaded</span>
            <strong>
              {String(
                status.encryption?.secret_key_loaded ?? "UNKNOWN"
              )}
            </strong>
          </div>

          <div>
            <span>Registered templates</span>
            <strong>{templates.length}</strong>
          </div>

          <div>
            <span>Last verification</span>
            <strong>
              {result
                ? "Completed"
                : "Awaiting manual submission"}
            </strong>
          </div>
        </div>
      </section>

      <section className="server-console-card">
        <div className="eyebrow">MANUAL CIPHERTEXT INPUT</div>

        <h3>Submit encrypted template</h3>

        <p>
          Copy the Base64 ciphertext from the Command Center and paste it
          below. Nothing is sent to the verification server until you click
          Verify Ciphertext.
        </p>

        {encryptedEmbedding && (
          <div className="cipher-meta">
            <span>Generated ciphertext is available</span>
            <span>
              Length: {encryptedEmbedding.length.toLocaleString()} characters
            </span>
          </div>
        )}

        <div className="server-form-field">
          <label htmlFor="server-claimed-user">
            Identity to verify
          </label>

          <select
            id="server-claimed-user"
            value={claimedUser}
            onChange={(event) => setClaimedUser(event.target.value)}
            disabled={busy}
          >
            <option value="">Select an enrolled identity</option>

            {templates.map((template) => (
              <option key={template.name} value={template.name}>
                {template.name}
              </option>
            ))}
          </select>
        </div>

        <div className="server-form-field">
          <label htmlFor="manual-ciphertext">
            Base64 CKKS ciphertext
          </label>

          <textarea
            id="manual-ciphertext"
            className="server-ciphertext"
            value={ciphertext}
            onChange={(event) => setCiphertext(event.target.value)}
            placeholder="Paste the complete encrypted ciphertext here..."
            aria-label="Paste Base64 CKKS ciphertext"
            spellCheck={false}
            rows={8}
            disabled={busy}
          />
        </div>

        <div className="server-console-actions">
          <button
            type="button"
            className="secondary-button"
            onClick={() => setCiphertext(encryptedEmbedding || "")}
            disabled={!encryptedEmbedding || busy}
          >
            USE GENERATED CIPHERTEXT
          </button>

          <button
            type="button"
            className="secondary-button"
            onClick={copy}
            disabled={!encryptedEmbedding}
          >
            {copied ? "COPIED" : "COPY GENERATED CIPHERTEXT"}
          </button>

          <button
            type="button"
            className="secondary-button"
            onClick={() => setCiphertext("")}
            disabled={!ciphertext || busy}
          >
            CLEAR
          </button>
        </div>

        <button
          type="button"
          className="primary-button"
          onClick={handleSubmit}
          disabled={busy || !ciphertext.trim() || !claimedUser}
        >
          {busy ? "VERIFYING..." : "VERIFY CIPHERTEXT"}
        </button>

        <div className="server-empty">
          The generated ciphertext is not submitted automatically. You can
          paste a copied ciphertext or use the generated ciphertext directly
          in this input.
        </div>
      </section>

      <section
        className={`server-console-card verification-result-card ${
          result?.decision === "REJECT"
            ? "verification-rejected"
            : result?.decision === "ACCEPT"
            ? "verification-accepted"
            : "verification-pending"
        }`}
      >
        <div className="eyebrow">VERIFICATION RESULT</div>

        <h3>Server Interaction Summary</h3>

        {result ? (
          <div className="server-metrics">
            <div>
              <span>Identity</span>
              <strong>{result.best_match || claimedUser || "Unknown"}</strong>
            </div>

            <div className="decision-metric">
              <span>Verification Decision</span>
              <strong
                className={`decision-badge ${
                  result.decision === "ACCEPT"
                    ? "decision-accept"
                    : result.decision === "REJECT"
                    ? "decision-reject"
                    : "decision-pending"
                }`}
              >
                {result.decision || "Completed"}
              </strong>
            </div>

            <div>
              <span>Similarity Score</span>
              <strong>
                {result.similarity != null &&
                Number.isFinite(Number(result.similarity))
                  ? Number(result.similarity).toFixed(6)
                  : "Not reported"}
              </strong>
            </div>

            <div>
              <span>Verification Threshold</span>
              <strong>
                {result.threshold != null &&
                Number.isFinite(Number(result.threshold))
                  ? Number(result.threshold).toFixed(6)
                  : "Not reported"}
              </strong>
            </div>

            <div>
              <span>Server Input</span>
              <strong>
                {result.server_input || "ENCRYPTED_QUERY_ONLY"}
              </strong>
            </div>

            <div>
              <span>Server Output</span>
              <strong>
                {result.server_output || "ENCRYPTED_SIMILARITY_ONLY"}
              </strong>
            </div>

            <div>
              <span>Ciphertext Size</span>
              <strong>
                {result.server_received_ciphertext_bytes != null
                  ? `${Number(result.server_received_ciphertext_bytes).toLocaleString()} bytes`
                  : "Not reported"}
              </strong>
            </div>

            <div className="hash-metric">
              <span>Server-Observed SHA-256</span>
              <strong>
                {result.server_received_ciphertext_sha256 || "Not reported"}
              </strong>
            </div>

            <div className="hash-metric">
              <span>Server Request ID</span>
              <strong>
                {result.server_request_id || "Not reported"}
              </strong>
            </div>
          </div>
        ) : (
          <div className="server-empty">
            No manual verification has been completed yet. Paste the encrypted
            ciphertext above and click Verify Ciphertext to begin.
          </div>
        )}
      </section>
    </div>
  );
}

function LegacyApp() {
  const [role, setRole] = useState("member");
  const [activeTab, setActiveTab] = useState("verify");
  const [clientAction, setClientAction] = useState("verify");

  const [status, setStatus] = useState(initialStatus);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [templates, setTemplates] = useState([]);

  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [inspection, setInspection] = useState(null);
  const [encryptedEmbedding, setEncryptedEmbedding] = useState("");

  const [verifyFile, setVerifyFile] = useState(null);
  const [enrollFile, setEnrollFile] = useState(null);
  const [name, setName] = useState("");
  const [claimedUser, setClaimedUser] = useState("");

  async function refresh() {
    try {
      const [nextStatus, nextTemplates] = await Promise.all([
        getStatus(),
        getTemplates(),
      ]);

      setStatus(nextStatus);
      setTemplates(nextTemplates);
      setClaimedUser((current) => current || nextTemplates[0]?.name || "");
      setStatusLoaded(true);
      setNotice("");
    } catch (error) {
      setStatusLoaded(false);
      setNotice(error.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function submitVerify(event) {
    event.preventDefault();

    if (!verifyFile) {
      return setNotice("Choose a face image before generating ciphertext.");
    }

    if (
      !verifyFile.type.startsWith("image/") ||
      verifyFile.size > 10 * 1024 * 1024
    ) {
      return setNotice("Use a valid image file up to 10 MB.");
    }

    if (!claimedUser) {
      return setNotice("Choose the identity you are claiming.");
    }

    setBusy(true);
    setNotice("");
    setResult(null);
    setInspection(null);
    setEncryptedEmbedding("");

    try {
      const response = await verifyImage(verifyFile, claimedUser);

      setInspection(response);
      setEncryptedEmbedding(response.encrypted_embedding || "");

      setResult(null);

      setNotice(
        "Ciphertext generated locally. Copy it to the Server Console and click Verify Ciphertext."
      );

      setVerifyFile(null);

      setAuditEvents(
        response.privacy_audit || await getAuditEvents()
      );
    } catch (error) {
      setNotice(error.message || "Failed to generate encrypted ciphertext.");
    } finally {
      setBusy(false);
    }
  }

  async function submitManualVerification(event, ciphertext) {
    event.preventDefault();

    if (!ciphertext.trim()) {
      setNotice("Paste the encrypted ciphertext before verifying.");
      return;
    }

    if (!claimedUser) {
      setNotice("Choose the identity you are claiming.");
      return;
    }

    setBusy(true);
    setNotice("");
    setResult(null);

    try {
      const response = await verifyEncryptedEmbedding(
        ciphertext.trim(),
        claimedUser
      );

      setResult(response);
      setNotice(
        `Verification completed: ${response.decision || "Result received"}.`
      );

      await refresh();
    } catch (error) {
      setNotice(error.message || "Manual ciphertext verification failed.");
    } finally {
      setBusy(false);
    }
  }

  async function submitEnroll(event) {
    event.preventDefault();

    if (!name.trim()) {
      return setNotice("Enter a name before enrolling.");
    }

    if (!enrollFile) {
      return setNotice("Choose an enrollment image.");
    }

    if (
      !enrollFile.type.startsWith("image/") ||
      enrollFile.size > 10 * 1024 * 1024
    ) {
      return setNotice("Use a valid image file up to 10 MB.");
    }

    if (
      templates.some(
        (template) =>
          template.name.toLowerCase() === name.trim().toLowerCase()
      )
    ) {
      return setNotice("That identity is already enrolled.");
    }

    setBusy(true);
    setNotice("");

    try {
      const response = await enrollImage(name, enrollFile);

      setNotice(`${response.name} is now encrypted and enrolled.`);

      setName("");
      setEnrollFile(null);

      await refresh();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  const memberNav = [["client", "Client workspace", ScanFace]];

  const evaluatorNav = [
    ["overview", "Overview", Activity],
    ["monitor", "Verification monitor", Network],
    ["transfer", "Encrypted embedding transfer", LockKeyhole],
    ["database", "Encrypted database", Database],
    ["security", "Security / privacy model", ShieldCheck],
    ["evaluation", "Evaluation", Zap],
  ];

  const nav = role === "member" ? memberNav : evaluatorNav;

  const switchRole = (nextRole) => {
    setRole(nextRole);
    setActiveTab(nextRole === "member" ? "client" : "overview");
    setClientAction("verify");
    setNotice("");
  };

  const title =
    role === "member"
      ? "Privacy-Preserving Face Verification"
      : {
          overview: "System overview",
          monitor: "Verification monitor",
          transfer: "Encrypted embedding transfer",
          database: "Encrypted template registry",
          security: "Trust boundary",
          evaluation: "Recorded evaluation",
        }[activeTab];

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Fingerprint size={21} />
          </span>

          <span>
            CIPHER<span>FACE</span>
          </span>
        </div>

        <div
          className="role-switcher"
          role="tablist"
          aria-label="Choose workspace role"
        >
          <button
            className={
              role === "member" ? "role-tab active" : "role-tab"
            }
            onClick={() => switchRole("member")}
          >
            CLIENT
          </button>

          <button
            className={
              role === "evaluator" ? "role-tab active" : "role-tab"
            }
            onClick={() => switchRole("evaluator")}
          >
            SERVER
          </button>
        </div>

        <div className="topbar-meta">
          <span className="live-dot" />

          {status.system === "ONLINE"
            ? "SYSTEM ONLINE"
            : "API OFFLINE"}

          <span className="topbar-divider" />

          <span className="mono">
            {role === "member"
              ? "CLIENT PORTAL"
              : "SERVER OBSERVABILITY"}
          </span>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <div className="eyebrow">
            {role === "member"
              ? "Client-side access"
              : "Server-side view"}
          </div>

          <div className="role-caption">
            {role === "member"
              ? "Identity processing workspace"
              : "Encrypted matching and system status"}
          </div>

          <nav>
            {nav.map(([id, label, Icon]) => (
              <button
                className={
                  activeTab === id
                    ? "nav-item active"
                    : "nav-item"
                }
                key={id}
                onClick={() => {
                  setActiveTab(id);
                  setNotice("");
                }}
              >
                <Icon size={17} />

                {label}

                {activeTab === id && (
                  <ArrowRight
                    size={14}
                    className="nav-arrow"
                  />
                )}
              </button>
            ))}
          </nav>

          <div className="sidebar-foot">
            <LockKeyhole size={16} />

            <div>
              <strong>
                {role === "member"
                  ? "Client-side privacy"
                  : "Server blind by design"}
              </strong>

              <span>
                {role === "member"
                  ? "Original image stays in the client flow."
                  : "Plaintext stays client-side."}
              </span>
            </div>
          </div>
        </aside>

        <main className="content">
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                {role === "member"
                  ? "Client workspace"
                  : "Server workspace"}{" "}
                /{" "}
                {String(
                  nav.findIndex(([id]) => id === activeTab) + 1
                ).padStart(2, "0")}
              </div>

              <h1>{title}</h1>

              <p>
                {role === "member"
                  ? "CIPHERFACE / Privacy-Preserving Face Verification"
                  : "Encrypted templates, homomorphic matching, and system status."}
              </p>
            </div>

            <button
              className="icon-button"
              onClick={refresh}
              title="Refresh system status"
            >
              <RefreshCw size={17} />
            </button>
          </div>

          {notice && (
            <div className="notice">
              <X size={16} />
              {notice}
            </div>
          )}

          {role === "member" && (
            <ClientWorkspace
              action={clientAction}
              setAction={setClientAction}
              busy={busy}
              verifyFile={verifyFile}
              setVerifyFile={setVerifyFile}
              enrollFile={enrollFile}
              setEnrollFile={setEnrollFile}
              name={name}
              setName={setName}
              claimedUser={claimedUser}
              setClaimedUser={setClaimedUser}
              result={result}
              inspection={inspection}
              encryptedEmbedding={encryptedEmbedding}
              submitVerify={submitVerify}
              submitEnroll={submitEnroll}
              status={status}
              templates={templates}
              setNotice={setNotice}
              setResult={setResult}
              setInspection={setInspection}
              setEncryptedEmbedding={setEncryptedEmbedding}
            />
          )}

          {role === "evaluator" &&
            activeTab === "overview" && (
              <Dashboard
                status={status}
                templates={templates}
                statusLoaded={statusLoaded}
                onNavigate={setActiveTab}
              />
            )}

          {role === "evaluator" &&
            activeTab === "monitor" && (
              <Verify
                busy={busy}
                file={verifyFile}
                setFile={setVerifyFile}
                result={result}
                submit={submitVerify}
                threshold={status.verification.threshold}
              />
            )}

          {role === "evaluator" &&
            activeTab === "transfer" && (
              <EncryptedTransferServer templates={templates} />
            )}

          {role === "evaluator" &&
            activeTab === "database" && (
              <DatabaseView templates={templates} />
            )}

          {role === "evaluator" &&
            activeTab === "security" && <Security />}

          {role === "evaluator" &&
            activeTab === "evaluation" && (
              <Evaluation status={status} />
            )}
        </main>
      </div>
    </div>
  );
}

function AuditPanel({ events = [] }) {
  const labels = {
    image_processed_locally: "IMAGE_PROCESSED_LOCAL",
    embedding_generated_locally: "EMBEDDING_GENERATED_LOCAL",
    embedding_encrypted: "CKKS_ENCRYPTION_COMPLETE",
    ciphertext_transmitted: "CIPHERTEXT_TRANSMITTED",
    ciphertext_only_matching: "SERVER_CIPHERTEXT_MATCH",
    encrypted_result_returned: "ENCRYPTED_RESULT_RETURNED",
    decision_completed_locally: "LOCAL_DECISION_COMPLETE",
    template_encrypted: "TEMPLATE_ENCRYPTED_LOCAL",
    ciphertext_template_transmitted: "CIPHERTEXT_TEMPLATE_SENT",
    template_stored_encrypted: "ENCRYPTED_TEMPLATE_STORED",
    server_template_deleted: "ENCRYPTED_TEMPLATE_DELETED",
  };
  return (
    <section className="audit-panel">
      <div className="section-heading">
        <div><span className="eyebrow">Privacy audit / allowlisted events</span><h2>Recent protocol activity</h2></div>
        <span className="audit-retention">IN-MEMORY / BOUNDED</span>
      </div>
      {events.length ? (
        <ol className="audit-list">
          {events.slice(-10).reverse().map((event, index) => (
            <li key={`${event.request_id}-${event.timestamp}-${index}`}>
              <time>{new Date(event.timestamp * 1000).toLocaleTimeString()}</time>
              <span className={`audit-state ${event.outcome}`}>{event.outcome === "failure" ? "!" : "✓"}</span>
              <strong>{labels[event.stage] || event.stage.replaceAll("_", " ").toUpperCase()}</strong>
              <small>{event.latency_ms == null ? event.operation : `${event.latency_ms.toFixed(1)} ms`}</small>
            </li>
          ))}
        </ol>
      ) : <div className="audit-empty">No audited biometric operation in this process session.</div>}
    </section>
  );
}

function PostureLine({ label, value, state }) {
  return <div className="posture-line"><span className={`posture-indicator ${state}`} /><span>{label}</span><strong className={`posture-state ${state}`}>{value}</strong></div>;
}

function CommandCenter({ status, operational, trustVerified, onNavigate, auditEvents, workspace }) {
  const posture = status.privacy_posture || {};
  const activeCount = status.database?.enrolled_users ?? 0;
  return (
    <div className="command-center">
      <section className="command-hero">
        <div className="command-hero-copy">
          <div className="eyebrow"><span className="eyebrow-rule" /> AUTHENTICATION NODE / LOCAL TRUST DOMAIN</div>
          <h2>Authenticate without revealing the biometric.</h2>
          <p>Face processing and decryption stay on this device. The verification service receives encrypted vectors and returns an encrypted similarity.</p>
          <button className="text-button hero-link" onClick={() => onNavigate("privacy")}>VIEW TRUST BOUNDARY <ArrowRight size={14} /></button>
        </div>
        <div className="hero-status-readout">
          <span className="eyebrow">SYSTEM POSTURE</span>
          <strong className={trustVerified ? "verified-text" : "partial-text"}>{trustVerified ? "KEY BOUNDARY VERIFIED" : "PARTIALLY PROTECTED"}</strong>
          <span className="mono">{operational ? "CLIENT + SERVER RESPONDING" : "API UNAVAILABLE"}</span>
          <div className="hero-readout-rule"><span style={{ width: trustVerified ? "100%" : "68%" }} /></div>
          <span className="mono">{activeCount} ENCRYPTED TEMPLATES</span>
        </div>
      </section>

      <div className="command-grid">
        <section className="command-operation">
          <div className="section-heading"><div><span className="eyebrow">Live verification</span><h2>Claimed-identity check</h2></div><span className="route-label"><span /> CLIENT PROCESSING</span></div>
          {workspace}
        </section>
        <aside className="command-aside">
          <section className="posture-panel">
            <div className="section-heading"><div><span className="eyebrow">Security status</span><h2>Verified controls</h2></div><ShieldCheck size={17} /></div>
            <PostureLine label="Client model runtime" value={posture.client_models_loaded ? "ACTIVE" : "UNAVAILABLE"} state={posture.client_models_loaded ? "good" : "bad"} />
            <PostureLine label="Server secret key" value={posture.server_context_secret_free ? "ABSENT" : "UNVERIFIED"} state={posture.server_context_secret_free ? "good" : "bad"} />
            <PostureLine label="Client key custody" value={posture.secret_context_permissions_owner_only && posture.secret_context_outside_project ? "LOCAL / 0600" : "CHECK REQUIRED"} state={posture.secret_context_permissions_owner_only && posture.secret_context_outside_project ? "good" : "bad"} />
            <PostureLine label="Plaintext image route on server" value={posture.server_has_image_route === false ? "NOT EXPOSED" : "UNVERIFIED"} state={posture.server_has_image_route === false ? "good" : "bad"} />
            <PostureLine label="Replay protection" value={posture.replay_protection ? "ACTIVE" : "NOT IMPLEMENTED"} state={posture.replay_protection ? "good" : "partial"} />
            <PostureLine label="Biometric memory clearing" value="BEST-EFFORT" state="partial" />
            <p className="posture-footnote">Array buffers are cleared after encryption; Python and ML allocator copies cannot be guaranteed erased.</p>
          </section>
          <section className="key-custody-panel">
            <div className="eyebrow">Key custody / CKKS</div>
            <div className="key-custody-row"><LockKeyhole size={17} /><span><b>CLIENT SECRET CONTEXT</b><small>{posture.secret_context_permissions_owner_only ? "PRESENT · USER-LOCAL · OWNER ONLY" : "NOT VERIFIED"}</small></span><strong>{posture.secret_context_permissions_owner_only ? "LOCAL" : "CHECK"}</strong></div>
            <div className="key-custody-row public"><UnlockIcon /><span><b>SERVER PUBLIC CONTEXT</b><small>{posture.server_context_secret_free ? "PRESENT · SECRET KEY ABSENT" : "NOT VERIFIED"}</small></span><strong>{posture.server_context_secret_free ? "PUBLIC" : "CHECK"}</strong></div>
          </section>
        </aside>
      </div>

      <PrivacyBoundary />
      <AuditPanel events={auditEvents} />
    </div>
  );
}

function UnlockIcon() {
  return <span className="public-key-mark" aria-hidden="true">PUB</span>;
}

function PrivacyBoundary() {
  const stages = [
    ["01", "IMAGE / CAMERA", "PLAINTEXT", "CLIENT", "client"],
    ["02", "FACE EMBEDDING", "PLAINTEXT", "CLIENT", "client"],
    ["03", "512-D FUSION", "PLAINTEXT", "CLIENT", "client"],
    ["04", "CKKS ENCRYPTION", "KEY CLIENT-ONLY", "CLIENT", "crypto"],
    ["05", "HOMOMORPHIC DOT", "CIPHERTEXT ONLY", "SERVER", "server"],
    ["06", "DECRYPT + DECIDE", "PLAINTEXT SCORE", "CLIENT", "client"],
  ];
  return (
    <section className="privacy-boundary-section">
      <div className="section-heading"><div><span className="eyebrow">Protocol architecture</span><h2>The privacy boundary is the system.</h2></div><span className="boundary-legend"><i className="plaintext-key" /> PLAINTEXT <i className="ciphertext-key" /> ENCRYPTED</span></div>
      <div className="boundary-zones"><span>TRUSTED CLIENT / PLAINTEXT ALLOWED</span><span>UNTRUSTED VERIFIER / CIPHERTEXT ONLY</span></div>
      <div className="boundary-rail">
        {stages.map(([number, title, data, location, kind], index) => (
          <div className={`boundary-node ${kind}`} key={number}>
            {index === 4 && <div className="boundary-wall"><LockKeyhole size={14} /><span>TRUST BOUNDARY / PLAINTEXT NEVER CROSSES</span></div>}
            <span className="boundary-number">{number}</span><strong>{title}</strong><small>{data}</small><em>{location}</em>
          </div>
        ))}
      </div>
    </section>
  );
}

function PrivacyCenter({ status, auditEvents }) {
  const posture = status.privacy_posture || {};
  const rows = [
    ["Original image", "TRUSTED CLIENT ONLY", "verified", "Local upload/camera endpoint is loopback-bound; not accepted by verifier API."],
    ["Face crop", "TRUSTED CLIENT MEMORY", "verified", "Not returned by API and best-effort cleared after embedding."],
    ["Plaintext embedding", "TRUSTED CLIENT MEMORY", "verified", "Never included in the remote JSON request; working arrays are best-effort zeroed."],
    ["Encrypted query/template", "CLIENT → SERVER", "verified", "TenSEAL CKKS serialization is the only biometric vector representation sent."],
    ["Plaintext similarity", "TRUSTED CLIENT ONLY", "verified", "Encrypted score is returned then decrypted locally."],
    ["Secret CKKS context", "TRUSTED CLIENT STORAGE", posture.secret_context_permissions_owner_only && posture.secret_context_outside_project ? "verified" : "partial", "User-local path outside project; permissions are checked at runtime."],
    ["Protocol metadata", "VISIBLE TO SERVER", "partial", "User identifier, timing, ciphertext sizes, and request/access patterns remain visible."],
  ];
  return <div className="privacy-center-page">
    <section className="privacy-center-hero"><span className="eyebrow">Privacy center / observed runtime posture</span><h2>Biometric plaintext stays in the local trust domain.</h2><p>Claims below are derived from the active API configuration and tested route design. Memory clearing remains best-effort, not a guarantee against process or OS memory inspection.</p><span className={`posture-stamp ${posture.server_context_secret_free && posture.secret_context_outside_project ? "verified" : "partial"}`}>{posture.server_context_secret_free && posture.secret_context_outside_project ? "BOUNDARY VERIFIED" : "PARTIALLY PROTECTED"}</span></section>
    <section className="privacy-data-table"><div className="table-head-row"><span>DATA CLASS</span><span>LOCATION / EXPOSURE</span><span>STATUS</span><span>IMPLEMENTATION FACT</span></div>{rows.map(([name, exposure, state, note]) => <div className="privacy-data-row" key={name}><strong>{name}</strong><span>{exposure}</span><span className={`evidence-tag ${state}`}>{state === "verified" ? "VERIFIED" : "PARTIAL"}</span><small>{note}</small></div>)}</section>
    <PrivacyBoundary />
    <AuditPanel events={auditEvents} />
  </div>;
}

function ThreatModel({ status }) {
  const posture = status.privacy_posture || {};
  const threats = [
    ["Read access to encrypted template store", "Stored CKKS vectors are ciphertext; identifiers and metadata remain readable.", "CKKS encryption", posture.server_context_secret_free ? "PROTECTED AGAINST PLAINTEXT READ" : "NOT VERIFIED", "verified"],
    ["Network interception", "Ciphertexts and protocol metadata may be observed; local demo uses loopback HTTP.", "Use HTTPS/TLS for remote deployment.", "PARTIALLY PROTECTED", "partial"],
    ["Replay of an authenticated mutation", "Reusing a valid enrollment/query request could repeat an operation.", "One-time nonce + timestamp window in one-process cache.", posture.replay_protection ? "PROTECTED IN SINGLE PROCESS" : "NOT IMPLEMENTED", posture.replay_protection ? "verified" : "bad"],
    ["Unauthorized API access", "A shared bearer secret can be misused if exposed; no per-user accounts or enrollment ownership.", "Environment token and constant-time comparison; replace with managed identities.", posture.server_authentication ? "PARTIAL / SHARED TOKEN" : "NOT CONFIGURED", "partial"],
    ["Excessive verification attempts", "Repeated attempts can enable abuse or probing.", "Per-IP in-memory rate limiting; not shared across workers.", posture.rate_limiting ? "PARTIAL / PROCESS LOCAL" : "NOT IMPLEMENTED", "partial"],
    ["Malicious verifier result", "A malicious server may return a fabricated similarity ciphertext.", "No proof of correct computation or malicious-secure protocol.", "NOT PROTECTED", "bad"],
    ["Client compromise / memory inspection", "A compromised client can access images, vectors, and secret key while processing.", "Trusted-device assumption; best-effort mutable-buffer clearing.", "NOT PROTECTED", "bad"],
    ["Presentation attack / spoofed face", "A photograph or replay may be presented to camera.", "No liveness or anti-spoofing model is implemented.", "NOT IMPLEMENTED", "bad"],
    ["Traffic analysis / template linkage", "Timing, size, user ID, and access patterns remain visible.", "No padding, PIR, or unlinkable template scheme.", "NOT PROTECTED", "bad"],
    ["Sensitive operational logging", "Logs may accidentally retain biometric data.", "Allowlisted audit schema excludes vectors, image content, names, and ciphertext.", "PROTECTED BY CURRENT LOGGING PATH", "verified"],
  ];
  return <div className="threat-model-page"><section className="threat-intro"><span className="eyebrow">Threat model / honest-but-curious verifier</span><h2>Explicit assumptions. No greenwashing.</h2><p>The server follows the protocol but inspects all data it receives and stores. This prototype does not protect against malicious computation, compromised client devices, or liveness attacks.</p></section><div className="threat-table"><div className="threat-table-header"><span>THREAT</span><span>RISK / MITIGATION</span><span>ASSESSMENT</span></div>{threats.map(([name, risk, mitigation, state, tone]) => <article className="threat-row" key={name}><h3>{name}</h3><div><p>{risk}</p><small>{mitigation}</small></div><span className={`evidence-tag ${tone}`}>{state}</span></article>)}</div></div>;
}

function ResearchView({ research }) {
  if (!research) return <div className="research-loading">Loading saved experiment artifacts…</div>;
  const modelEntries = Object.entries(research.models || {});
  const ckks = research.ckks || {};
  const error = ckks.similarity_error || {};
  const timing = ckks.timing_seconds || {};
  return <div className="research-page">
    <section className="research-intro"><span className="eyebrow">LFW / official DevTest / frozen validation configuration</span><h2>Measured system performance</h2><p>{research.pair_count} scored pairs · validation-selected alpha and thresholds · artifacts loaded from the recorded results directory.</p></section>
    <div className="research-model-table"><div className="research-table-header"><span>MODEL</span><span>ACCURACY</span><span>FAR</span><span>FRR</span><span>EER</span><span>AUC</span><span>THRESHOLD</span></div>{modelEntries.map(([name, metric]) => <div className="research-table-row" key={name}><strong>{name.replaceAll("_", " ").toUpperCase()}</strong><span>{(metric.accuracy * 100).toFixed(1)}%</span><span>{(metric.far * 100).toFixed(1)}%</span><span>{(metric.frr * 100).toFixed(1)}%</span><span>{(metric.eer_estimate * 100).toFixed(1)}%</span><span>{metric.auc.toFixed(5)}</span><span>{metric.threshold.toFixed(5)}</span></div>)}</div>
    <div className="research-metric-grid">
      <div className="research-metric"><span>FUSION SELECTION</span><strong>{research.fusion_selection?.method?.replaceAll("_", " ")}</strong><small>alpha {research.fusion_selection?.alpha} · validation AUC {(research.fusion_selection?.validation_auc ?? 0).toFixed(5)}</small></div>
      <div className="research-metric"><span>CKKS ABS ERROR / MEAN</span><strong>{Number(error.absolute_mean).toExponential(2)}</strong><small>max {Number(error.absolute_max).toExponential(2)}</small></div>
      <div className="research-metric"><span>CKKS ENCRYPT / PAIR</span><strong>{((timing.per_pair_encryption || 0) * 1000).toFixed(2)} ms</strong><small>measured on recorded run</small></div>
      <div className="research-metric"><span>HOMOMORPHIC / PAIR</span><strong>{((timing.per_pair_homomorphic || 0) * 1000).toFixed(2)} ms</strong><small>deserialize + dot + serialize</small></div>
      <div className="research-metric"><span>LOCAL DECRYPT / PAIR</span><strong>{((timing.per_pair_decryption || 0) * 1000).toFixed(2)} ms</strong><small>client-side scalar decryption</small></div>
      <div className="research-metric"><span>END-TO-END TEST BATCH</span><strong>{Number(timing.end_to_end_total || 0).toFixed(1)} s</strong><small>includes local detection and embedding</small></div>
      <div className="research-metric"><span>MEAN QUERY CIPHERTEXT</span><strong>{(Number(ckks.ciphertext_bytes?.query_mean || 0) / 1024).toFixed(1)} KB</strong><small>serialized CKKS vector</small></div>
    </div>
    <section className="fusion-grid"><div className="section-heading"><div><span className="eyebrow">Validation grid search</span><h2>All fusion candidates</h2></div></div><div className="fusion-bars">{(research.fusion_weight_grid || []).map((row) => <div className="fusion-bar-row" key={`${row.method}-${row.alpha}`}><span>{row.method.replaceAll("_", " ")} / {row.alpha ?? "baseline"}</span><div><i style={{ width: `${Math.max(0, Math.min(100, Number(row.auc) * 100))}%` }} /></div><b>{Number(row.auc).toFixed(5)}</b></div>)}</div></section>
    <p className="research-caveat">AUC/EER are calculated on the recorded test set for reporting only. Configuration selection and operating thresholds came from validation. Benchmark metrics do not imply deployment performance.</p>
  </div>;
}

function StatusCard({
  icon: Icon,
  label,
  value,
  tone = "teal",
}) {
  return (
    <div className={`status-card ${tone}`}>
      <div className="status-icon">
        <Icon size={18} />
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>

      <Check size={15} className="status-check" />
    </div>
  );
}

function Dashboard({
  status,
  templates,
  statusLoaded,
  onNavigate,
}) {
  const count = statusLoaded
    ? status.database.enrolled_users
    : "UNAVAILABLE";

  return (
    <>
      <section className="metric-grid">
        <StatusCard
          icon={Zap}
          label="System state"
          value={
            statusLoaded ? status.system : "API UNAVAILABLE"
          }
        />

        <StatusCard
          icon={LockKeyhole}
          label="Encryption"
          value={
            statusLoaded && status.encryption.scheme
              ? "CKKS ACTIVE"
              : "UNAVAILABLE"
          }
        />

        <StatusCard
          icon={Database}
          label="Protected templates"
          value={
            statusLoaded
              ? `${status.database.enrolled_users} ENROLLED`
              : "UNAVAILABLE"
          }
        />

        <StatusCard
          icon={Server}
          label="Server visibility"
          value={
            statusLoaded
              ? status.encryption.server_mode || "UNKNOWN"
              : "UNAVAILABLE"
          }
          tone="amber"
        />
      </section>

      <section className="dashboard-grid">
        <div className="panel pipeline-panel">
          <PanelTitle
            label="Client to server path"
            title="Encrypted verification flow"
            action="MONITOR"
            onClick={() => onNavigate("monitor")}
          />

          <Pipeline completed={false} />
        </div>

        <div className="side-stack">
          <div className="panel health-panel">
            <PanelTitle
              label="System status"
              title="Pipeline components"
            />

            <HealthRow
              label="MTCNN detection"
              ok={statusLoaded && status.models.mtcnn}
            />

            <HealthRow
              label="FaceNet encoder"
              ok={statusLoaded && status.models.facenet}
            />

            <HealthRow
              label="ArcFace encoder"
              ok={statusLoaded && status.models.arcface}
            />

            <HealthRow
              label="CKKS context"
              ok={
                statusLoaded &&
                Boolean(status.encryption.scheme)
              }
            />

            <HealthRow
              label="Metadata database"
              ok={
                statusLoaded && status.database.status
              }
            />
          </div>

          <div className="panel evaluation-panel">
            <div className="eyebrow">Saved research artifacts</div>
            <strong className="research-available">LFW DEVTEST / CKKS</strong>
            <p>Model comparisons and encryption measurements are available in the Research view.</p>
            <button className="text-button" onClick={() => onNavigate("research")}>OPEN RESEARCH <ArrowRight size={14} /></button>
          </div>
        </div>
      </section>

      <section className="bottom-strip">
        <div>
          <span className="eyebrow">Decision threshold</span>
          <strong>
            {statusLoaded
              ? status.verification.threshold
              : "Unavailable"}
          </strong>
        </div>

        <div>
          <span className="eyebrow">Template count</span>
          <strong>
            {count === "UNAVAILABLE"
              ? count
              : `${count} encrypted records`}
          </strong>
        </div>

        <div>
          <span className="eyebrow">CKKS scale</span>
          <strong>
            {statusLoaded
              ? status.encryption.global_scale ||
                "Unavailable"
              : "Unavailable"}
          </strong>
        </div>
      </section>
    </>
  );
}

function PanelTitle({
  label,
  title,
  action,
  onClick,
}) {
  return (
    <div className="panel-title">
      <div>
        <div className="eyebrow">{label}</div>
        <h2>{title}</h2>
      </div>

      {action && (
        <button
          className="text-button"
          onClick={onClick}
        >
          {action}
          <ArrowRight size={14} />
        </button>
      )}
    </div>
  );
}

function HealthRow({ label, ok }) {
  return (
    <div className="health-row">
      <span
        className={
          ok
            ? "health-dot"
            : "health-dot muted"
        }
      />

      {label}

      <strong>
        {ok ? "READY" : "UNAVAILABLE"}
      </strong>
    </div>
  );
}

function Pipeline({ completed }) {
  return (
    <div className="pipeline">
      <div className="zone-label client-label">
        <span>CLIENT / PLAINTEXT</span>
        <small>Trust zone A</small>
      </div>

      {pipeline.map(
        ([label, detail, zone], index) => (
          <div
            className="pipeline-item"
            key={label}
          >
            {label ===
              "Encrypted template database" && (
              <div className="zone-label server-label">
                <span>
                  SERVER / CIPHERTEXT ONLY
                </span>
                <small>
                  No image. No plaintext. No secret key.
                </small>
              </div>
            )}

            <div
              className={`pipeline-node ${zone}`}
            >
              <div className="node-index">
                {String(index + 1).padStart(2, "0")}
              </div>

              <div>
                <strong>{label}</strong>
                <span>{detail}</span>
              </div>

              <div className="node-state">
                {completed ? (
                  <Check size={13} />
                ) : zone === "server" ? (
                  <Network size={13} />
                ) : (
                  <span />
                )}
              </div>
            </div>

            {label === "Encryption" && (
              <div className="pipeline-divider">
                TRUST BOUNDARY / CLIENT TO SERVER
              </div>
            )}

            {label ===
              "Encrypted similarity" && (
              <div className="pipeline-divider return">
                TRUST BOUNDARY / SERVER TO CLIENT
              </div>
            )}
          </div>
        )
      )}

      <div className="boundary-note">
        <LockKeyhole size={14} />
        Server computes on ciphertext. Client
        decrypts the result.
      </div>
    </div>
  );
}


function LFWBrowser({ setFile, setName }) {
  const [search, setSearch] = useState("");
  const [people, setPeople] = useState([]);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  async function searchPeople(value) {
    setSearch(value);

    if (!value.trim()) {
      setPeople([]);
      setSelectedPerson(null);
      return;
    }

    setLoading(true);

    try {
      const response = await getLfwIdentities(value);
      setPeople(response.identities || response || []);
    } catch (error) {
      console.error(error);
      setPeople([]);
    } finally {
      setLoading(false);
    }
  }

  async function selectPerson(person) {
    const personName = person.name || person;

    try {
      const response = await getLfwIdentity(personName);
      setSelectedPerson(response);
    } catch (error) {
      console.error(error);
      setSelectedPerson(null);
    }
  }

  async function selectImage(person, filename) {
    setSelectedImage("CLICK WORKS: " + filename);
    console.log("LFW IMAGE CLICKED:", person, filename);

    try {
      const url =
        `${API_BASE}/api/lfw/image/${encodeURIComponent(person)}/${encodeURIComponent(filename)}`;

      console.log("FETCHING:", url);

      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`Image request failed: ${response.status}`);
      }

      const blob = await response.blob();

      const file = new File(
        [blob],
        filename,
        { type: blob.type || "image/jpeg" }
      );

      console.log("FILE CREATED:", file.name, file.size);

      setFile(file);
      setSelectedImage(filename);
    } catch (error) {
      console.error("LFW IMAGE ERROR:", error);
      setSelectedImage(`ERROR: ${error.message}`);
    }
  }

  return (
    <div className="lfw-browser panel">
      <div className="lfw-browser-header">
        <div>
          <span className="eyebrow">LFW DATASET</span>
          <strong>Choose a real LFW face</strong>
        </div>
        <span>13,233 images · 5,749 identities</span>
      </div>

      <input
        className="lfw-search"
        type="text"
        placeholder="Search person, e.g. Aaron Peirsol"
        value={search}
        onChange={(event) => searchPeople(event.target.value)}
      />

      {loading && (
        <div className="lfw-loading">Searching LFW...</div>
      )}

      {people.length > 0 && (
        <div className="lfw-people">
          {people.slice(0, 20).map((person) => {
            const personName = person.name || person;

            return (
              <button
                type="button"
                key={personName}
                className="lfw-person"
                onClick={() => selectPerson(person)}
              >
                {personName}
              </button>
            );
          })}
        </div>
      )}

      {selectedImage && (
        <div style={{ marginTop: "12px", fontSize: "12px", opacity: 0.8 }}>
          LFW image selected: <strong>{selectedImage}</strong>
        </div>
      )}

      {selectedImage && (
        <div
          style={{
            marginTop: "12px",
            padding: "10px 12px",
            borderRadius: "8px",
            background: "rgba(110,168,255,0.12)",
            fontSize: "13px"
          }}
        >
          LFW image selected: <strong>{selectedImage}</strong>
        </div>
      )}

      {selectedPerson && (
        <div className="lfw-images">
          <div className="lfw-person-title">
            <strong>{selectedPerson.name}</strong>
            <span>{selectedPerson.image_count} images</span>
          </div>

          <div className="lfw-image-grid">
            {selectedPerson.images.map((filename) => {
              const imageUrl =
                `${API_BASE}/api/lfw/image/${encodeURIComponent(
                  selectedPerson.name
                )}/${encodeURIComponent(filename)}`;

              return (
                <div
                  className="lfw-image-card"
                  key={filename}
                >
                  <img
                    src={imageUrl}
                    alt={filename}
                    draggable={false}
                    style={{
                      pointerEvents: "none"
                    }}
                  />

                  <span>{filename}</span>

                  <button
                    type="button"
                    onMouseDown={(event) => {
                      event.preventDefault();
                      event.stopPropagation();

                      setSelectedImage(filename);
                      if (setName) {
                        setName(selectedPerson.name);
                      }

                      fetch(imageUrl)
                        .then((response) => {
                          if (!response.ok) {
                            throw new Error(
                              `Image request failed: ${response.status}`
                            );
                          }
                          return response.blob();
                        })
                        .then((blob) => {
                          const file = new File(
                            [blob],
                            filename,
                            {
                              type: blob.type || "image/jpeg"
                            }
                          );

                          setFile(file);
                        })
                        .catch((error) => {
                          console.error(
                            "LFW ERROR:",
                            error
                          );
                          setSelectedImage(
                            "ERROR: " + error.message
                          );
                        });
                    }}
                    style={{
                      width: "100%",
                      padding: "10px",
                      marginTop: "6px",
                      border: "none",
                      borderRadius: "7px",
                      background: "#6ea8ff",
                      color: "#05070b",
                      fontWeight: "700",
                      cursor: "pointer"
                    }}
                  >
                    USE THIS IMAGE
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function CameraCapture({ file, setFile, disabled }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [error, setError] = useState("");
  const [cameraOpen, setCameraOpen] = useState(false);
  const [previewUrl, setPreviewUrl] = useState("");

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
  }

  async function openCamera() {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera capture is unavailable in this browser. Use image upload instead.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      setCameraOpen(true);
    } catch (cameraError) {
      setError(
        cameraError.name === "NotAllowedError"
          ? "Camera permission was denied. Allow access or use image upload."
          : "No usable camera was found. Use image upload instead.",
      );
    }
  }

  useEffect(() => {
    if (videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [cameraOpen]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  function captureFrame() {
    const video = videoRef.current;
    if (!video?.videoWidth || !video?.videoHeight) {
      setError("Camera preview is not ready yet.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) {
        setError("Could not capture the camera image.");
        return;
      }
      setFile(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
      stopCamera();
    }, "image/jpeg", 0.92);
  }

  function retake() {
    setFile(null);
    openCamera();
  }

  return (
    <div className="camera-capture">
      {cameraOpen && (
        <video ref={videoRef} className="camera-preview" playsInline muted />
      )}
      {!cameraOpen && previewUrl && (
        <img className="camera-preview" src={previewUrl} alt="Captured image preview" />
      )}
      <div className="camera-actions">
        {cameraOpen ? (
          <>
            <button type="button" className="secondary-button" disabled={disabled} onClick={captureFrame}>
              <Camera size={16} /> CAPTURE STILL
            </button>
            <button type="button" className="secondary-button" onClick={stopCamera}>
              CLOSE CAMERA
            </button>
          </>
        ) : file ? (
          <button type="button" className="secondary-button" disabled={disabled} onClick={retake}>
            <RefreshCw size={15} /> RETAKE
          </button>
        ) : (
          <button type="button" className="secondary-button" disabled={disabled} onClick={openCamera}>
            <Camera size={16} /> OPEN CAMERA
          </button>
        )}
      </div>
      {error && <div className="camera-error" role="alert">{error}</div>}
      <span className="camera-privacy">One still is captured locally. No frames are streamed to the server.</span>
    </div>
  );
}

function UploadBox({
  file,
  setFile,
  label,
}) {
  async function handleDrop(event) {
    event.preventDefault();

    const droppedFile = event.dataTransfer.files?.[0];

    if (droppedFile) {
      setFile(droppedFile);
      return;
    }

    const url =
      event.dataTransfer.getData("text/uri-list") ||
      event.dataTransfer.getData("text/plain");

    if (!url) return;

    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error("Could not load dropped image.");
      }

      const blob = await response.blob();

      const filename =
        url.split("/").pop() || "lfw-image.jpg";

      const newFile = new File(
        [blob],
        filename,
        { type: blob.type || "image/jpeg" }
      );

      setFile(newFile);
    } catch (error) {
      console.error("DROP ERROR:", error);
    }
  }

  return (
    <label
      className={`upload-box ${
        file ? "has-file" : ""
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
      }}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept="image/*"
        onChange={(event) =>
          setFile(
            event.target.files?.[0] || null
          )
        }
      />

      <Upload size={22} />

      <strong>
        {file ? file.name : label}
      </strong>

      <span>
        {file
          ? `${(
              file.size /
              1024 /
              1024
            ).toFixed(2)} MB selected`
          : "Drag an LFW image here, or choose a file"}
      </span>
    </label>
  );
}

function ClientWorkspace({
  action,
  setAction,
  busy,
  verifyFile,
  setVerifyFile,
  enrollFile,
  setEnrollFile,
  name,
  setName,
  claimedUser,
  setClaimedUser,
  result,
  inspection,
  encryptedEmbedding,
  submitVerify,
  submitEnroll,
  status,
  templates,
  setNotice,
  setResult,
  setInspection,
  hideActionSelector = false,
  conferenceMode = false,
}) {
  const resetVerification = () => {
    setVerifyFile(null);
    setResult(null);
    setInspection(null);
    setEncryptedEmbedding("");
    setNotice("");
    setAction("verify");
  };

  const noIdentities =
    status.database.enrolled_users === 0;

  return (
    <div className={`client-workspace ${hideActionSelector ? "hide-action-selector" : ""}`}>
      {!hideActionSelector && <div className="client-actions">
        <button
          className={
            action === "verify"
              ? "client-action active"
              : "client-action"
          }
          onClick={() => {
            setAction("verify");
            setNotice("");
          }}
        >
          <ScanFace size={18} />

          <span>
            <strong>VERIFY IDENTITY</strong>
            <small>
              Check an enrolled identity
            </small>
          </span>
        </button>

        <button
          className={
            action === "enroll"
              ? "client-action enroll-action active"
              : "client-action enroll-action"
          }
          onClick={() => {
            setAction("enroll");
            setNotice("");
          }}
        >
          <UserRoundPlus size={18} />

          <span>
            <strong>ENROLL NEW USER</strong>
            <small>
              Create an encrypted template
            </small>
          </span>
        </button>
      </div>}

      <div className="privacy-boundary-banner">
        <LockKeyhole size={16} />

        <span>
          <strong>CLIENT ONLY:</strong> original face
          image and plaintext processing
        </span>

        <span className="boundary-arrow">
          →
        </span>

        <span>
          <strong>CROSSES:</strong> encrypted embedding
          only
        </span>
      </div>

      {action === "verify" && (
        <div className="client-flow">
          <div className="client-flow-header">
            <div>
              <div className="eyebrow">
                Client verification
              </div>

              <h2>
                {result
                  ? "Verification complete"
                  : "Upload a face to verify"}
              </h2>
            </div>

            {result && (
              <button
                className="secondary-button"
                onClick={resetVerification}
              >
                VERIFY ANOTHER FACE
                <ArrowRight size={16} />
              </button>
            )}
          </div>

          {noIdentities && !result ? (
            <div className="setup-state">
              <Database size={26} />

              <h2>No enrolled identities</h2>

              <p>
                Enroll the first identity to begin.
              </p>

              <button
                className="primary-button"
                onClick={() =>
                  setAction("enroll")
                }
              >
                ENROLL FIRST USER
                <UserRoundPlus size={17} />
              </button>
            </div>
          ) : (
            <>
              <LFWBrowser setFile={setVerifyFile} />

              <div className="panel operation-panel">
                <form onSubmit={submitVerify}>
                  <label className="field-label">
                    Claimed identity
                    <select value={claimedUser} onChange={(event) => setClaimedUser(event.target.value)} required>
                      <option value="" disabled>Select enrolled identity</option>
                      {templates.map((template) => (
                        <option key={template.name} value={template.name}>{template.name}</option>
                      ))}
                    </select>
                  </label>

                  <CameraCapture file={verifyFile} setFile={setVerifyFile} disabled={busy} />

                  <UploadBox
                    file={verifyFile}
                    setFile={setVerifyFile}
                    label="Select original face image"
                  />

                  <button
                    className="primary-button"
                    disabled={busy}
                  >
                    {busy
                      ? "PROCESSING FACE VERIFICATION"
                      : "VERIFY IDENTITY"}

                    <ArrowRight size={17} />
                  </button>
                </form>
              </div>

              <CompactPipeline
                file={verifyFile}
                result={result}
              />

              {result && <ClientResult result={result} />}

              {result && encryptedEmbedding && (
                <EncryptedEmbeddingClient value={encryptedEmbedding} />
              )}

              {conferenceMode && inspection && (
                <PipelineStepper inspection={inspection} file={verifyFile} completed />
              )}

            </>
          )}
        </div>
      )}

      {action === "enroll" && (
        <div className="client-flow enrollment-view">
          <div className="client-flow-header">
            <div>
              <div className="eyebrow">
                Client enrollment
              </div>

              <h2>
                Enroll a new identity
              </h2>
            </div>

            <span className="count-badge">
              {status.database.enrolled_users} enrolled
            </span>
          </div>

          <LFWBrowser
            setFile={setEnrollFile}
            setName={setName}
          />

          <div className="panel operation-panel">
            <form onSubmit={submitEnroll}>
              <label className="field-label">
                Person name

                <input
                  value={name}
                  onChange={(event) =>
                    setName(event.target.value)
                  }
                  placeholder="e.g. Jordan Lee"
                  maxLength={80}
                />
              </label>

              <CameraCapture file={enrollFile} setFile={setEnrollFile} disabled={busy} />

              <UploadBox
                file={enrollFile}
                setFile={setEnrollFile}
                label="Select original face image"
              />

              <button
                className="primary-button"
                disabled={busy}
              >
                {busy
                  ? "ENCRYPTING AND STORING"
                  : "ENROLL NEW USER"}

                <LockKeyhole size={17} />
              </button>
            </form>
          </div>

          <div className="privacy-note">
            <LockKeyhole size={15} />

            <span>
              Original image stays in the client
              processing flow; only the encrypted
              template is stored.
            </span>
          </div>

          <div className="enrollment-summary">
            <span>
              Current encrypted identities
            </span>

            <strong>
              {status.database.enrolled_users}
            </strong>
          </div>
        </div>
      )}
    </div>
  );
}

function EncryptedEmbeddingClient({
  value,
}) {
  const [copied, setCopied] = useState(false);

  if (!value) return null;

  const copy = async () => {
    await navigator.clipboard.writeText(value);

    setCopied(true);

    setTimeout(
      () => setCopied(false),
      1600
    );
  };

  return (
    <div className="transfer-panel client-transfer">
      <div className="transfer-heading">
        <div>
          <div className="eyebrow">
            Encrypted Embedding Transfer Demo
          </div>

          <h3>Actual CKKS ciphertext sent to server</h3>

          <p>
            Base64 serialization of the encrypted query embedding transmitted
            from the local client to port 8002. This is ciphertext, not the
            plaintext embedding or secret key.
          </p>
        </div>

        <LockKeyhole size={22} />
      </div>

      <textarea
        readOnly
        value={value}
        aria-label="Serialized encrypted embedding"
      />

      <button
        className="secondary-button"
        onClick={copy}
      >
        {copied
          ? "COPIED"
          : "COPY ENCRYPTED EMBEDDING"}

        <Check size={15} />
      </button>
    </div>
  );
}

function EncryptedTransferServer({ templates }) {
  const [value, setValue] = useState("");
  const [userId, setUserId] = useState("");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async () => {
    if (!value.trim() || !userId) {
      return setError(
        "Choose a claimed identity and paste an encrypted embedding."
      );
    }

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const verification = await verifyEncryptedEmbedding(value, userId);

      setResult(verification);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="transfer-server-view">
      <div className="transfer-server-header">
        <div>
          <div className="eyebrow">
            Server / ciphertext-only input
          </div>

          <h2>
            Encrypted Embedding Transfer Demo
          </h2>

          <p>
            Paste the serialized embedding produced
            by the Client. This endpoint accepts
            ciphertext, never an image.
          </p>
        </div>

        <Server size={28} />
      </div>

      <div className="privacy-boundary-banner server-transfer-banner">
        <LockKeyhole size={16} />

        <span>
          <strong>SERVER RECEIVES:</strong> encrypted
          embedding only
        </span>

        <span>
          <strong>SERVER NEVER RECEIVES:</strong>{" "}
          original image, plaintext embedding, or
          secret key
        </span>
      </div>

      <div className="panel transfer-panel">
        <label className="field-label">
          Claimed identity
          <select value={userId} onChange={(event) => setUserId(event.target.value)} required>
            <option value="" disabled>Select enrolled identity</option>
            {templates.map((template) => (
              <option key={template.name} value={template.name}>{template.name}</option>
            ))}
          </select>
        </label>

        <label className="field-label">
          Serialized CKKS ciphertext / Base64

          <textarea
            value={value}
            onChange={(event) =>
              setValue(event.target.value)
            }
            placeholder="Paste encrypted embedding here"
          />
        </label>

        <button
          className="primary-button"
          disabled={busy}
          onClick={submit}
        >
          {busy
            ? "MATCHING CIPHERTEXT"
            : "VERIFY ENCRYPTED EMBEDDING"}

          <ArrowRight size={17} />
        </button>

        {error && (
          <div className="notice">
            <X size={16} />
            {error}
          </div>
        )}
      </div>

      {result && (
        <div
          className={`client-result-card ${
            result.decision === "ACCEPT"
              ? "accepted"
              : "rejected"
          }`}
        >
          <div className="result-kicker">
            ENCRYPTED VERIFICATION COMPLETE
          </div>

          <h2>{result.best_match}</h2>

          <div className="client-result-grid">
            <span>Similarity</span>
            <strong>
              {result.similarity.toFixed(4)}
            </strong>

            <span>Threshold</span>
            <strong>{result.threshold}</strong>

            <span>Decision</span>
            <strong>{result.decision}</strong>
          </div>
        </div>
      )}
    </div>
  );
}

function ClientFlowStages() {
  return (
    <div className="client-stages">
      <span>
        ORIGINAL IMAGE / CLIENT ONLY
      </span>

      <ArrowRight size={14} />

      <span>DETECT + ALIGN</span>

      <ArrowRight size={14} />

      <span>FACENET + ARCFACE</span>

      <ArrowRight size={14} />

      <span>FUSE + NORMALIZE</span>

      <ArrowRight size={14} />

      <span>CKKS ENCRYPTION</span>

      <ArrowRight size={14} />

      <span>SERVER: CIPHERTEXT ONLY</span>

      <ArrowRight size={14} />

      <span>CLIENT DECRYPTS</span>

      <ArrowRight size={14} />

      <span>ACCEPT / REJECT</span>
    </div>
  );
}

function CompactPipeline({
  file,
  result,
}) {
  const stages = [
    [
      "Original image",
      file
        ? "Selected locally"
        : "Awaiting image",
      "client",
    ],
    [
      "MTCNN + alignment",
      "Face detection and crop",
      "client",
    ],
    [
      "FaceNet + ArcFace",
      "512-D + 512-D",
      "ai",
    ],
    [
      "Fusion",
      "512-D weighted embedding",
      "ai",
    ],
    [
      "L2 normalization",
      "512-D normalized",
      "ai",
    ],
    [
      "CKKS encryption",
      "Encrypted embedding",
      "encryption",
    ],
    [
      "Server matching",
      "Ciphertext only",
      "server",
    ],
    [
      "Similarity",
      result && Number.isFinite(Number(result.similarity))
        ? `${Number(result.similarity).toFixed(4)} / threshold ${
            result.threshold ?? "N/A"
          }`
        : "Awaiting verification",
      "encryption",
    ],
    [
      "Decision",
      result
        ? result.decision
        : "Awaiting verification",
      result?.decision === "ACCEPT"
        ? "accept"
        : result?.decision === "REJECT"
        ? "reject"
        : "client",
    ],
  ];

  return (
    <div className="compact-pipeline">
      {stages.map(
        ([label, detail, tone], index) => (
          <div
            className={`compact-stage ${tone}`}
            key={label}
          >
            <span className="compact-index">
              {String(index + 1).padStart(
                2,
                "0"
              )}
            </span>

            <div>
              <strong>{label}</strong>
              <small>{detail}</small>
            </div>

            {index <
              stages.length - 1 && (
              <ArrowRight size={14} />
            )}
          </div>
        )
      )}
    </div>
  );
}

function PipelineStepper({
  inspection,
  file,
  completed = false,
}) {
  const [step, setStep] = useState(0);

  const stages = [
    {
      title: "Original image",
      detail:
        "CLIENT ONLY - never crosses the trust boundary",
      body: file ? (
        <LocalPreview
          file={file}
          className="original-preview"
          alt="Selected original face"
        />
      ) : (
        <div className="visual-placeholder">
          Original image selected locally
        </div>
      ),
    },

    {
      title: "MTCNN face detection",
      detail: `${inspection.detection.faces_detected} face(s); largest bounding-box area selected. Confidence ${inspection.detection.confidence}`,
      body: file ? (
        <DetectionPreview
          file={file}
          detection={inspection.detection}
        />
      ) : null,
    },

    {
      title: "Five-landmark alignment",
      detail: "Canonical similarity transform; crop remains local",
      body: <div className="visual-placeholder">Aligned face is not returned by the local API.</div>,
    },

    {
      title: "Feature extraction",
      detail: "Two real model outputs",
      body: (
        <div className="feature-readout">
          <FeatureMetric
            label="FaceNet"
            value={`${inspection.features.facenet.dimensions}-D`}
            timing={
              inspection.features.facenet
                .timing_ms
            }
          />

          <FeatureMetric
            label="ArcFace"
            value={`${inspection.features.arcface.dimensions}-D`}
            timing={
              inspection.features.arcface
                .timing_ms
            }
          />
        </div>
      ),
    },

    {
      title: "Fusion + L2 normalization",
      detail: `${inspection.fusion.input_dimensions.join(
        " + "
      )} -> ${
        inspection.fusion.output_dimensions
      }-D normalized embedding`,
      body: (
        <div className="feature-readout">
          <FeatureMetric
            label="Fused vector"
            value={`${inspection.fusion.output_dimensions}-D`}
          />

          <FeatureMetric
            label="L2 norm"
            value={
              inspection.normalization.l2_norm
            }
          />
        </div>
      ),
    },

    {
      title: "CKKS encryption",
      detail:
        "Plaintext embedding -> CKKS -> encrypted embedding",
      body: (
        <div className="encryption-transition">
          <span>PLAINTEXT EMBEDDING</span>

          <ArrowRight size={18} />

          <b>CKKS</b>

          <ArrowRight size={18} />

          <span>ENCRYPTED EMBEDDING</span>
        </div>
      ),
    },

    {
      title:
        "Trust boundary: Client -> Server",
      detail:
        "Only the encrypted embedding crosses",
      body: (
        <div className="boundary-stage">
          <LockKeyhole size={25} />

          <strong>
            ONLY ENCRYPTED EMBEDDING CROSSES
          </strong>

          <span>
            Original image, plaintext embedding,
            and secret key remain on the Client.
          </span>
        </div>
      ),
    },

    {
      title: "Send encrypted probe",
      detail: "Ciphertext only",
      body: (
        <div className="server-only">
          <Network size={25} />

          <strong>Encrypted probe</strong>

          <span>
            The Server receives no original image
            or plaintext embedding.
          </span>
        </div>
      ),
    },

    {
      title:
        "Encrypted template database",
      detail: "Already stored on the Server",
      body: (
        <div className="server-only">
          <Database size={25} />

          <strong>
            Server-resident encrypted templates
          </strong>

          <span>
            Stored ciphertext records are not
            transmitted during this request.
          </span>
        </div>
      ),
    },

    {
      title: "Homomorphic matching",
      detail: "Encrypted dot product",
      body: (
        <div className="server-only">
          <Server size={25} />

          <strong>
            Server computes on ciphertext
          </strong>

          <span>
            Encrypted probe + encrypted templates
            -&gt; encrypted similarity
          </span>
        </div>
      ),
    },

    {
      title: "Encrypted similarity",
      detail:
        "Return ciphertext to the Client",
      body: (
        <div className="encryption-transition">
          <span>ENCRYPTED SIMILARITY</span>

          <ArrowRight size={18} />

          <b>RETURNED</b>
        </div>
      ),
    },

    {
      title:
        "Trust boundary: Server -> Client",
      detail:
        "Encrypted similarity crosses back",
      body: (
        <div className="boundary-stage return">
          <LockKeyhole size={25} />

          <strong>
            ENCRYPTED SIMILARITY ONLY
          </strong>

          <span>
            The Server does not decrypt the score.
          </span>
        </div>
      ),
    },

    {
      title: "Client decision",
      detail: completed
        ? "Encrypted similarity decrypted on the Client"
        : "Ready for verification result",
      body: (
        <div className="client-decision">
          <LockKeyhole size={22} />

          <span>CLIENT DECRYPTS</span>

          <ArrowRight size={18} />

          <span>THRESHOLD {result?.threshold ?? "validation-selected"}</span>

          <ArrowRight size={18} />

          <b>ACCEPT / REJECT</b>
        </div>
      ),
    },
  ];

  const current = stages[step];
  const location = step >= 8 && step <= 9 ? "SERVER" : step === 7 || step === 10 || step === 11 ? "CLIENT ↔ SERVER" : "CLIENT";
  const dataState = step <= 4 || step === 12 ? "PLAINTEXT / CLIENT ONLY" : step === 5 ? "PLAINTEXT → CIPHERTEXT" : "CIPHERTEXT";
  const stageTiming = ({
    1: inspection.timing_ms?.detection,
    3: inspection.features.facenet.timing_ms,
    4: inspection.features.arcface.timing_ms,
    5: inspection.timing_ms?.fusion_normalization,
    6: inspection.timing_ms?.encryption,
    9: inspection.timing_ms?.server_roundtrip,
    12: inspection.timing_ms?.decryption,
  })[step];

  return (
    <div className="pipeline-stepper">
      <div className="stepper-top">
        <div>
          <div className="eyebrow">
            Real pipeline inspection
          </div>

          <h3>{current.title}</h3>

          <p>{current.detail}</p>
          <div className="conference-stage-meta">
            <span>LOCATION / <b>{location}</b></span>
            <span>DATA / <b>{dataState}</b></span>
            <span>STATE / <b>{completed ? "COMPLETE" : "READY"}</b></span>
            {stageTiming !== undefined && <span>LATENCY / <b>{Number(stageTiming).toFixed(2)} ms</b></span>}
          </div>
        </div>

        <div className="step-counter">
          {String(step + 1).padStart(2, "0")} /{" "}
          {String(stages.length).padStart(
            2,
            "0"
          )}
        </div>
      </div>

      <div className="stepper-body">
        {current.body}
      </div>

      <div className="stepper-controls">
        <button
          className="secondary-button"
          disabled={step === 0}
          onClick={() =>
            setStep(step - 1)
          }
        >
          BACK
        </button>

        <div className="step-dots">
          {stages.map(
            (stage, index) => (
              <button
                aria-label={`Show ${stage.title}`}
                className={
                  index === step
                    ? "step-dot active"
                    : "step-dot"
                }
                key={stage.title}
                onClick={() =>
                  setStep(index)
                }
              />
            )
          )}
        </div>

        <button
          className="primary-button"
          disabled={
            step === stages.length - 1
          }
          onClick={() =>
            setStep(step + 1)
          }
        >
          NEXT
          <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}

function FeatureMetric({
  label,
  value,
  timing,
}) {
  return (
    <div className="feature-metric">
      <span>{label}</span>

      <strong>{value}</strong>

      {timing !== undefined && (
        <small>
          {timing} ms forward pass
        </small>
      )}
    </div>
  );
}

function LocalPreview({
  file,
  className,
  alt,
}) {
  const [source, setSource] = useState(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);

    setSource(url);

    return () =>
      URL.revokeObjectURL(url);
  }, [file]);

  return (
    <img
      className={className}
      src={source || ""}
      alt={alt}
    />
  );
}

function DetectionPreview({
  file,
  detection,
}) {
  const [source, setSource] = useState(null);

  useEffect(() => {
    const url = URL.createObjectURL(file);

    setSource(url);

    return () =>
      URL.revokeObjectURL(url);
  }, [file]);

  return (
    <div className="detection-preview">
      <img
        src={source || ""}
        alt="Face detection input"
      />

      <svg
        viewBox={`0 0 ${
          detection.image_width || 1
        } ${
          detection.image_height || 1
        }`}
        preserveAspectRatio="none"
      >
        <rect
          x={detection.bbox.x1}
          y={detection.bbox.y1}
          width={
            detection.bbox.x2 -
            detection.bbox.x1
          }
          height={
            detection.bbox.y2 -
            detection.bbox.y1
          }
        />

        <g>
          {detection.landmarks.map(
            ([x, y], index) => (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="5"
              />
            )
          )}
        </g>
      </svg>
    </div>
  );
}

function ClientResult({ result }) {
  return (
    <div
      className={`client-result-card ${
        result.decision === "ACCEPT"
          ? "accepted"
          : "rejected"
      }`}
    >
      <div className="result-kicker">
        VERIFICATION COMPLETE
      </div>

      <h2>
        {result.decision === "ACCEPT"
          ? "Identity verified"
          : "Identity not verified"}
      </h2>

      <div className="client-result-grid">
        <span>Best match</span>
        <strong>{result.best_match}</strong>

        <span>Similarity</span>
        <strong>
          {Number.isFinite(Number(result?.similarity)) &&
          result?.similarity != null
            ? Number(result.similarity).toFixed(4)
            : "—"}
        </strong>

        <span>Threshold</span>
        <strong>{result.threshold}</strong>
      </div>

      <div className="decision">
        {result.decision}
      </div>

      {result.timing_ms && (
        <div className="timings">
          {Object.entries(
            result.timing_ms
          ).map(([label, value]) => (
            <span key={label}>
              {label.replaceAll(
                "_",
                " "
              )}{" "}
              <b>{value} ms</b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function MemberVerify({
  busy,
  file,
  setFile,
  submit,
}) {
  return (
    <div className="member-layout">
      <div className="member-intro">
        <ScanFace size={28} />

        <div>
          <div className="eyebrow">
            Private verification
          </div>

          <h2>Confirm it is you.</h2>

          <p>
            Upload a clear face image. Your image is
            processed for this request and is never
            exposed to the matching service.
          </p>
        </div>
      </div>

      <div className="panel operation-panel">
        <form onSubmit={submit}>
          <UploadBox
            file={file}
            setFile={setFile}
            label="Drop your face image here"
          />

          <button
            className="primary-button"
            disabled={busy}
          >
            {busy
              ? "CHECKING YOUR IDENTITY"
              : "VERIFY MY IDENTITY"}

            <ArrowRight size={17} />
          </button>
        </form>
      </div>

      <div className="member-reassurance">
        <LockKeyhole size={16} />

        <span>
          Your result is decided after encrypted
          matching. No face image or biometric data
          is displayed.
        </span>
      </div>
    </div>
  );
}

function MemberResult({
  result,
  onVerify,
  threshold,
}) {
  return (
    <div className="member-result-wrap">
      {result ? (
        <div
          className={`member-result ${
            result.decision === "ACCEPT"
              ? "accepted"
              : "rejected"
          }`}
        >
          <div className="result-symbol">
            {result.decision === "ACCEPT" ? (
              <Check size={30} />
            ) : (
              <X size={30} />
            )}
          </div>

          <div className="eyebrow">
            Verification complete
          </div>

          <h2>
            {result.decision === "ACCEPT"
              ? "Identity confirmed"
              : "Identity not confirmed"}
          </h2>

          <p>
            {result.decision === "ACCEPT"
              ? "The submitted image matched an enrolled identity."
              : "The submitted image did not meet the verification threshold."}
          </p>

          <div className="member-score">
            <span>Similarity</span>

            <strong>
              {result.similarity.toFixed(4)}
            </strong>

            <span>Threshold</span>

            <strong>{threshold}</strong>
          </div>

          <div className="decision">
            {result.decision}
          </div>
        </div>
      ) : (
        <div className="panel empty-member">
          <Check size={26} />

          <h2>No verification yet</h2>

          <p>
            Complete a verification to see your
            result here.
          </p>
        </div>
      )}

      <button
        className="secondary-button"
        onClick={onVerify}
      >
        VERIFY ANOTHER IMAGE
        <ArrowRight size={16} />
      </button>
    </div>
  );
}

function Verify({
  busy,
  file,
  setFile,
  result,
  submit,
  threshold,
}) {
  return (
    <div className="operation-grid">
      <div className="panel operation-panel">
        <PanelTitle
          label="Client operation"
          title="Submit a probe"
        />

        <form onSubmit={submit}>
          <UploadBox
            file={file}
            setFile={setFile}
            label="Drop a face image here"
          />

          <button
            className="primary-button"
            disabled={busy}
          >
            {busy
              ? "PROCESSING PIPELINE"
              : "START VERIFICATION"}

            <ArrowRight size={17} />
          </button>
        </form>

        <div className="privacy-note">
          <LockKeyhole size={15} />

          <span>
            Image is processed locally, encrypted
            before matching, and never stored by
            this interface.
          </span>
        </div>
      </div>

      <div className="panel result-panel">
        <PanelTitle
          label="Pipeline telemetry"
          title="Request path"
        />

        <Pipeline
          completed={Boolean(result)}
        />

        {result && (
          <div
            className={`result-card ${
              result.decision === "ACCEPT"
                ? "accepted"
                : "rejected"
            }`}
          >
            <div className="result-kicker">
              {result.decision === "ACCEPT"
                ? "MATCH FOUND"
                : "NO MATCH"}
            </div>

            <strong>
              {result.best_match}
            </strong>

            <div className="score-line">
              <span>Similarity</span>

              <b>
                {result.similarity.toFixed(4)}
              </b>

              <span>Threshold</span>

              <b>{threshold}</b>
            </div>

            <div className="decision">
              <span>
                {result.decision === "ACCEPT" ? (
                  <Check size={16} />
                ) : (
                  <X size={16} />
                )}
              </span>

              {result.decision}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function Enroll({
  busy,
  file,
  setFile,
  name,
  setName,
  submit,
}) {
  return (
    <div className="operation-grid single enrollment-view">
      <div className="panel operation-panel">
        <PanelTitle
          label="Client operation"
          title="Create encrypted enrollment"
        />

        <form onSubmit={submit}>
          <label className="field-label">
            Person name

            <input
              value={name}
              onChange={(event) =>
                setName(event.target.value)
              }
              placeholder="e.g. Jordan Lee"
              maxLength={80}
            />
          </label>

          <UploadBox
            file={file}
            setFile={setFile}
            label="Drop an enrollment image here"
          />

          <button
            className="primary-button"
            disabled={busy}
          >
            {busy
              ? "ENCRYPTING TEMPLATE"
              : "ENROLL AND PROTECT"}

            <LockKeyhole size={17} />
          </button>
        </form>

        <div className="privacy-note">
          <LockKeyhole size={15} />

          <span>
            The stored artifact is an encrypted
            template. No embedding or key material is
            returned.
          </span>
        </div>
      </div>
    </div>
  );
}

function DatabaseView({ templates, onDelete }) {
  return (
    <div className="panel table-panel">
      <PanelTitle
        label="Metadata only"
        title="Encrypted template registry"
      />

      <div className="registry-banner">
        <ShieldCheck size={18} />

        <span>
          All records below reference ciphertext
          files. Biometric vectors are never
          displayed.
        </span>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Identity</th>
              <th>Template file</th>
              <th>Status</th>
              <th>Created</th>
              <th>Action</th>
            </tr>
          </thead>

          <tbody>
            {templates.map(
              (template) => (
                <tr key={template.name}>
                  <td>
                    <strong>
                      {template.name}
                    </strong>
                  </td>

                  <td className="mono">
                    {template.template_file}
                  </td>

                  <td>
                    <span className="table-status">
                      <span />
                      {template.status}
                    </span>
                  </td>

                  <td className="muted-text">
                    {template.created_at}
                  </td>
                  <td>
                    <button
                      className="table-action-button"
                      type="button"
                      title={`Delete encrypted template for ${template.name}`}
                      aria-label={`Delete ${template.name}`}
                      onClick={() => onDelete(template.name)}
                    >
                      DELETE
                    </button>
                  </td>
                </tr>
              )
            )}
          </tbody>
        </table>

        {templates.length === 0 && (
          <div className="empty-state">
            No metadata records available.
          </div>
        )}
      </div>
    </div>
  );
}

function Security() {
  return (
    <div className="security-layout">
      <div className="panel security-hero">
        <div className="eyebrow">
          Architecture / trust model
        </div>

        <h2>
          Compute on ciphertext.
          <br />
          <span>Decide on the client.</span>
        </h2>

        <p>
          The face image and biometric representation
          remain in the client zone. The server can
          perform the comparison, but it cannot read
          the probe, stored templates, or similarity
          result.
        </p>
      </div>

      <div className="security-flow">
        <div className="flow-card amber">
          <span className="flow-number">
            01
          </span>

          <LockKeyhole size={20} />

          <h3>Client</h3>

          <p>
            Detection, alignment, embedding, fusion,
            normalization.
          </p>

          <strong>
            Plaintext permitted
          </strong>
        </div>

        <ArrowRight className="flow-arrow" />

        <div className="flow-card boundary">
          <span className="flow-number">
            02
          </span>

          <Zap size={20} />

          <h3>CKKS</h3>

          <p>
            Normalized representation becomes
            serialized ciphertext.
          </p>

          <strong>
            Secret key stays client-side
          </strong>
        </div>

        <ArrowRight className="flow-arrow" />

        <div className="flow-card teal">
          <span className="flow-number">
            03
          </span>

          <Server size={20} />

          <h3>Server</h3>

          <p>
            Homomorphic dot product over encrypted
            templates.
          </p>

          <strong>
            Ciphertext only
          </strong>
        </div>
      </div>
    </div>
  );
}

function Evaluation({ status }) {
  const [job, setJob] = useState(null);
  const [running, setRunning] =
    useState(false);
  const [error, setError] = useState("");

  async function run() {
    setRunning(true);
    setError("");

    try {
      const next =
        await startEvaluation();

      setJob(next);
    } catch (requestError) {
      setError(requestError.message);
      setRunning(false);
    }
  }

  useEffect(() => {
    if (
      !job?.job_id ||
      job.state !== "running"
    ) {
      return undefined;
    }

    const timer = setInterval(
      async () => {
        try {
          const next =
            await getEvaluation(
              job.job_id
            );

          setJob(next);

          if (
            next.state === "complete"
          ) {
            setRunning(false);
          }
        } catch (requestError) {
          setError(
            requestError.message
          );
          setRunning(false);
        }
      },
      700
    );

    return () =>
      clearInterval(timer);
  }, [job?.job_id, job?.state]);

  const progress = job
    ? Math.round(
        (job.processed / job.total) * 100
      )
    : 0;

  const metrics = job?.results;

  return (
    <div className="evaluation-console">
      <div className="evaluation-console-head">
        <div>
          <div className="eyebrow">
            Server / research evaluation
          </div>

          <h2>
            LFW DATASET EVALUATION
          </h2>

          <p>
            Real encrypted comparisons using the
            local LFW deep-funneled dataset.
          </p>
        </div>

        <button
          className="primary-button"
          disabled={running}
          onClick={run}
        >
          {running
            ? "EVALUATION RUNNING"
            : "RUN 500-IMAGE EVALUATION"}

          <Zap size={16} />
        </button>
      </div>

      <div className="dataset-facts">
        <div>
          <span>Dataset</span>
          <strong>LFW</strong>
        </div>

        <div>
          <span>Images</span>
          <strong>13,233</strong>
        </div>

        <div>
          <span>Identities</span>
          <strong>5,749</strong>
        </div>

        <div>
          <span>Test design</span>
          <strong>
            250 genuine + 250 impostor
          </strong>
        </div>
      </div>

      {error && (
        <div className="notice">
          <X size={16} />
          {error}
        </div>
      )}

      {job && (
        <div className="evaluation-run">
          <div className="run-line">
            <span>
              {job.state === "complete"
                ? "EVALUATION COMPLETE"
                : "PROCESSING REAL LFW IMAGES"}
            </span>

            <strong>
              {job.processed} / {job.total}
            </strong>
          </div>

          <div className="progress-track">
            <span
              style={{
                width: `${progress}%`,
              }}
            />
          </div>

          <div className="run-subline">
            <span>
              Current test:{" "}
              {job.sample
                ? `${job.sample.probe_name} → ${job.sample.target_name}`
                : "Preparing enrolled identities"}
            </span>

            <span>{progress}%</span>
          </div>
        </div>
      )}

      {metrics && (
        <>
          <div className="evaluation-metrics-grid">
            {[
              [
                "Total tests",
                metrics.total_tests,
              ],
              [
                "Genuine tests",
                metrics.genuine_tests,
              ],
              [
                "Impostor tests",
                metrics.impostor_tests,
              ],
              [
                "True accepts",
                metrics.true_accepts,
              ],
              [
                "True rejects",
                metrics.true_rejects,
              ],
              [
                "False accepts",
                metrics.false_accepts,
              ],
              [
                "False rejects",
                metrics.false_rejects,
              ],
              [
                "Average similarity",
                metrics.average_similarity,
              ],
            ].map(
              ([label, value]) => (
                <div
                  className="evaluation-metric"
                  key={label}
                >
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              )
            )}
          </div>

          <div className="rate-cards">
            <div>
              <span>Accuracy</span>
              <strong>
                {metrics.accuracy}%
              </strong>
            </div>

            <div>
              <span>FAR</span>
              <strong>
                {metrics.far}%
              </strong>
            </div>

            <div>
              <span>FRR</span>
              <strong>
                {metrics.frr}%
              </strong>
            </div>

            <div>
              <span>Threshold</span>
              <strong>
                {metrics.threshold}
              </strong>
            </div>
          </div>

          <SampleTest job={job} />
        </>
      )}
    </div>
  );
}

function SampleTest({ job }) {
  const sample = job.sample;

  if (!sample) return null;

  return (
    <div className="sample-test panel">
      <PanelTitle
        label="View sample test"
        title="Real comparison sample"
      />

      <div className="sample-images">
        <div>
          <img
            src={`${API_BASE}/api/evaluation/${job.job_id}/image/probe`}
            alt={`${sample.probe_name} probe`}
          />

          <strong>
            IMAGE A / {sample.probe_name}
          </strong>

          <span>
            Client pipeline → CKKS
          </span>
        </div>

        <ArrowRight size={20} />

        <div>
          <img
            src={`${API_BASE}/api/evaluation/${job.job_id}/image/template`}
            alt={`${sample.target_name} template`}
          />

          <strong>
            IMAGE B / {sample.target_name}
          </strong>

          <span>
            Encrypted template source
          </span>
        </div>
      </div>

      <div className="sample-result">
        <span>
          Encrypted probe + encrypted template
          → homomorphic matching → encrypted
          similarity → client decryption
        </span>

        <strong>
          {sample.similarity} /{" "}
          {sample.decision}
        </strong>
      </div>
    </div>
  );
}

export default App;