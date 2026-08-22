/*
 * frontend/app.js
 * Gemini-Style Multi-Turn Chat Application & Live Thinking Engine for MAS
 *
 * Separates internal research activity (phase events, tool activity, memory updates)
 * from the clean user-facing response. Implements a "Show internal activity" UI toggle switch.
 */

let eventSource = null;
let selectedMode = "standard";
let showInternalActivity = false;
let activeChatId = null;
let chatSessions = {};
let activeReportContext = "";
let currentPromptMode = null;
let uploadedFiles = [];

window.togglePromptMode = function(mode) {
  const queryInput = document.getElementById("query-input");
  const pills = {
    search: document.getElementById("pill-search"),
    think: document.getElementById("pill-think"),
    canvas: document.getElementById("pill-canvas")
  };
  
  if (currentPromptMode === mode) {
    currentPromptMode = null;
  } else {
    currentPromptMode = mode;
  }

  pills.search?.classList.remove("active-search");
  pills.think?.classList.remove("active-think");
  pills.canvas?.classList.remove("active-canvas");

  if (currentPromptMode === "search") {
    pills.search?.classList.add("active-search");
    if (queryInput) {
      queryInput.placeholder = "Search web & academic literature...";
      if (!queryInput.value.startsWith("/search")) {
        queryInput.value = "/search " + queryInput.value.replace(/^\/\w+\s*/, "");
      }
      queryInput.focus();
    }
  } else if (currentPromptMode === "think") {
    pills.think?.classList.add("active-think");
    selectedMode = "deep";
    document.querySelectorAll(".mode-btn").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === "deep");
    });
    if (queryInput) {
      queryInput.placeholder = "Deep reasoning mode (Flash/Pro smart routing)...";
      if (!queryInput.value.startsWith("/think")) {
        queryInput.value = "/think " + queryInput.value.replace(/^\/\w+\s*/, "");
      }
      queryInput.focus();
    }
  } else if (currentPromptMode === "canvas") {
    pills.canvas?.classList.add("active-canvas");
    if (queryInput) {
      queryInput.placeholder = "Create structured report on canvas...";
      if (!queryInput.value.startsWith("/canvas")) {
        queryInput.value = "/canvas " + queryInput.value.replace(/^\/\w+\s*/, "");
      }
      queryInput.focus();
    }
  } else {
    if (queryInput) {
      queryInput.placeholder = "Type your message here or type '/' for commands…";
      queryInput.value = queryInput.value.replace(/^\/\w+\s*/, "");
    }
  }
};

function initFileUpload() {
  const uploadBtn = document.getElementById("upload-btn");
  const fileInput = document.getElementById("file-upload-input");

  if (uploadBtn && fileInput) {
    uploadBtn.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadedFiles.push(e.target.files[0]);
        renderFilePreviews();
        fileInput.value = "";
      }
    });
  }
}

function renderFilePreviews() {
  const previewBar = document.getElementById("file-preview-bar");
  if (!previewBar) return;

  if (uploadedFiles.length === 0) {
    previewBar.style.display = "none";
    previewBar.innerHTML = "";
    return;
  }

  previewBar.style.display = "flex";
  previewBar.innerHTML = uploadedFiles.map((file, idx) => `
    <div class="file-preview-item">
      ${file.type.startsWith("image/") ? `<img src="${URL.createObjectURL(file)}" alt="Preview">` : `<div style="padding:8px;font-size:11px;color:#fff;">${file.name}</div>`}
      <button class="file-preview-remove" onclick="removeUploadedFile(${idx})">✕</button>
    </div>
  `).join("");
}

window.removeUploadedFile = function(idx) {
  uploadedFiles.splice(idx, 1);
  renderFilePreviews();
};

function insertCommand(cmd) {
  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.value = cmd;
    queryInput.focus();
  }
  const palette = document.getElementById("command-palette");
  if (palette) palette.style.display = "none";
}

function initAuralisBackground() {
  const canvas = document.getElementById("auralis-bg-canvas");
  const container = document.getElementById("auralis-container");
  if (!canvas || !container) return;

  const gl = canvas.getContext("webgl", { antialias: true });
  if (!gl) return;

  const vertexShaderGLSL = `
  attribute vec2 position;
  varying vec2 vUv;
  void main() {
    vUv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
  }
  `;

  const fragmentShaderGLSL = `
  precision highp float;
  varying vec2 vUv;

  uniform vec2  u_resolution;
  uniform float u_time;
  uniform float u_grain;
  uniform vec3  u_colors[3];

  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec3 permute(vec3 x) { return mod289(((x*34.0)+1.0)*x); }

  float snoise(vec2 v) {
    const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v - i + dot(i, C.xx);
    vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i);
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
    m = m*m; m = m*m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    vec2 uv = vUv;
    float ratio = u_resolution.x / u_resolution.y;
    vec2 p = uv * vec2(ratio, 1.0);
    float t = u_time * 0.2;

    float n1 = snoise(p * 0.5 + t);
    float n2 = snoise(p * 0.9 - t * 0.5 + n1);
    
    float light = pow(abs(n2), 2.5) * 0.5; 

    vec3 col = vec3(0.02, 0.01, 0.01); 

    col += u_colors[0] * smoothstep(0.1, 1.0, n1) * 0.5;
    col += u_colors[1] * light;

    float grain = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453 + u_time);
    col += (grain - 0.5) * u_grain * 0.5;

    float dist = length(uv - 0.5);
    col *= smoothstep(1.2, 0.2, dist);

    gl_FragColor = vec4(col, 1.0);
  }
  `;

  const createShader = (type, src) => {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    return s;
  };

  const program = gl.createProgram();
  gl.attachShader(program, createShader(gl.VERTEX_SHADER, vertexShaderGLSL));
  gl.attachShader(program, createShader(gl.FRAGMENT_SHADER, fragmentShaderGLSL));
  gl.linkProgram(program);
  gl.useProgram(program);

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

  const pos = gl.getAttribLocation(program, "position");
  gl.enableVertexAttribArray(pos);
  gl.vertexAttribPointer(pos, 2, gl.FLOAT, false, 0, 0);

  const locs = {
    res: gl.getUniformLocation(program, "u_resolution"),
    time: gl.getUniformLocation(program, "u_time"),
    grain: gl.getUniformLocation(program, "u_grain"),
    colors: gl.getUniformLocation(program, "u_colors"),
  };

  const hexToRgb = (hex) => {
    const h = hex.replace("#", "");
    return [
      parseInt(h.slice(0, 2), 16) / 255,
      parseInt(h.slice(2, 4), 16) / 255,
      parseInt(h.slice(4, 6), 16) / 255,
    ];
  };

  const colors = ["#ef4444", "#dc2626", "#b91c1c"];
  const speed = 0.3;
  const grain = 0.6;

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = container.clientWidth * dpr;
    canvas.height = container.clientHeight * dpr;
    gl.viewport(0, 0, canvas.width, canvas.height);
  };

  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  function render(t) {
    gl.uniform2f(locs.res, canvas.width, canvas.height);
    gl.uniform1f(locs.time, t * 0.001 * speed);
    gl.uniform1f(locs.grain, grain);

    const flat = new Float32Array(colors.slice(0, 3).flatMap(hexToRgb));
    gl.uniform3fv(locs.colors, flat);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

// Initialize Chat App
document.addEventListener("DOMContentLoaded", () => {
  initAuralisBackground();
  loadSavedChats();
  initFileUpload();

  // Clickable Noesis brand navigation
  const brandHome = document.getElementById("nav-brand-home");
  if (brandHome) {
    brandHome.addEventListener("click", () => {
      startNewChat();
    });
  }

  // Browser Back/Forward navigation listener
  window.addEventListener("popstate", (event) => {
    const state = event.state;
    if (state && state.view === "research") {
      if (state.sessionId && chatSessions[state.sessionId]) {
        loadChatSession(state.sessionId, false);
      } else {
        navigateToView("research", null, false);
      }
    } else {
      navigateToView("home", null, false);
    }
  });

  // Initial Route Check
  const currentPath = window.location.pathname;
  if (currentPath.startsWith("/research/")) {
    const sessionMatch = currentPath.split("/research/")[1];
    if (sessionMatch && chatSessions[sessionMatch]) {
      loadChatSession(sessionMatch, false);
    } else {
      navigateToView("home", null, false);
    }
  } else {
    navigateToView("home", null, false);
  }

  // Sidebar collapse toggle handler
  const sidebarToggleBtn = document.getElementById("sidebar-toggle-btn");
  const sidebar = document.getElementById("sidebar");
  if (sidebarToggleBtn && sidebar) {
    sidebarToggleBtn.addEventListener("click", () => {
      sidebar.classList.toggle("collapsed");
    });
  }

  // Mode Selection buttons
  document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      selectedMode = btn.dataset.mode;
    });
  });

  // Toggle "Show internal activity" switch
  const toggleBtn = document.getElementById("toggle-internal-activity");
  if (toggleBtn) {
    toggleBtn.addEventListener("change", (e) => {
      showInternalActivity = e.target.checked;
      updateActivityPanelVisibility();
    });
  }

  // Send button & enter key
  document.getElementById("send-btn").addEventListener("click", handleSend);
  document.getElementById("new-chat-btn").addEventListener("click", startNewChat);

  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    });

    queryInput.addEventListener("input", function() {
      this.style.height = "auto";
      this.style.height = Math.min(this.scrollHeight, 120) + "px";

      const cmdPalette = document.getElementById("command-palette");
      if (cmdPalette) {
        if (this.value.startsWith("/") && !this.value.includes(" ")) {
          cmdPalette.style.display = "block";
        } else {
          cmdPalette.style.display = "none";
        }
      }
    });
  }

  // Health check
  fetch("/api/health")
    .then(r => r.json())
    .then(data => {
      document.getElementById("api-status").innerText = `Online · ${data.default_model}`;
    })
    .catch(() => {
      document.getElementById("api-status").innerText = "Offline";
    });
});

// Update activity panel visibility instantly across all message turns
function updateActivityPanelVisibility() {
  document.querySelectorAll(".activity-panel").forEach(panel => {
    if (showInternalActivity) {
      panel.classList.add("expanded");
    } else {
      panel.classList.remove("expanded");
    }
  });
}

// Render KaTeX Math
function renderKaTeXMath(elementId) {
  const elem = document.getElementById(elementId);
  if (!elem || typeof renderMathInElement !== "function") return;

  renderMathInElement(elem, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
    throwOnError: false
  });
}

// Load Chat Sessions from localStorage
function loadSavedChats() {
  try {
    const raw = localStorage.getItem("MAS_CHAT_SESSIONS");
    if (raw) {
      chatSessions = JSON.parse(raw);
    }
  } catch (e) {
    chatSessions = {};
  }
  renderSidebarHistory();
}

let currentView = "home";

window.resetKGCounters = function() {
  const nodesElem = document.getElementById("kg-nodes");
  const edgesElem = document.getElementById("kg-edges");
  if (nodesElem) nodesElem.innerText = "0";
  if (edgesElem) edgesElem.innerText = "0";
};

window.scrollToBottom = function() {
  const scrollArea = document.getElementById("research-scroll-area");
  if (scrollArea) {
    requestAnimationFrame(() => {
      scrollArea.scrollTop = scrollArea.scrollHeight;
    });
    setTimeout(() => {
      scrollArea.scrollTop = scrollArea.scrollHeight;
    }, 100);
  }
};

window.navigateToView = function(viewName, sessionId = null, pushState = true) {
  currentView = viewName;
  const homeView = document.getElementById("home-view");
  const researchView = document.getElementById("research-view");

  if (viewName === "research") {
    document.body.classList.add("research-active");
    if (homeView) homeView.style.display = "none";
    if (researchView) researchView.style.display = "flex";

    if (sessionId) {
      activeChatId = sessionId;
      renderSidebarHistory();
    }

    if (pushState) {
      const path = sessionId ? `/research/${sessionId}` : "/research/session";
      try {
        history.pushState({ view: "research", sessionId: activeChatId }, "", path);
      } catch (e) {}
    }
  } else {
    document.body.classList.remove("research-active");
    if (researchView) researchView.style.display = "none";
    if (homeView) homeView.style.display = "flex";

    resetKGCounters();

    if (pushState) {
      try {
        history.pushState({ view: "home" }, "", "/");
      } catch (e) {}
    }
  }
};

function saveChats() {
  try {
    localStorage.setItem("MAS_CHAT_SESSIONS", JSON.stringify(chatSessions));
  } catch (e) {}
  renderSidebarHistory();
}

function renderSidebarHistory() {
  const list = document.getElementById("chat-history-list");
  if (!list) return;
  list.innerHTML = "";

  const sortedIds = Object.keys(chatSessions).sort((a, b) => chatSessions[b].updatedAt - chatSessions[a].updatedAt);
  if (sortedIds.length === 0) return;

  const now = Date.now();
  const oneDayMs = 24 * 60 * 60 * 1000;

  const todaySessions = [];
  const yesterdaySessions = [];
  const previousSessions = [];

  sortedIds.forEach(id => {
    const session = chatSessions[id];
    const diff = now - (session.updatedAt || now);
    if (diff < oneDayMs) {
      todaySessions.push({ id, session });
    } else if (diff < 2 * oneDayMs) {
      yesterdaySessions.push({ id, session });
    } else {
      previousSessions.push({ id, session });
    }
  });

  const renderGroup = (groupTitle, items) => {
    if (items.length === 0) return;
    const groupHeader = document.createElement("div");
    groupHeader.className = "history-group-header";
    groupHeader.style.cssText = "font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(255, 255, 255, 0.4); margin: 8px 0 4px 4px;";
    groupHeader.innerText = groupTitle;
    list.appendChild(groupHeader);

    items.forEach(({ id, session }) => {
      const item = document.createElement("div");
      item.className = `chat-history-item ${id === activeChatId ? 'active' : ''}`;
      item.innerText = session.title || "Research run";
      item.title = session.title || "Research run";
      item.onclick = () => loadChatSession(id, true);
      list.appendChild(item);
    });
  };

  renderGroup("TODAY", todaySessions);
  renderGroup("YESTERDAY", yesterdaySessions);
  renderGroup("PREVIOUS", previousSessions);
}

function startNewChat() {
  if (eventSource) eventSource.close();
  activeChatId = null;
  activeReportContext = "";

  resetKGCounters();
  navigateToView("home", null, true);

  const container = document.getElementById("chat-messages");
  if (container) container.innerHTML = "";
  renderSidebarHistory();
}

function loadChatSession(id, pushState = true) {
  const session = chatSessions[id];
  if (!session) return;
  activeChatId = id;
  activeReportContext = session.reportContext || "";

  navigateToView("research", id, pushState);

  const container = document.getElementById("chat-messages");
  container.innerHTML = "";

  session.messages.forEach((msg, idx) => {
    if (msg.role === "user") {
      appendUserBubbleNoNav(msg.text);
    } else {
      appendAssistantTurnHtml(msg.text, msg.thinkingLogs, msg.judgeResult, `msg-${idx}`);
    }
  });

  const scrollArea = document.getElementById("research-scroll-area");
  if (scrollArea) {
    scrollArea.scrollTop = scrollArea.scrollHeight;
  }
}

function appendUserBubbleNoNav(text) {
  const container = document.getElementById("chat-messages");
  const turn = document.createElement("div");
  turn.className = "message-turn";
  
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.innerText = text;

  turn.appendChild(bubble);
  container.appendChild(turn);
}

// User Message Bubble
function appendUserBubble(text) {
  navigateToView("research", activeChatId, true);

  appendUserBubbleNoNav(text);

  const scrollArea = document.getElementById("research-scroll-area");
  if (scrollArea) {
    scrollArea.scrollTop = scrollArea.scrollHeight;
  }
}

// Assistant Turn with Dedicated Internal Activity Panel & Clean Main Response Area
function appendAssistantTurn(turnId) {
  const container = document.getElementById("chat-messages");
  const turn = document.createElement("div");
  turn.className = "message-turn assistant-turn";
  turn.id = turnId;

  const isExpandedClass = showInternalActivity ? "expanded" : "";

  turn.innerHTML = `
    <div class="activity-panel ${isExpandedClass}" id="panel-${turnId}">
      <div class="activity-header" onclick="toggleActivityPanel('${turnId}')">
        <span id="act-summary-${turnId}">Research activity</span>
        <span style="opacity: 0.5; font-size: 11px;">▼</span>
      </div>
      <div class="activity-log" id="act-log-${turnId}"></div>
    </div>
    <div class="assistant-content markdown-body" id="content-${turnId}"></div>
    <div class="assistant-actions-bar">
      <button class="copy-response-btn" onclick="copyResponseText('content-${turnId}', this)">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>Copy response</span>
      </button>
    </div>
    <div id="judge-${turnId}"></div>
  `;

  container.appendChild(turn);
  const scrollArea = document.getElementById("research-scroll-area");
  if (scrollArea) {
    scrollArea.scrollTop = scrollArea.scrollHeight;
  }
}

function toggleActivityPanel(turnId) {
  const panel = document.getElementById(`panel-${turnId}`);
  if (panel) {
    panel.classList.toggle("expanded");
  }
}

function appendAssistantTurnHtml(text, thinkingLogs, judgeResult, turnId) {
  const container = document.getElementById("chat-messages");
  const turn = document.createElement("div");
  turn.className = "message-turn assistant-turn";
  turn.id = turnId;

  let judgeHtml = "";
  if (judgeResult) {
    judgeHtml = `<div class="judge-score-tag">Score: ${judgeResult.overall_score || 8.5}/10 · ${judgeResult.verdict || 'APPROVED'}</div>`;
  }

  let thinkingLogsHtml = "";
  if (thinkingLogs && thinkingLogs.length) {
    thinkingLogsHtml = thinkingLogs.map(l => `<div class="activity-item"><span class="activity-badge phase">Log</span>${l}</div>`).join("");
  }

  const isExpandedClass = showInternalActivity ? "expanded" : "";

  turn.innerHTML = `
    ${thinkingLogsHtml ? `
    <div class="activity-panel ${isExpandedClass}">
      <div class="activity-header" onclick="this.parentElement.classList.toggle('expanded')">
        <span>Research activity (${thinkingLogs.length} events)</span>
        <span style="opacity: 0.5; font-size: 11px;">▼</span>
      </div>
      <div class="activity-log">${thinkingLogsHtml}</div>
    </div>` : ''}
    <div class="assistant-content markdown-body" id="content-${turnId}"></div>
    <div class="assistant-actions-bar">
      <button class="copy-response-btn" onclick="copyResponseText('content-${turnId}', this)">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>Copy response</span>
      </button>
    </div>
    ${judgeHtml}
  `;

  container.appendChild(turn);

  const contentElem = document.getElementById(`content-${turnId}`);
  const cleanText = window.normalizeLaTeX ? window.normalizeLaTeX(text) : text;
  if (typeof marked !== "undefined") {
    contentElem.innerHTML = marked.parse(cleanText);
  } else {
    contentElem.innerText = cleanText;
  }
  renderKaTeXMath(`content-${turnId}`);
  scrollToBottom();
}

// Handle User Input
function handleSend() {
  const queryInput = document.getElementById("query-input");
  const query = queryInput.value.trim();
  if (!query) return;

  queryInput.value = "";
  queryInput.style.height = "auto";

  appendUserBubble(query);

  if (activeReportContext) {
    sendFollowupQuestion(query);
  } else {
    startPipelineResearch(query);
  }
}

// Pipeline Research Run
function startPipelineResearch(query) {
  const chatId = "chat_" + Date.now();
  activeChatId = chatId;
  
  chatSessions[chatId] = {
    id: chatId,
    title: query.slice(0, 35) + "...",
    mode: selectedMode,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [{ role: "user", text: query }],
    reportContext: ""
  };

  saveChats();

  const turnId = "turn_" + Date.now();
  appendAssistantTurn(turnId);

  const actSummary = document.getElementById(`act-summary-${turnId}`);
  const actLog = document.getElementById(`act-log-${turnId}`);
  const contentElem = document.getElementById(`content-${turnId}`);

  let accumulatedMarkdown = "";
  let thinkingLogs = [];
  let startTime = Date.now();

  if (eventSource) eventSource.close();

  const selectedFormat = document.getElementById("format-select") ? document.getElementById("format-select").value : "explanation";
  const streamUrl = `/api/research/stream?question=${encodeURIComponent(query)}&depth=${encodeURIComponent(selectedMode)}&mode=${encodeURIComponent(selectedFormat)}`;
  eventSource = new EventSource(streamUrl);

  const addInternalLog = (badgeCategory, msg) => {
    thinkingLogs.push(msg);
    const item = document.createElement("div");
    item.className = "activity-item";
    item.innerHTML = `<span class="activity-badge ${badgeCategory}">${badgeCategory}</span><span>${msg}</span>`;
    actLog.appendChild(item);
    actLog.scrollTop = actLog.scrollHeight;
  };

  eventSource.addEventListener("status", (e) => {
    const data = JSON.parse(e.data);
    actSummary.innerText = data.message || "Initializing...";
    addInternalLog("phase", data.message);
  });

  eventSource.addEventListener("pipeline_start", (e) => {
    const data = JSON.parse(e.data);
    actSummary.innerText = `Researching (${data.total_phases} phases)...`;
    addInternalLog("phase", `Pipeline active across ${data.total_phases} phases: ${data.active_phases.join(" -> ")}`);
  });

  const updateKGHeader = (nodes, edges) => {
    const n = document.getElementById("kg-nodes");
    const e = document.getElementById("kg-edges");
    if (n && nodes !== undefined) n.innerText = nodes;
    if (e && edges !== undefined) e.innerText = edges;
  };

  eventSource.addEventListener("thinking", (e) => {
    const data = JSON.parse(e.data);
    if (data.phase_name && data.phase_num) {
      actSummary.innerText = `Researching... Phase ${data.phase_num} of ${data.total_phases || 8} (${data.phase_name})`;
    }
    addInternalLog("phase", data.message);

    if (data.message) {
      const match = data.message.match(/Knowledge Graph updated:\s*(\d+)\s*nodes,\s*(\d+)\s*relations/i);
      if (match) {
        updateKGHeader(match[1], match[2]);
      }
    }
  });

  eventSource.addEventListener("kg_update", (e) => {
    const data = JSON.parse(e.data);
    updateKGHeader(data.num_nodes, data.num_edges);
    addInternalLog("memory", `Knowledge Graph updated: ${data.num_nodes} nodes, ${data.num_edges} relations`);
  });

  eventSource.addEventListener("research_complete", (e) => {
    const data = JSON.parse(e.data);
    const elapsedSec = Math.round((Date.now() - startTime) / 1000);
    
    actSummary.innerText = `Research complete (${elapsedSec}s)`;

    if (data.report_markdown) {
      accumulatedMarkdown = data.report_markdown;
      if (typeof marked !== "undefined") {
        contentElem.innerHTML = marked.parse(accumulatedMarkdown);
      } else {
        contentElem.innerText = accumulatedMarkdown;
      }
      renderKaTeXMath(`content-${turnId}`);
    }

    if (data.judge_result) {
      const judgeElem = document.getElementById(`judge-${turnId}`);
      const j = data.judge_result;
      if (judgeElem) {
        judgeElem.innerHTML = `<div class="judge-score-tag">Score: ${j.overall_score || 8.5}/10 · ${j.verdict || 'APPROVED'}</div>`;
      }
    }

    activeReportContext = accumulatedMarkdown;
    chatSessions[chatId].reportContext = accumulatedMarkdown;
    chatSessions[chatId].messages.push({
      role: "assistant",
      text: accumulatedMarkdown,
      thinkingLogs: thinkingLogs,
      judgeResult: data.judge_result
    });
    saveChats();

    eventSource.close();
  });

  eventSource.addEventListener("pipeline_paused", (e) => {
    const data = JSON.parse(e.data);
    actSummary.innerText = "Paused (Rate limit)";
    addInternalLog("provider", `Paused: ${data.message}`);
    const judgeElem = document.getElementById(`judge-${turnId}`);
    if (judgeElem) {
      judgeElem.innerHTML = `<div class="judge-score-tag" style="background: rgba(239,68,68,0.15); color: #f87171;">Status: PAUSED_RATE_LIMIT (Run ID: ${data.run_id})</div>`;
    }
    eventSource.close();
  });

  eventSource.addEventListener("pipeline_error", (e) => {
    const data = JSON.parse(e.data);
    addInternalLog("provider", `Error: ${data.error}`);
  });

  eventSource.onerror = () => {
    actSummary.innerText = "Research session ended";
    eventSource.close();
  };
}

// Send Follow-up Question
function sendFollowupQuestion(question) {
  const turnId = "turn_followup_" + Date.now();
  appendAssistantTurn(turnId);

  const actSummary = document.getElementById(`act-summary-${turnId}`);
  const actLog = document.getElementById(`act-log-${turnId}`);
  const contentElem = document.getElementById(`content-${turnId}`);

  actSummary.innerText = "Answering follow-up question...";
  const item = document.createElement("div");
  item.className = "activity-item";
  item.innerHTML = `<span class="activity-badge memory">MEMORY</span><span>Retrieving context from Working Memory...</span>`;
  actLog.appendChild(item);

  const history = chatSessions[activeChatId]?.messages.map(m => ({
    role: m.role,
    content: m.text
  })) || [];

  fetch("/api/chat/followup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question: question,
      report_context: activeReportContext,
      history: history
    })
  })
  .then(r => r.json())
  .then(data => {
    actSummary.innerText = "Response complete";
    const answer = data.answer || "No response generated.";
    if (typeof marked !== "undefined") {
      contentElem.innerHTML = marked.parse(answer);
    } else {
      contentElem.innerText = answer;
    }
    renderKaTeXMath(`content-${turnId}`);

    if (activeChatId && chatSessions[activeChatId]) {
      chatSessions[activeChatId].messages.push({ role: "user", text: question });
      chatSessions[activeChatId].messages.push({ role: "assistant", text: answer });
      chatSessions[activeChatId].updatedAt = Date.now();
      saveChats();
    }
  })
  .catch(err => {
    actSummary.innerText = "Error";
    contentElem.innerText = `Error answering follow-up: ${err.message}`;
  });
}


// =====================================================================
// GraphRAG Studio Dual-Mode (2D Globe & 3D Spatial) Engine
// =====================================================================
let currentMode = "2d"; // Default to user's favorite 2D Globe initial theme!
let visNetworkInstance = null;
let visNodesDataSet = null;
let visEdgesDataSet = null;
let graph3DInstance = null;
let rawGraphData = { nodes: [], edges: [] };
let isPhysicsActive = true;
let hoverNode = null;
const highlightNodes = new Set();
const highlightLinks = new Set();

// 2D Vis.js exact initial theme color map
const TYPE_COLOR_MAP_2D = {
  CONCEPT: { background: "rgba(129, 140, 248, 0.2)", border: "#818cf8", font: "#c7d2fe" },
  EQUATION: { background: "rgba(56, 189, 248, 0.2)", border: "#38bdf8", font: "#bae6fd" },
  METHOD: { background: "rgba(167, 139, 250, 0.2)", border: "#a78bfa", font: "#ddd6fe" },
  VARIABLE: { background: "rgba(52, 211, 153, 0.2)", border: "#34d399", font: "#a7f3d0" },
  METRIC: { background: "rgba(251, 191, 36, 0.2)", border: "#fbbf24", font: "#fde68a" },
  DEFAULT: { background: "rgba(148, 163, 184, 0.2)", border: "#94a3b8", font: "#e2e8f0" }
};

const TYPE_COLOR_MAP_3D = {
  CONCEPT: "#818cf8",
  EQUATION: "#38bdf8",
  METHOD: "#a78bfa",
  VARIABLE: "#34d399",
  METRIC: "#fbbf24",
  DEFAULT: "#94a3b8"
};

function initGraphStudio() {
  const openBtn = document.getElementById("open-kg-btn");
  const closeBtn = document.getElementById("close-kg-btn");
  const modal = document.getElementById("kg-modal");
  const searchInput = document.getElementById("kg-search-input");
  const physicsBtn = document.getElementById("kg-physics-toggle");
  const fitBtn = document.getElementById("kg-fit-btn");
  const inspectorClose = document.getElementById("kg-inspector-close");
  const btn2D = document.getElementById("kg-mode-2d");
  const btn3D = document.getElementById("kg-mode-3d");

  if (openBtn) {
    openBtn.addEventListener("click", () => {
      modal.style.display = "flex";
      requestAnimationFrame(() => {
        setTimeout(() => {
          loadAndRenderGraph();
        }, 100);
      });
    });
  }

  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      modal.style.display = "none";
    });
  }

  if (inspectorClose) {
    inspectorClose.addEventListener("click", () => {
      document.getElementById("kg-inspector").style.display = "none";
    });
  }

  if (btn2D && btn3D) {
    btn2D.addEventListener("click", () => {
      if (currentMode === "2d") return;
      currentMode = "2d";
      btn2D.classList.add("active");
      btn3D.classList.remove("active");
      renderGraphForCurrentMode();
    });

    btn3D.addEventListener("click", () => {
      if (currentMode === "3d") return;
      currentMode = "3d";
      btn3D.classList.add("active");
      btn2D.classList.remove("active");
      renderGraphForCurrentMode();
    });
  }

  if (physicsBtn) {
    physicsBtn.addEventListener("click", () => {
      isPhysicsActive = !isPhysicsActive;
      if (currentMode === "2d" && visNetworkInstance) {
        visNetworkInstance.setOptions({ physics: { enabled: isPhysicsActive } });
      } else if (currentMode === "3d" && graph3DInstance) {
        if (isPhysicsActive) graph3DInstance.resumeAnimation();
        else graph3DInstance.pauseAnimation();
      }
      physicsBtn.innerText = isPhysicsActive ? "Pause physics" : "Resume physics";
    });
  }

  if (fitBtn) {
    fitBtn.addEventListener("click", () => {
      if (currentMode === "2d" && visNetworkInstance) {
        visNetworkInstance.fit({ animation: { duration: 500 } });
      } else if (currentMode === "3d" && graph3DInstance) {
        graph3DInstance.zoomToFit(800, 40);
      }
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (currentMode === "2d" && visNodesDataSet) {
        const updates = [];
        rawGraphData.nodes.forEach(n => {
          const matches = !q || n.label.toLowerCase().includes(q);
          updates.push({
            id: n.id,
            opacity: matches ? 1.0 : 0.15,
            font: { opacity: matches ? 1.0 : 0.2 }
          });
        });
        visNodesDataSet.update(updates);
      } else if (currentMode === "3d" && graph3DInstance && rawGraphData.nodes) {
        if (!q) {
          graph3DInstance.zoomToFit(800, 40);
          return;
        }
        const match = rawGraphData.nodes.find(n => n.id.toLowerCase().includes(q));
        if (match) focusNodeIn3D(match);
      }
    });
  }

  // Type filter checkboxes
  const filterChips = document.querySelectorAll(".kg-type-filters input");
  filterChips.forEach(chip => {
    chip.addEventListener("change", () => {
      chip.parentElement.classList.toggle("active", chip.checked);
      filterGraphNodes();
    });
  });
}

function filterGraphNodes() {
  const activeTypes = Array.from(document.querySelectorAll(".kg-type-filters input:checked")).map(c => c.value);

  if (currentMode === "2d" && visNodesDataSet) {
    const updates = [];
    rawGraphData.nodes.forEach(n => {
      updates.push({
        id: n.id,
        hidden: !activeTypes.includes(n.type)
      });
    });
    visNodesDataSet.update(updates);
  } else if (currentMode === "3d" && graph3DInstance) {
    const filteredNodes = rawGraphData.nodes.filter(n => activeTypes.includes(n.type));
    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredEdges = rawGraphData.edges.filter(e => filteredNodeIds.has(e.source.id || e.source) && filteredNodeIds.has(e.target.id || e.target));
    graph3DInstance.graphData({ nodes: filteredNodes, links: filteredEdges });
  }
}

function loadAndRenderGraph() {
  fetch("/api/graph")
    .then(r => r.json())
    .then(data => {
      rawGraphData = data;

      // Update counters
      const nodeCount = data.num_nodes !== undefined ? data.num_nodes : data.nodes.length;
      const edgeCount = data.num_edges !== undefined ? data.num_edges : data.edges.length;

      document.getElementById("kg-nodes").innerText = nodeCount;
      document.getElementById("kg-edges").innerText = edgeCount;
      document.getElementById("kg-modal-node-count").innerText = nodeCount;
      document.getElementById("kg-modal-edge-count").innerText = edgeCount;

      renderGraphForCurrentMode();
    })
    .catch(err => {
      console.error("Failed to load Knowledge Graph:", err);
    });
}

function renderGraphForCurrentMode() {
  if (!rawGraphData.nodes) return;

  if (currentMode === "2d") {
    render2DVisNetwork(rawGraphData.nodes, rawGraphData.edges);
  } else {
    render3DForceGraph(rawGraphData.nodes, rawGraphData.edges);
  }
}

// ---------------------------------------------------------------------
// 2D Globe Mode (Vis.js Network Engine - Exact Initial Theme Restored)
// ---------------------------------------------------------------------
function render2DVisNetwork(nodes, edges) {
  const container = document.getElementById("kg-network-canvas");
  if (!container || typeof vis === "undefined") return;

  container.innerHTML = "";
  if (graph3DInstance) {
    graph3DInstance = null;
  }

  const visNodes = nodes.map(n => {
    const style = TYPE_COLOR_MAP_2D[n.type] || TYPE_COLOR_MAP_2D.DEFAULT;
    return {
      id: n.id,
      label: n.label,
      type: n.type,
      description: n.description,
      facts: n.facts,
      shape: "box",
      margin: 10,
      color: {
        background: style.background,
        border: style.border,
        highlight: { background: "rgba(56, 189, 248, 0.4)", border: "#38bdf8" }
      },
      font: { color: style.font, size: 12, face: "Inter" },
      borderWidth: 1.5,
      shadow: { enabled: true, color: "rgba(0,0,0,0.5)", size: 8, x: 2, y: 4 }
    };
  });

  const visEdges = edges.map((e, idx) => ({
    id: `edge_${idx}`,
    from: e.source,
    to: e.target,
    label: e.label,
    arrows: { to: { enabled: true, scaleFactor: 0.6 } },
    color: { color: "rgba(148, 163, 184, 0.3)", highlight: "#38bdf8" },
    font: { color: "#64748b", size: 10, align: "horizontal" },
    smooth: { type: "continuous" }
  }));

  visNodesDataSet = new vis.DataSet(visNodes);
  visEdgesDataSet = new vis.DataSet(visEdges);

  const graphData = { nodes: visNodesDataSet, edges: visEdgesDataSet };

  const options = {
    physics: {
      enabled: isPhysicsActive,
      solver: "forceAtlas2Based",
      forceAtlas2Based: {
        gravitationalConstant: -30,
        centralGravity: 0.012,
        springLength: 90,
        springConstant: 0.08
      },
      stabilization: { iterations: 150 }
    },
    interaction: {
      hover: true,
      tooltipDelay: 200,
      zoomView: true,
      dragView: true
    }
  };

  if (visNetworkInstance) {
    visNetworkInstance.destroy();
  }

  visNetworkInstance = new vis.Network(container, graphData, options);

  setTimeout(() => {
    if (visNetworkInstance) {
      visNetworkInstance.fit({ animation: false });
      visNetworkInstance.redraw();
    }
  }, 120);

  // Click Node Inspector Handler
  visNetworkInstance.on("selectNode", (params) => {
    const nodeId = params.nodes[0];
    const nodeObj = visNodesDataSet.get(nodeId);
    if (!nodeObj) return;

    showNodeInspector(nodeObj.label, nodeObj.type, nodeObj.description, nodeId);
  });

  visNetworkInstance.on("deselectNode", () => {
    document.getElementById("kg-inspector").style.display = "none";
  });
}

// ---------------------------------------------------------------------
// 3D Spatial Mode (Three.js + 3d-force-graph Engine)
// ---------------------------------------------------------------------
function render3DForceGraph(nodes, edges) {
  const container = document.getElementById("kg-network-canvas");
  if (!container || typeof ForceGraph3D === "undefined") return;

  container.innerHTML = "";
  if (visNetworkInstance) {
    visNetworkInstance.destroy();
    visNetworkInstance = null;
  }

  const gData = {
    nodes: nodes.map(n => ({
      id: n.id,
      name: n.label,
      type: n.type,
      description: n.description,
      facts: n.facts,
      color: TYPE_COLOR_MAP_3D[n.type] || TYPE_COLOR_MAP_3D.DEFAULT,
      val: n.type === "EQUATION" ? 8 : (n.type === "CONCEPT" ? 7 : 5)
    })),
    links: edges.map(e => ({
      source: e.source,
      target: e.target,
      label: e.label
    }))
  };

  graph3DInstance = ForceGraph3D()(container)
    .graphData(gData)
    .backgroundColor("#0b0c0e")
    .nodeColor(node => node.color)
    .nodeRelSize(6)
    .nodeResolution(24)
    .nodeVal("val")
    .nodeOpacity(0.95)
    .nodeThreeObject(node => {
      const group = new THREE.Group();
      const radius = node.val || 6;
      
      const sphereGeo = new THREE.SphereGeometry(radius, 24, 24);
      const sphereMat = new THREE.MeshPhongMaterial({
        color: node.color,
        emissive: node.color,
        emissiveIntensity: 0.35,
        shininess: 90,
        transparent: true,
        opacity: 0.95
      });
      const sphereMesh = new THREE.Mesh(sphereGeo, sphereMat);
      group.add(sphereMesh);

      const auraGeo = new THREE.SphereGeometry(radius * 1.4, 16, 16);
      const auraMat = new THREE.MeshBasicMaterial({
        color: node.color,
        transparent: true,
        opacity: 0.2
      });
      const auraMesh = new THREE.Mesh(auraGeo, auraMat);
      group.add(auraMesh);

      if (typeof SpriteText !== "undefined") {
        const sprite = new SpriteText(node.name);
        sprite.color = "#f8fafc";
        sprite.backgroundColor = "rgba(15, 17, 21, 0.7)";
        sprite.padding = [2, 4];
        sprite.borderRadius = 3;
        sprite.textHeight = 3.2;
        sprite.fontFace = "Inter";
        sprite.position.set(0, radius + 5, 0);
        group.add(sprite);
      }

      return group;
    })
    .linkWidth(link => (highlightLinks.has(link) ? 2.5 : 1.2))
    .linkColor(link => (highlightLinks.has(link) ? "#38bdf8" : "rgba(129, 140, 248, 0.35)"))
    .linkDirectionalParticles(link => (highlightLinks.has(link) ? 5 : 3))
    .linkDirectionalParticleWidth(link => (highlightLinks.has(link) ? 3.5 : 2.0))
    .linkDirectionalParticleSpeed(0.003)
    .onNodeHover(node => {
      highlightNodes.clear();
      highlightLinks.clear();
      if (node) {
        highlightNodes.add(node);
        gData.links.forEach(link => {
          if (link.source.id === node.id || link.target.id === node.id) {
            highlightLinks.add(link);
            highlightNodes.add(link.source);
            highlightNodes.add(link.target);
          }
        });
      }
      hoverNode = node || null;
      container.style.cursor = node ? "pointer" : "default";
    })
    .onNodeClick(node => {
      focusNodeIn3D(node);
    });

  graph3DInstance.d3Force("charge").strength(-35).distanceMax(180);
  graph3DInstance.d3Force("link").distance(32);

  const controls = graph3DInstance.controls();
  if (controls) {
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.35;
    controls.zoomSpeed = 0.5;
    controls.autoRotate = false;
  }
}

function focusNodeIn3D(node) {
  if (!graph3DInstance || !node) return;

  const distance = 90;
  const distRatio = 1 + distance / Math.hypot(node.x || 1, node.y || 1, node.z || 1);

  graph3DInstance.cameraPosition(
    { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
    { x: node.x || 0, y: node.y || 0, z: node.z || 0 },
    1500
  );

  showNodeInspector(node.name || node.id, node.type, node.description, node.id);
}

function showNodeInspector(title, type, description, nodeId) {
  const inspector = document.getElementById("kg-inspector");
  document.getElementById("kg-inspector-title").innerText = title;
  document.getElementById("kg-inspector-type").innerText = type || "CONCEPT";
  document.getElementById("kg-inspector-type").className = `kg-inspector-type-badge chip-${(type || 'concept').toLowerCase()}`;
  document.getElementById("kg-inspector-desc").innerText = description || "No description provided.";

  const connectedEdges = rawGraphData.edges.filter(e => e.source === nodeId || e.target === nodeId);
  const linksUl = document.getElementById("kg-inspector-links");
  linksUl.innerHTML = "";

  if (connectedEdges.length === 0) {
    linksUl.innerHTML = `<li>No direct connections</li>`;
  } else {
    connectedEdges.forEach(e => {
      const isOutgoing = e.source === nodeId;
      const target = isOutgoing ? e.target : e.source;
      const arrowStr = isOutgoing ? "→" : "←";
      const li = document.createElement("li");
      li.innerHTML = `<b>${e.label || 'relates_to'}</b> ${arrowStr} <span style="color: var(--text);">${target}</span>`;
      linksUl.appendChild(li);
    });
  }

  inspector.style.display = "block";
}

// Copy Response Text Helper
window.copyResponseText = function(containerId, btnElem) {
  const elem = document.getElementById(containerId);
  if (!elem) return;

  const textToCopy = elem.innerText || elem.textContent;
  navigator.clipboard.writeText(textToCopy).then(() => {
    if (btnElem) {
      const originalHtml = btnElem.innerHTML;
      btnElem.innerHTML = `
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        <span style="color: #10b981;">Copied!</span>
      `;
      setTimeout(() => {
        btnElem.innerHTML = originalHtml;
      }, 2000);
    }
  }).catch(err => {
    console.error("Failed to copy text:", err);
  });
};

// LaTeX Normalization & Preprocessing Engine
window.normalizeLaTeX = function(text) {
  if (!text || typeof text !== "string") return text || "";

  // 1. Convert display brackets \[ ... \] -> $$ ... $$ and \( ... \) -> $ ... $
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$');
  text = text.replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');

  // 2. Unescape escaped underscores in math contexts (P\_{win} -> P_{win}, Amount\_{loss} -> Amount_{loss})
  text = text.replace(/\\_\{([^}]+)\}/g, '_{$1}');
  text = text.replace(/\\_([a-zA-Z0-9]+)/g, '_$1');

  // 3. Fix malformed asterisk subscripts produced by LLMs (Amount*{win} -> Amount_{win}, P*{loss} -> P_{loss})
  text = text.replace(/([a-zA-Z0-9\}])\s*\*\{([^}]+)\}/g, '$1_{$2}');
  text = text.replace(/([a-zA-Z0-9\}])\s*\*([a-zA-Z0-9]+)/g, '$1_$2');

  // 4. Ensure \text{...} macros have properly attached subscripts
  text = text.replace(/\\text\{([^}]+)\}\\_/g, '\\text{$1}_');
  text = text.replace(/\\text\{([^}]+)\}\*/g, '\\text{$1}_');

  // 5. Clean up duplicate underscores
  text = text.replace(/_{2,}/g, '_');

  return text;
};

// ChatGPT-style KaTeX LaTeX Math Renderer
window.renderKaTeXMath = function(containerId) {
  const elem = document.getElementById(containerId);
  if (!elem) return;

  // Pre-normalize any raw text or innerHTML if needed
  if (elem.childNodes.length === 1 && elem.childNodes[0].nodeType === 3) {
    elem.textContent = window.normalizeLaTeX(elem.textContent);
  }

  // Render KaTeX safely if library is available
  if (typeof renderMathInElement !== "undefined") {
    try {
      renderMathInElement(elem, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$", right: "$", display: false },
          { left: "\\[", right: "\\]", display: true },
          { left: "\\(", right: "\\)", display: false }
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      });
    } catch (err) {
      console.warn("KaTeX rendering warning:", err);
    }
  }
};

document.addEventListener("DOMContentLoaded", () => {
  initGraphStudio();
  resetKGCounters();
});
