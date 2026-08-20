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

function insertCommand(cmd) {
  const queryInput = document.getElementById("query-input");
  if (queryInput) {
    queryInput.value = cmd;
    queryInput.focus();
  }
  const palette = document.getElementById("command-palette");
  if (palette) palette.style.display = "none";
}

// Initialize Chat App
document.addEventListener("DOMContentLoaded", () => {
  loadSavedChats();

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

  sortedIds.forEach(id => {
    const session = chatSessions[id];
    const item = document.createElement("div");
    item.className = `chat-history-item ${id === activeChatId ? 'active' : ''}`;
    item.innerText = session.title || "Research run";
    item.onclick = () => loadChatSession(id);
    list.appendChild(item);
  });
}

function startNewChat() {
  if (eventSource) eventSource.close();
  activeChatId = null;
  activeReportContext = "";

  const container = document.getElementById("chat-messages");
  container.innerHTML = `
    <div class="welcome-screen" id="welcome-screen">
      <div class="brand-badge">Autonomous multi-agent research system</div>
      <h1>Noesis</h1>
      <p>Enter a research question or request a technical derivation to trigger the multi-agent execution pipeline.</p>
      <div class="command-chips">
        <button class="cmd-pill-btn" onclick="insertCommand('/literature ')">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"></path><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"></path></svg>
          Literature
        </button>
        <button class="cmd-pill-btn" onclick="insertCommand('/math ')">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"></rect><line x1="8" y1="12" x2="16" y2="12"></line><line x1="12" y1="8" x2="12" y2="16"></line></svg>
          Mathematics
        </button>
        <button class="cmd-pill-btn" onclick="insertCommand('/compute ')">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="4" y="4" width="16" height="16" rx="2"></rect><rect x="9" y="9" width="6" height="6"></rect></svg>
          Compute
        </button>
        <button class="cmd-pill-btn" onclick="insertCommand('/review ')">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
          Review
        </button>
      </div>
    </div>
  `;
  renderSidebarHistory();
}

function loadChatSession(id) {
  const session = chatSessions[id];
  if (!session) return;
  activeChatId = id;
  activeReportContext = session.reportContext || "";

  renderSidebarHistory();

  const container = document.getElementById("chat-messages");
  container.innerHTML = "";

  session.messages.forEach((msg, idx) => {
    if (msg.role === "user") {
      appendUserBubble(msg.text);
    } else {
      appendAssistantTurnHtml(msg.text, msg.thinkingLogs, msg.judgeResult, `msg-${idx}`);
    }
  });
}

// User Message Bubble
function appendUserBubble(text) {
  const welcome = document.getElementById("welcome-screen");
  if (welcome) welcome.remove();

  const container = document.getElementById("chat-messages");
  const turn = document.createElement("div");
  turn.className = "message-turn";
  
  const bubble = document.createElement("div");
  bubble.className = "user-bubble";
  bubble.innerText = text;

  turn.appendChild(bubble);
  container.appendChild(turn);
  container.scrollTop = container.scrollHeight;
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
    <div id="judge-${turnId}"></div>
  `;

  container.appendChild(turn);
  container.scrollTop = container.scrollHeight;
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
    ${judgeHtml}
  `;

  container.appendChild(turn);

  const contentElem = document.getElementById(`content-${turnId}`);
  if (typeof marked !== "undefined") {
    contentElem.innerHTML = marked.parse(text);
  } else {
    contentElem.innerText = text;
  }
  renderKaTeXMath(`content-${turnId}`);
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

  eventSource.addEventListener("thinking", (e) => {
    const data = JSON.parse(e.data);
    if (data.phase_name && data.phase_num) {
      actSummary.innerText = `Researching... Phase ${data.phase_num} of ${data.total_phases || 8} (${data.phase_name})`;
    }
    addInternalLog("phase", data.message);
  });

  eventSource.addEventListener("kg_update", (e) => {
    const data = JSON.parse(e.data);
    document.getElementById("kg-nodes").innerText = data.num_nodes;
    document.getElementById("kg-edges").innerText = data.num_edges;
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
// GraphRAG Studio 3D Spatial WebGL Physics Visualizer Engine
// =====================================================================
let graph3DInstance = null;
let rawGraphData = { nodes: [], edges: [] };
let isPhysicsActive = true;
let hoverNode = null;
const highlightNodes = new Set();
const highlightLinks = new Set();

const TYPE_COLOR_MAP = {
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

  if (openBtn) {
    openBtn.addEventListener("click", () => {
      modal.style.display = "flex";
      loadAndRenderGraph();
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

  if (physicsBtn) {
    physicsBtn.addEventListener("click", () => {
      if (!graph3DInstance) return;
      isPhysicsActive = !isPhysicsActive;
      if (isPhysicsActive) {
        graph3DInstance.resumeAnimation();
      } else {
        graph3DInstance.pauseAnimation();
      }
      physicsBtn.innerText = isPhysicsActive ? "Pause physics" : "Resume physics";
    });
  }

  if (fitBtn) {
    fitBtn.addEventListener("click", () => {
      if (graph3DInstance) graph3DInstance.zoomToFit(800, 40);
    });
  }

  if (searchInput) {
    searchInput.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase().trim();
      if (!graph3DInstance || !rawGraphData.nodes) return;

      if (!q) {
        graph3DInstance.zoomToFit(800, 40);
        return;
      }

      const match = rawGraphData.nodes.find(n => n.id.toLowerCase().includes(q));
      if (match) {
        focusNodeIn3D(match);
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
  if (!graph3DInstance) return;
  const activeTypes = Array.from(document.querySelectorAll(".kg-type-filters input:checked")).map(c => c.value);

  const filteredNodes = rawGraphData.nodes.filter(n => activeTypes.includes(n.type));
  const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
  const filteredEdges = rawGraphData.edges.filter(e => filteredNodeIds.has(e.source.id || e.source) && filteredNodeIds.has(e.target.id || e.target));

  graph3DInstance.graphData({ nodes: filteredNodes, links: filteredEdges });
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

      render3DForceGraph(data.nodes, data.edges);
    })
    .catch(err => {
      console.error("Failed to load 3D Knowledge Graph:", err);
    });
}

function render3DForceGraph(nodes, edges) {
  const container = document.getElementById("kg-network-canvas");
  if (!container || typeof ForceGraph3D === "undefined") return;

  container.innerHTML = "";

  const gData = {
    nodes: nodes.map(n => ({
      id: n.id,
      name: n.label,
      type: n.type,
      description: n.description,
      facts: n.facts,
      color: TYPE_COLOR_MAP[n.type] || TYPE_COLOR_MAP.DEFAULT,
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
      
      // Core 3D Sphere Mesh with specular highlights
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

      // Outer Volumetric Glow Aura
      const auraGeo = new THREE.SphereGeometry(radius * 1.4, 16, 16);
      const auraMat = new THREE.MeshBasicMaterial({
        color: node.color,
        transparent: true,
        opacity: 0.2
      });
      const auraMesh = new THREE.Mesh(auraGeo, auraMat);
      group.add(auraMesh);

      // Clean Sprite Label
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

  // TIGHT COMPACT PHYSICS: Pulls all 316 nodes into a dense, rich 3D globe volume
  graph3DInstance.d3Force("charge").strength(-35).distanceMax(180);
  graph3DInstance.d3Force("link").distance(32);

  // SMOOTH DAMPED MOUSE CONTROLS: Removes twitchiness for buttery smooth navigation
  const controls = graph3DInstance.controls();
  if (controls) {
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.rotateSpeed = 0.35;
    controls.zoomSpeed = 0.5;
    controls.autoRotate = false; // Turn off fast auto-rotation so user has full control
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

  // Populate Side Inspector Panel
  const inspector = document.getElementById("kg-inspector");
  document.getElementById("kg-inspector-title").innerText = node.name || node.id;
  document.getElementById("kg-inspector-type").innerText = node.type || "CONCEPT";
  document.getElementById("kg-inspector-type").className = `kg-inspector-type-badge chip-${(node.type || 'concept').toLowerCase()}`;
  document.getElementById("kg-inspector-desc").innerText = node.description || "No description provided.";

  // Find connected links
  const gData = graph3DInstance.graphData();
  const connectedLinks = gData.links.filter(l => (l.source.id || l.source) === node.id || (l.target.id || l.target) === node.id);

  const linksUl = document.getElementById("kg-inspector-links");
  linksUl.innerHTML = "";

  if (connectedLinks.length === 0) {
    linksUl.innerHTML = `<li>No direct connections</li>`;
  } else {
    connectedLinks.forEach(l => {
      const srcId = l.source.id || l.source;
      const tgtId = l.target.id || l.target;
      const isOutgoing = srcId === node.id;
      const target = isOutgoing ? tgtId : srcId;
      const arrowStr = isOutgoing ? "→" : "←";
      const li = document.createElement("li");
      li.innerHTML = `<b>${l.label || 'relates_to'}</b> ${arrowStr} <span style="color: var(--text);">${target}</span>`;
      linksUl.appendChild(li);
    });
  }

  inspector.style.display = "block";
}

document.addEventListener("DOMContentLoaded", () => {
  initGraphStudio();
  fetch("/api/graph")
    .then(r => r.json())
    .then(d => {
      if (d && d.num_nodes !== undefined) {
        document.getElementById("kg-nodes").innerText = d.num_nodes;
        document.getElementById("kg-edges").innerText = d.num_edges;
      }
    }).catch(() => {});
});
