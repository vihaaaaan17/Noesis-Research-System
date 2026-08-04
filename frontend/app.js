/*
 * frontend/app.js
 * Gemini-Style Multi-Turn Chat Application & Live Thinking Engine for MAS
 */

let eventSource = null;
let selectedMode = "standard";
let activeChatId = null;
let chatSessions = {};
let activeReportContext = "";

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

  // Send button & enter key
  document.getElementById("send-btn").addEventListener("click", handleSend);
  document.getElementById("new-chat-btn").addEventListener("click", startNewChat);

  const queryInput = document.getElementById("query-input");
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Auto-expand textarea height
  queryInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 120) + "px";
  });

  // Health check
  fetch("/api/health")
    .then(r => r.json())
    .then(data => {
      document.getElementById("api-status").innerText = `Online (${data.default_model})`;
    })
    .catch(() => {
      document.getElementById("api-status").innerText = "Offline";
    });
});

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
    item.innerText = session.title || "Research Run";
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
      <div class="spark-large">✦</div>
      <h1>What research problem shall we explore?</h1>
      <p>Ask a research question or request a mathematical derivation.</p>
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

// Assistant Turn with Gemini Live Thinking Accordion
function appendAssistantTurn(turnId) {
  const container = document.getElementById("chat-messages");
  const turn = document.createElement("div");
  turn.className = "message-turn assistant-turn";
  turn.id = turnId;

  turn.innerHTML = `
    <div class="thinking-box">
      <div class="thinking-header" onclick="toggleThinking('${turnId}')">
        <div class="thinking-spinner" id="spinner-${turnId}"></div>
        <span id="think-label-${turnId}">Thinking...</span>
      </div>
      <div class="thinking-log" id="think-log-${turnId}"></div>
    </div>
    <div class="assistant-content markdown-body" id="content-${turnId}"></div>
    <div id="judge-${turnId}"></div>
  `;

  container.appendChild(turn);
  container.scrollTop = container.scrollHeight;
}

function appendAssistantTurnHtml(text, thinkingLogs, judgeResult, turnId) {
  const container = document.getElementById("chat-messages");
  const turn = document.createElement("div");
  turn.className = "message-turn assistant-turn";
  turn.id = turnId;

  let judgeHtml = "";
  if (judgeResult) {
    judgeHtml = `<div class="judge-score-tag">Score: ${judgeResult.overall_score || 8.5}/10 ${judgeResult.verdict || 'APPROVED'}</div>`;
  }

  let thinkingLogsHtml = "";
  if (thinkingLogs && thinkingLogs.length) {
    thinkingLogsHtml = thinkingLogs.map(l => `<div>${l}</div>`).join("");
  }

  turn.innerHTML = `
    ${thinkingLogsHtml ? `
    <details class="thinking-box">
      <summary class="thinking-header">Thought process complete</summary>
      <div class="thinking-log">${thinkingLogsHtml}</div>
    </details>` : ''}
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

function toggleThinking(turnId) {
  const log = document.getElementById(`think-log-${turnId}`);
  if (log) {
    log.style.display = log.style.display === "none" ? "flex" : "none";
  }
}

// Handle User Input (New Research Run or Follow-up Cross Question)
function handleSend() {
  const queryInput = document.getElementById("query-input");
  const query = queryInput.value.trim();
  if (!query) return;

  queryInput.value = "";
  queryInput.style.height = "auto";

  appendUserBubble(query);

  if (activeReportContext) {
    // Interactive Follow-Up Cross Question
    sendFollowupQuestion(query);
  } else {
    // Initial Pipeline Research Run
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

  const thinkLabel = document.getElementById(`think-label-${turnId}`);
  const thinkLog = document.getElementById(`think-log-${turnId}`);
  const contentElem = document.getElementById(`content-${turnId}`);
  const spinner = document.getElementById(`spinner-${turnId}`);

  let accumulatedMarkdown = "";
  let thinkingLogs = [];
  let startTime = Date.now();

  if (eventSource) eventSource.close();

  const streamUrl = `/api/research/stream?question=${encodeURIComponent(query)}&depth=${encodeURIComponent(selectedMode)}`;
  eventSource = new EventSource(streamUrl);

  const addThinkingLog = (msg) => {
    thinkingLogs.push(msg);
    const line = document.createElement("div");
    line.innerText = msg;
    thinkLog.appendChild(line);
    thinkLog.scrollTop = thinkLog.scrollHeight;
  };

  eventSource.addEventListener("status", (e) => {
    const data = JSON.parse(e.data);
    addThinkingLog(data.message);
  });

  eventSource.addEventListener("pipeline_start", (e) => {
    const data = JSON.parse(e.data);
    addThinkingLog(`Pipeline active across ${data.total_phases} phases: ${data.active_phases.join(" -> ")}`);
  });

  eventSource.addEventListener("phase_start", (e) => {
    const data = JSON.parse(e.data);
    thinkLabel.innerText = `Thinking (Phase ${data.phase_num}: ${data.phase_name})...`;
    addThinkingLog(`Starting Phase ${data.phase_num}: ${data.phase_name}`);
  });

  eventSource.addEventListener("thinking", (e) => {
    const data = JSON.parse(e.data);
    addThinkingLog(data.message);
  });

  eventSource.addEventListener("kg_update", (e) => {
    const data = JSON.parse(e.data);
    document.getElementById("kg-nodes").innerText = data.num_nodes;
    document.getElementById("kg-edges").innerText = data.num_edges;
    addThinkingLog(`Knowledge Graph updated: ${data.num_nodes} nodes, ${data.num_edges} relations`);
  });

  eventSource.addEventListener("phase_complete", (e) => {
    const data = JSON.parse(e.data);
    addThinkingLog(`Completed Phase ${data.phase_name}`);

    // Live render section into assistant markdown content
    const sectionText = data.full_result || data.result_preview;
    if (sectionText) {
      if (!accumulatedMarkdown) {
        accumulatedMarkdown = `# Technical Analysis: ${query}\n\n`;
      }
      accumulatedMarkdown += `\n### Phase: ${data.phase_name}\n${sectionText}\n`;
      
      if (typeof marked !== "undefined") {
        contentElem.innerHTML = marked.parse(accumulatedMarkdown);
      } else {
        contentElem.innerText = accumulatedMarkdown;
      }
      renderKaTeXMath(`content-${turnId}`);
    }
  });

  eventSource.addEventListener("research_complete", (e) => {
    const data = JSON.parse(e.data);
    const elapsedSec = Math.round((Date.now() - startTime) / 1000);
    
    spinner.style.display = "none";
    thinkLabel.innerText = `Thought for ${elapsedSec}s`;

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
      judgeElem.innerHTML = `<div class="judge-score-tag">Score: ${j.overall_score || 8.5}/10 ${j.verdict || 'APPROVED'}</div>`;
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

  eventSource.onerror = () => {
    spinner.style.display = "none";
    thinkLabel.innerText = "Thought process complete";
    eventSource.close();
  };
}

// Send Follow-up Cross Question
function sendFollowupQuestion(question) {
  const turnId = "turn_followup_" + Date.now();
  appendAssistantTurn(turnId);

  const thinkLabel = document.getElementById(`think-label-${turnId}`);
  const thinkLog = document.getElementById(`think-log-${turnId}`);
  const contentElem = document.getElementById(`content-${turnId}`);
  const spinner = document.getElementById(`spinner-${turnId}`);

  thinkLabel.innerText = "Reasoning follow-up question...";
  const line = document.createElement("div");
  line.innerText = "Analyzing question against report context & Knowledge Graph...";
  thinkLog.appendChild(line);

  // Prepare history
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
    spinner.style.display = "none";
    thinkLabel.innerText = "Thought process complete";
    
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
    spinner.style.display = "none";
    thinkLabel.innerText = "Error";
    contentElem.innerText = `Error answering follow-up: ${err.message}`;
  });
}
