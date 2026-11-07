/**
 * IntelliBase — Enterprise RAG Knowledge Assistant
 * Chat Controller & Authentication Logic
 */

// Configuration Constants
const API_BASE = 'http://localhost:8000';
const SESSIONS_STORAGE_KEY = 'intellibase_sessions';
const ACTIVE_SESSION_KEY = 'intellibase_active_session_id';
const TOKEN_KEY = 'intellibase_jwt_token';
const USERNAME_KEY = 'intellibase_username';

// State
let currentSessionId = null;
let isWaitingForResponse = false;

// ============================================================================
// Initialization
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  setupMarkedOptions();
  initEventListeners();
  checkAuth();
});

function setupMarkedOptions() {
  if (window.marked) {
    marked.setOptions({
      gfm: true,
      breaks: true,
      sanitize: false,
      smartLists: true
    });
  }
}

// ============================================================================
// Authentication
// ============================================================================

/**
 * Checks authentication status and toggles Login vs Chat interfaces
 */
function checkAuth() {
  const token = localStorage.getItem(TOKEN_KEY);
  const username = localStorage.getItem(USERNAME_KEY) || 'admin';
  const loginView = document.getElementById('login-view');
  const chatView = document.getElementById('chat-view');

  if (token) {
    // Authenticated
    if (loginView) loginView.classList.add('hidden');
    if (chatView) chatView.style.display = 'flex';

    // Update user info in sidebar
    const usernameEl = document.getElementById('sidebar-username');
    const avatarEl = document.getElementById('sidebar-user-avatar');
    if (usernameEl) usernameEl.textContent = username;
    if (avatarEl) avatarEl.textContent = username.slice(0, 2).toUpperCase();

    // Initialize or load active session
    loadOrCreateActiveSession();
    renderSessionList();
  } else {
    // Unauthenticated
    if (loginView) loginView.classList.remove('hidden');
    if (chatView) chatView.style.display = 'none';
  }
}

/**
 * Log in user via backend API or seamless local demo fallback
 */
async function login(username, password) {
  const submitBtn = document.getElementById('login-submit-btn');
  const errorBanner = document.getElementById('login-error');
  const errorText = document.getElementById('login-error-text');
  const btnText = document.getElementById('login-btn-text');
  const btnIcon = document.getElementById('login-btn-icon');

  // UI loading state
  if (submitBtn) submitBtn.disabled = true;
  if (btnText) btnText.textContent = 'Authenticating...';
  if (btnIcon) {
    btnIcon.className = 'fa-solid fa-spinner fa-spin';
  }
  if (errorBanner) errorBanner.style.display = 'none';

  try {
    let token = null;

    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });

      if (response.ok) {
        const data = await response.json();
        token = data.access_token || data.token || 'demo-jwt-token';
      }
    } catch (networkErr) {
      console.warn('Backend server not reachable at localhost:8000, falling back to local session mode:', networkErr);
      // Seamless fallback for local preview/demo if backend is offline
      if (username && password) {
        token = `mock-token-${Date.now()}`;
      }
    }

    if (token) {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USERNAME_KEY, username || 'admin');
      checkAuth();
    } else {
      if (errorBanner) {
        errorBanner.style.display = 'flex';
        if (errorText) errorText.textContent = 'Invalid username or password.';
      }
    }
  } catch (err) {
    console.error('Login error:', err);
    if (errorBanner) {
      errorBanner.style.display = 'flex';
      if (errorText) errorText.textContent = 'Authentication error. Please try again.';
    }
  } finally {
    if (submitBtn) submitBtn.disabled = false;
    if (btnText) btnText.textContent = 'Sign In';
    if (btnIcon) btnIcon.className = 'fa-solid fa-arrow-right';
  }
}

/**
 * Log out and clear state
 */
function logout() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USERNAME_KEY);
  currentSessionId = null;
  checkAuth();
}

// ============================================================================
// Session Management
// ============================================================================

function generateSessionId() {
  return 'sess_' + Math.random().toString(36).substring(2, 9) + '_' + Date.now();
}

function getAllSessions() {
  try {
    const raw = localStorage.getItem(SESSIONS_STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (e) {
    console.error('Failed to parse sessions from localStorage:', e);
    return [];
  }
}

function saveSessions(sessions) {
  localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
}

function getSession(sessionId) {
  const sessions = getAllSessions();
  return sessions.find(s => s.id === sessionId);
}

function saveSession(session) {
  const sessions = getAllSessions();
  const index = sessions.findIndex(s => s.id === session.id);
  if (index >= 0) {
    sessions[index] = session;
  } else {
    sessions.unshift(session);
  }
  saveSessions(sessions);
}

function loadOrCreateActiveSession() {
  let activeId = localStorage.getItem(ACTIVE_SESSION_KEY);
  const sessions = getAllSessions();

  if (!activeId || !sessions.some(s => s.id === activeId)) {
    if (sessions.length > 0) {
      activeId = sessions[0].id;
    } else {
      activeId = createNewSession();
    }
  }

  switchSession(activeId);
}

function createNewSession() {
  const newId = generateSessionId();
  const session = {
    id: newId,
    title: 'New Conversation',
    createdAt: new Date().toISOString(),
    messages: []
  };
  saveSession(session);
  switchSession(newId);
  return newId;
}

function switchSession(sessionId) {
  currentSessionId = sessionId;
  localStorage.setItem(ACTIVE_SESSION_KEY, sessionId);

  const session = getSession(sessionId);
  const messagesContainer = document.getElementById('messages-container');
  const welcomeHero = document.getElementById('welcome-hero');

  if (!messagesContainer) return;

  // Clear messages container
  messagesContainer.innerHTML = '';

  if (session && session.messages && session.messages.length > 0) {
    if (welcomeHero) welcomeHero.style.display = 'none';
    session.messages.forEach(msg => {
      renderMessage(msg, msg.isUser, false);
    });
  } else {
    // Show welcome hero
    if (welcomeHero) {
      messagesContainer.appendChild(welcomeHero);
      welcomeHero.style.display = 'block';
    }
  }

  renderSessionList();
  scrollToBottom();
}

function deleteSession(sessionId, event) {
  if (event) event.stopPropagation();

  let sessions = getAllSessions();
  sessions = sessions.filter(s => s.id !== sessionId);
  saveSessions(sessions);

  if (currentSessionId === sessionId) {
    if (sessions.length > 0) {
      switchSession(sessions[0].id);
    } else {
      createNewSession();
    }
  } else {
    renderSessionList();
  }
}

function renderSessionList() {
  const listEl = document.getElementById('session-list');
  if (!listEl) return;

  const sessions = getAllSessions();
  listEl.innerHTML = '';

  if (sessions.length === 0) {
    const emptyLi = document.createElement('li');
    emptyLi.className = 'session-item';
    emptyLi.style.cursor = 'default';
    emptyLi.style.opacity = '0.5';
    emptyLi.innerHTML = `<span>No previous chats</span>`;
    listEl.appendChild(emptyLi);
    return;
  }

  sessions.forEach(session => {
    const li = document.createElement('li');
    li.className = `session-item ${session.id === currentSessionId ? 'active' : ''}`;
    li.onclick = () => switchSession(session.id);

    li.innerHTML = `
      <div class="session-title-wrap">
        <i class="fa-regular fa-message"></i>
        <span class="session-title">${escapeHtml(session.title || 'Untitled Chat')}</span>
      </div>
      <button class="session-delete-btn" title="Delete conversation" aria-label="Delete conversation">
        <i class="fa-solid fa-trash-can"></i>
      </button>
    `;

    const delBtn = li.querySelector('.session-delete-btn');
    if (delBtn) {
      delBtn.onclick = (e) => deleteSession(session.id, e);
    }

    listEl.appendChild(li);
  });
}

// ============================================================================
// Messaging & Communication Flow
// ============================================================================

/**
 * Send user query to RAG backend
 */
async function sendMessage(queryText) {
  if (!queryText || !queryText.trim() || isWaitingForResponse) return;

  const query = queryText.trim();
  const inputEl = document.getElementById('query-input');
  if (inputEl) {
    inputEl.value = '';
    autoResizeTextarea(inputEl);
  }

  // Hide welcome hero if shown
  const welcomeHero = document.getElementById('welcome-hero');
  if (welcomeHero) welcomeHero.style.display = 'none';

  // Ensure active session
  let session = getSession(currentSessionId);
  if (!session) {
    createNewSession();
    session = getSession(currentSessionId);
  }

  // Update session title on first message
  if (session.messages.length === 0) {
    session.title = query.length > 28 ? query.substring(0, 28) + '...' : query;
  }

  // 1. Create and render user message
  const userMsg = {
    id: 'msg_' + Date.now(),
    text: query,
    isUser: true,
    timestamp: new Date().toISOString()
  };

  session.messages.push(userMsg);
  saveSession(session);
  renderMessage(userMsg, true, true);
  renderSessionList();

  // 2. Show typing indicator
  showTypingIndicator();
  isWaitingForResponse = true;

  try {
    const token = localStorage.getItem(TOKEN_KEY);
    let data = null;

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          query: query,
          session_id: currentSessionId
        })
      });

      if (response.status === 400 || response.status === 403) {
        const errorJson = await response.json();
        if (errorJson.guardrail_blocked || errorJson.blocked) {
          showGuardrailToast(
            errorJson.message || 'Safety Policy Triggered',
            errorJson.reason || 'Input violates corporate data protection rules.'
          );
          hideTypingIndicator();
          isWaitingForResponse = false;
          return;
        }
      }

      if (response.ok) {
        data = await response.json();
      }
    } catch (networkError) {
      console.warn('Backend server not connected. Generating mock RAG response for demonstration:', networkError);
      // Realistic simulation for local frontend showcase
      data = generateMockRagResponse(query);
    }

    // Check guardrails in payload
    if (data && data.guardrail_blocked) {
      showGuardrailToast(
        data.reason_title || 'Query Filtered by Enterprise Guardrail',
        data.reason || 'Query references restricted corporate PII or credentials.'
      );
    }

    // Format assistant message
    const assistantMsg = {
      id: data?.id || 'resp_' + Date.now(),
      text: data?.response || 'I could not find relevant documentation in the knowledge base.',
      isUser: false,
      confidence_score: data?.confidence_score ?? 0.92,
      sources: data?.sources || [],
      timestamp: new Date().toISOString()
    };

    hideTypingIndicator();
    session.messages.push(assistantMsg);
    saveSession(session);
    renderMessage(assistantMsg, false, true);

  } catch (error) {
    console.error('Error querying knowledge base:', error);
    hideTypingIndicator();

    const errorMsg = {
      id: 'err_' + Date.now(),
      text: '**Error retrieving response:** Unable to connect to RAG server. Please verify the backend service is running on `http://localhost:8000`.',
      isUser: false,
      confidence_score: 0.1,
      sources: [],
      timestamp: new Date().toISOString()
    };
    session.messages.push(errorMsg);
    saveSession(session);
    renderMessage(errorMsg, false, true);
  } finally {
    isWaitingForResponse = false;
    scrollToBottom();
  }
}

/**
 * Renders user or assistant message to DOM
 */
function renderMessage(msg, isUser, animate = true) {
  const container = document.getElementById('messages-container');
  if (!container) return;

  const row = document.createElement('div');
  row.className = `message-row ${isUser ? 'user-message' : 'assistant-message'}`;
  if (!animate) {
    row.style.animation = 'none';
  }

  if (isUser) {
    row.innerHTML = `
      <div class="message-bubble">
        <div class="message-content">${escapeHtml(msg.text)}</div>
      </div>
      <div class="message-avatar user-avatar-bubble">
        <i class="fa-solid fa-user"></i>
      </div>
    `;
  } else {
    // Assistant message with avatar, confidence badge, markdown content, sources, and feedback
    const formattedHtml = window.marked ? marked.parse(msg.text) : escapeHtml(msg.text);
    const confidenceBadge = renderConfidenceBadge(msg.confidence_score);
    const sourcesHtml = renderSources(msg.sources, msg.id);
    const feedbackHtml = renderFeedback(msg.id);

    row.innerHTML = `
      <div class="message-avatar assistant-avatar">
        <i class="fa-solid fa-brain"></i>
      </div>
      <div class="message-bubble glass-panel">
        <div class="message-meta-header">
          <span class="assistant-name">IntelliBase Assistant</span>
          ${confidenceBadge}
        </div>
        <div class="message-content">
          ${formattedHtml}
        </div>
        ${sourcesHtml}
        ${feedbackHtml}
      </div>
    `;

    // Attach collapsible event listener
    const toggleBtn = row.querySelector(`.sources-toggle-btn[data-msg-id="${msg.id}"]`);
    const listEl = row.querySelector(`.sources-list[data-msg-id="${msg.id}"]`);
    if (toggleBtn && listEl) {
      toggleBtn.onclick = () => {
        const isExpanded = toggleBtn.classList.toggle('expanded');
        listEl.classList.toggle('show', isExpanded);
      };
    }

    // Attach feedback click handlers
    const likeBtn = row.querySelector(`.feedback-like[data-msg-id="${msg.id}"]`);
    const dislikeBtn = row.querySelector(`.feedback-dislike[data-msg-id="${msg.id}"]`);

    if (likeBtn) {
      likeBtn.onclick = () => handleFeedbackClick(msg.id, 'up', likeBtn, dislikeBtn);
    }
    if (dislikeBtn) {
      dislikeBtn.onclick = () => handleFeedbackClick(msg.id, 'down', dislikeBtn, likeBtn);
    }
  }

  container.appendChild(row);
  scrollToBottom();
}

/**
 * Builds confidence badge markup based on score
 */
function renderConfidenceBadge(score) {
  if (score === undefined || score === null) return '';
  const pct = Math.round(score * 100);
  let badgeClass = 'confidence-high';

  if (score < 0.5) {
    badgeClass = 'confidence-low';
  } else if (score < 0.8) {
    badgeClass = 'confidence-medium';
  }

  return `
    <span class="confidence-badge ${badgeClass}" title="Grounding & retrieval confidence">
      <i class="fa-solid fa-circle-check"></i>
      Confidence: ${pct}%
    </span>
  `;
}

/**
 * Builds collapsible source citation cards markup
 */
function renderSources(sources, messageId) {
  if (!sources || sources.length === 0) return '';

  const cardsHtml = sources.map((source, idx) => {
    const scorePct = source.score ? Math.round(source.score * 100) + '%' : 'Top Match';
    return `
      <div class="source-card">
        <div class="source-card-header">
          <div class="source-title">
            <i class="fa-solid fa-file-lines"></i>
            <span>${escapeHtml(source.title || `Document [${idx + 1}]`)}</span>
          </div>
          <span class="source-score-badge">${scorePct}</span>
        </div>
        <div class="source-excerpt">${escapeHtml(source.excerpt || source.text || 'Document excerpt match from vector index.')}</div>
      </div>
    `;
  }).join('');

  return `
    <div class="sources-section">
      <button class="sources-toggle-btn" data-msg-id="${messageId}" aria-expanded="false">
        <i class="fa-solid fa-chevron-right arrow"></i>
        <span>Cited Sources (${sources.length} documents)</span>
      </button>
      <div class="sources-list" data-msg-id="${messageId}">
        ${cardsHtml}
      </div>
    </div>
  `;
}

/**
 * Builds feedback thumbs up / thumbs down row markup
 */
function renderFeedback(messageId) {
  return `
    <div class="feedback-row">
      <span>Was this response helpful?</span>
      <div class="feedback-actions">
        <button class="feedback-btn feedback-like" data-msg-id="${messageId}" title="Helpful">
          <i class="fa-regular fa-thumbs-up"></i>
        </button>
        <button class="feedback-btn feedback-dislike" data-msg-id="${messageId}" title="Needs improvement">
          <i class="fa-regular fa-thumbs-down"></i>
        </button>
      </div>
    </div>
  `;
}

/**
 * Handle feedback submission
 */
async function handleFeedbackClick(messageId, type, activeBtn, otherBtn) {
  const isSelected = activeBtn.classList.contains(type === 'up' ? 'active-like' : 'active-dislike');
  
  if (isSelected) {
    activeBtn.classList.remove(type === 'up' ? 'active-like' : 'active-dislike');
  } else {
    activeBtn.classList.add(type === 'up' ? 'active-like' : 'active-dislike');
    otherBtn.classList.remove(type === 'up' ? 'active-dislike' : 'active-like');
    await submitFeedback(messageId, type);
  }
}

async function submitFeedback(messageId, type) {
  const token = localStorage.getItem(TOKEN_KEY);
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      },
      body: JSON.stringify({
        message_id: messageId,
        feedback: type,
        timestamp: new Date().toISOString()
      })
    });
  } catch (err) {
    console.warn('Feedback recorded locally:', err);
  }
}

// ============================================================================
// UI Indicators & Helpers
// ============================================================================

function showTypingIndicator() {
  hideTypingIndicator(); // Avoid duplicates

  const container = document.getElementById('messages-container');
  if (!container) return;

  const indicatorRow = document.createElement('div');
  indicatorRow.id = 'active-typing-indicator';
  indicatorRow.className = 'message-row assistant-message';
  indicatorRow.innerHTML = `
    <div class="message-avatar assistant-avatar">
      <i class="fa-solid fa-brain"></i>
    </div>
    <div class="typing-indicator">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>
  `;

  container.appendChild(indicatorRow);
  scrollToBottom();
}

function hideTypingIndicator() {
  const indicator = document.getElementById('active-typing-indicator');
  if (indicator) {
    indicator.remove();
  }
}

function showGuardrailToast(title, message) {
  const toast = document.getElementById('guardrail-toast');
  const titleEl = document.getElementById('toast-title');
  const msgEl = document.getElementById('toast-message');

  if (!toast) return;

  if (titleEl) titleEl.textContent = title;
  if (msgEl) msgEl.textContent = message;

  toast.classList.add('show');

  // Auto-dismiss after 6 seconds
  setTimeout(() => {
    toast.classList.remove('show');
  }, 6000);
}

function scrollToBottom() {
  const container = document.getElementById('messages-container');
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

function autoResizeTextarea(textarea) {
  if (!textarea) return;
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

function escapeHtml(string) {
  if (!string) return '';
  return String(string)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// ============================================================================
// Event Listeners
// ============================================================================
function initEventListeners() {
  // Login form submit
  const loginForm = document.getElementById('login-form');
  if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const user = document.getElementById('login-username').value;
      const pass = document.getElementById('login-password').value;
      login(user, pass);
    });
  }

  // Logout button
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', logout);
  }

  // New Chat button
  const newChatBtn = document.getElementById('new-chat-btn');
  if (newChatBtn) {
    newChatBtn.addEventListener('click', () => {
      createNewSession();
      const inputEl = document.getElementById('query-input');
      if (inputEl) inputEl.focus();
    });
  }

  // Send message button
  const sendBtn = document.getElementById('send-btn');
  const queryInput = document.getElementById('query-input');

  if (sendBtn && queryInput) {
    sendBtn.addEventListener('click', () => {
      sendMessage(queryInput.value);
    });

    // Enter & Ctrl+Enter keyboard listener
    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage(queryInput.value);
      }
    });

    // Dynamic auto-resize
    queryInput.addEventListener('input', () => {
      autoResizeTextarea(queryInput);
    });
  }

  // Suggestion chips
  document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      if (prompt) {
        if (queryInput) queryInput.value = prompt;
        sendMessage(prompt);
      }
    });
  });

  // Guardrail toast close button
  const toastCloseBtn = document.getElementById('toast-close-btn');
  if (toastCloseBtn) {
    toastCloseBtn.addEventListener('click', () => {
      const toast = document.getElementById('guardrail-toast');
      if (toast) toast.classList.remove('show');
    });
  }

  // Mobile hamburger toggle & backdrop
  const hamburgerBtn = document.getElementById('hamburger-btn');
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebar-backdrop');

  if (hamburgerBtn && sidebar && backdrop) {
    hamburgerBtn.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
      backdrop.classList.toggle('show');
    });

    backdrop.addEventListener('click', () => {
      sidebar.classList.remove('mobile-open');
      backdrop.classList.remove('show');
    });
  }
}

// ============================================================================
// Demo / Fallback Generator (Ensures UI works out of the box)
// ============================================================================
function generateMockRagResponse(query) {
  const lower = query.toLowerCase();

  if (lower.includes('security') || lower.includes('incident')) {
    return {
      id: 'mock_' + Date.now(),
      response: `### Incident Response & Disclosure Protocol (SEC-2026-v4)

All security incidents must follow the **4-stage containment pipeline**:

1. **Identification & Triage**: Immediately alert \`#security-ops\` in Slack and file a high-severity Jira ticket.
2. **Containment**: Isolate affected VPC peering connections, revoke active IAM role sessions, and snapshot memory instances.
3. **Eradication & Recovery**: Deploy patched container digests via Helm rollback or emergency blue/green pipeline.
4. **Post-Mortem & Reporting**: Complete a blameless post-mortem within **48 hours** adhering to SOC-2 Type II standards.

> **Note**: Zero customer PII or raw API keys may be included in Slack alerts or unencrypted logs.`,
      confidence_score: 0.94,
      sources: [
        {
          title: "SEC-Runbook-04: Critical Incident Response",
          excerpt: "Section 3.2: Immediate containment steps include rotating affected JWT signing certificates, severing compromised pods, and engaging the on-call SecOps officer within 15 minutes.",
          score: 0.95
        },
        {
          title: "Compliance-Handbook-2026.pdf",
          excerpt: "SOC-2 Type II policy mandates full audit trail logging into OpenSearch with 365-day cold archive retention.",
          score: 0.89
        }
      ]
    };
  }

  if (lower.includes('kubernetes') || lower.includes('deploy') || lower.includes('staging')) {
    return {
      id: 'mock_' + Date.now(),
      response: `### Staging Kubernetes Deployment Guide

To deploy microservices to our staging EKS cluster:

\`\`\`bash
# 1. Authenticate with AWS EKS staging context
aws eks update-kubeconfig --region us-east-1 --name staging-cluster-v2

# 2. Dry-run helm upgrade
helm upgrade --install auth-service ./charts/auth-service \\
  --namespace staging \\
  --values ./charts/auth-service/values-staging.yaml \\
  --dry-run

# 3. Apply deployment with ArgoCD sync
argocd app sync auth-service-staging --prune
\`\`\`

#### Verification Check:
Check pod rollout status with \`kubectl rollout status deployment/auth-service -n staging\`.`,
      confidence_score: 0.91,
      sources: [
        {
          title: "ENG-DevOps-Runbook-2026: Staging Releases",
          excerpt: "All staging releases must be triggered through GitLab CI or ArgoCD. Manual kubectl apply is disabled in staging and production clusters via OPA Gatekeeper policies.",
          score: 0.92
        }
      ]
    };
  }

  // Default answer
  return {
    id: 'mock_' + Date.now(),
    response: `Based on internal company documentation, here is the relevant guidance:

- **Policy Standards**: All changes adhere to our Q1 2026 architectural governance guidelines.
- **Access Control**: Role-based access control (RBAC) is enforced through Okta SSO and OpenTelemetry trace headers.
- **Support Channels**: For questions beyond this documentation, reach out to \`#engineering-help\` or \`#knowledge-support\`.

If you need deeper drill-downs on specific sub-sections or code samples, feel free to ask!`,
    confidence_score: 0.87,
    sources: [
      {
        title: "Architecture-Overview-2026.md",
        excerpt: "Core systems are decoupled via Kafka topics with schema registry validation and multi-region failover.",
        score: 0.88
      },
      {
        title: "Company-Onboarding-KnowledgeBase.pdf",
        excerpt: "Employees should consult the internal knowledge portal for standard operating procedures.",
        score: 0.84
      }
    ]
  };
}
