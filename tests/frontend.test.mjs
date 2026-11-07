import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';
import { test } from 'node:test';

const root = new URL('../frontend/js/', import.meta.url);

class FakeElement {
  constructor() {
    this.style = {};
    this.textContent = '';
    this.innerHTML = '';
    this.value = '';
    this.disabled = false;
    this.children = [];
    this.listeners = {};
    this.classList = {
      add() {},
      remove() {},
      toggle() { return false; }
    };
  }

  appendChild(child) { this.children.push(child); return child; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  querySelector() { return null; }
  closest() { return null; }
}

async function loadController(fileName, initialStorage = {}) {
  const source = await readFile(new URL(fileName, root), 'utf8');
  const elements = new Map();
  const storage = new Map(Object.entries(initialStorage));
  const document = {
    addEventListener() {},
    getElementById(id) {
      if (!elements.has(id)) elements.set(id, new FakeElement());
      return elements.get(id);
    },
    createElement() { return new FakeElement(); }
  };
  const localStorage = {
    getItem(key) { return storage.has(key) ? storage.get(key) : null; },
    setItem(key, value) { storage.set(key, String(value)); },
    removeItem(key) { storage.delete(key); }
  };
  const window = { location: { href: '' }, open() {} };
  const context = {
    document,
    localStorage,
    window,
    console: { warn() {}, error() {}, log() {} },
    fetch: async () => ({ ok: false, status: 500, json: async () => ({}) }),
    setInterval: () => 1,
    clearInterval() {},
    setTimeout,
    Date,
    Math,
    JSON,
    Promise
  };

  runInNewContext(source, context, { filename: fileName });
  return {
    context,
    elements,
    storage,
    evaluate(expression) {
      return runInNewContext(expression, context);
    }
  };
}

test('chat controller escapes HTML in untrusted content', async () => {
  const app = await loadController('chat.js');

  assert.equal(
    app.evaluate(`escapeHtml('<script>&"\\'')`),
    '&lt;script&gt;&amp;&quot;&#039;'
  );
  assert.match(app.evaluate(`renderSources([{title: '<img>', excerpt: '<script>'}], 'm1')`), /&lt;img&gt;/);
  assert.doesNotMatch(app.evaluate(`renderSources([{title: '<img>', excerpt: '<script>'}], 'm1')`), /<script>/);
});

test('chat controller persists sessions and displays the stored username after login', async () => {
  const app = await loadController('chat.js');
  app.context.fetch = async () => ({
    ok: true,
    json: async () => ({ access_token: 'test-token' })
  });

  await app.evaluate(`login('alice', 'password123')`);

  assert.equal(app.storage.get('intellibase_jwt_token'), 'test-token');
  assert.equal(app.storage.get('intellibase_username'), 'alice');
  assert.equal(app.elements.get('sidebar-username').textContent, 'alice');
  const sessions = JSON.parse(app.storage.get('intellibase_sessions'));
  assert.equal(sessions.length, 1);
  assert.equal(sessions[0].title, 'New Conversation');
});

test('dashboard redirects unauthenticated users and fills the stats widgets', async () => {
  const app = await loadController('dashboard.js');

  app.evaluate('checkDashboardAuth()');
  assert.equal(app.context.window.location.href, 'index.html');

  app.storage.set('intellibase_jwt_token', 'token');
  app.storage.set('intellibase_username', 'alice');
  app.evaluate('checkDashboardAuth()');
  app.evaluate(`updateStatsDOM({
    active_sessions: 8,
    active_sessions_trend: 'steady',
    latency_p95: '300ms',
    latency_trend: 'improved',
    csat_percent: '95%',
    csat_trend: 'positive',
    error_rate: '0.1%',
    error_rate_trend: 'healthy'
  })`);

  assert.equal(app.elements.get('sidebar-username').textContent, 'alice');
  assert.equal(app.elements.get('stat-sessions').textContent, 8);
  assert.equal(app.elements.get('stat-latency').textContent, '300ms');
  assert.equal(app.elements.get('stat-csat').textContent, '95%');
  assert.equal(app.elements.get('stat-error-rate').textContent, '0.1%');
});

test('dashboard renders evaluation metrics and escapes trace queries', async () => {
  const app = await loadController('dashboard.js');

  app.evaluate(`renderEvaluationResults({
    layer1_data_quality: { chunk_quality: 98, overall_score: 0.96 },
    layer2_retrieval: { precision_at_5: 0.9 },
    layer3_generation: { faithfulness: 0.92 },
    layer4_reliability: { latency_p95: 300 },
    layer5_business: { csat: 95 }
  })`);
  app.evaluate(`renderTracesTable([{
    trace_id: 'trace-1',
    timestamp: '12:00',
    query: '<script>alert(1)</script>',
    latency: '10ms',
    status: '200 OK',
    guardrail: 'Passed'
  }])`);

  assert.equal(app.elements.get('val-chunk-quality').textContent, '98%');
  assert.equal(app.elements.get('val-p5').textContent, '0.9');
  assert.equal(app.elements.get('val-faithfulness').textContent, '0.92');
  assert.equal(app.elements.get('val-latency-p95').textContent, '300ms');
  assert.equal(app.elements.get('val-csat-biz').textContent, '95%');
  const tableHtml = app.elements.get('traces-table-body').innerHTML;
  assert.match(tableHtml, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.doesNotMatch(tableHtml, /<script>/);
});
