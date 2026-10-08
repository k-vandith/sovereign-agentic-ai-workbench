"""Shared visual system for the workbench UI."""

CSS = """
<style>
:root {
  --bg: #0e1116;
  --surface: #161b22;
  --surface-2: #1c232d;
  --text: #e7ecf3;
  --muted: #9aa6b2;
  --border: #2a3340;
  --accent: #7aa2ff;
  --ok: #3dbe8b;
  --warn: #e0b15a;
  --danger: #e06c75;
  --radius: 12px;
}
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--bg);
  color: var(--text);
  font-family: "Segoe UI", ui-sans-serif, system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.4rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 600; }
.wb-top {
  display: flex; justify-content: space-between; align-items: flex-end;
  gap: 16px; margin-bottom: 14px;
}
.wb-kicker { color: var(--muted); font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; }
.wb-title { font-size: 1.65rem; margin: 2px 0 0; }
.wb-pills { display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; }
.pill {
  border: 1px solid var(--border); background: var(--surface);
  border-radius: 999px; padding: 4px 10px; font-size: 0.78rem; color: var(--muted);
}
.pill strong { color: var(--text); font-weight: 600; }
.pill.ok strong { color: var(--ok); }
.pill.warn strong { color: var(--warn); }
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px; margin-bottom: 12px;
}
.panel h3 { margin: 0 0 8px; font-size: 0.95rem; }
.empty { color: var(--muted); font-size: 0.92rem; line-height: 1.5; }
.src { border-top: 1px solid var(--border); padding-top: 8px; margin-top: 8px; }
.src .name { color: var(--text); font-weight: 600; }
.src .meta { color: var(--muted); font-size: 0.78rem; }
div[data-testid="stChatMessage"] {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
</style>
"""
