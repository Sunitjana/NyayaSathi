const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface QueryResult {
  answer: string;
  sources: string[];
  confidence: string;
  retrieved_chunks: { text: string; source: string; score: number }[];
  related_questions: string[];
}

export interface AssistantResult {
  category: string;
  answer: string;
  sources: string[];
  confidence: string;
}

export interface StatsResult {
  collections: Record<string, number>;
  total_chunks: number;
  categories: string[];
}

export interface SessionMeta {
  id: string;
  title: string;
  mode: string;
  created_at: string;
  updated_at: string;
}

export interface SessionFull {
  id: string;
  title: string;
  mode: string;
  messages: Record<string, unknown>[];
}

export async function fetchStats(): Promise<StatsResult> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
    return res.ok;
  } catch {
    return false;
  }
}

export type StreamChunk =
  | { type: "category"; category: string }
  | { type: "sources"; sources: string[]; chunks: { text: string; source: string; score: number }[] }
  | { type: "chunk"; content: string }
  | { type: "done"; confidence?: string; related_questions?: string[] };

export interface HistoryTurn {
  role: string;
  content: string;
}

export async function streamQuery(
  query: string,
  mode: "search" | "assistant",
  category: string | undefined,
  history: HistoryTurn[],
  onChunk: (chunk: StreamChunk) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode, category, history }),
  });

  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`Backend error (${res.status}): ${text || "No response"}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          onChunk(data as StreamChunk);
        } catch {
          // skip malformed
        }
      }
    }
  }
}

export async function listSessions(): Promise<SessionMeta[]> {
  const res = await fetch(`${API_BASE}/sessions`);
  if (!res.ok) return [];
  return res.json();
}

export async function loadSession(id: string): Promise<SessionFull | null> {
  const res = await fetch(`${API_BASE}/sessions/${id}`);
  if (!res.ok) return null;
  return res.json();
}

export async function saveSession(
  id: string | null,
  title: string,
  mode: string,
  messages: Record<string, unknown>[]
): Promise<string> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, title, mode, messages }),
  });
  if (!res.ok) throw new Error("Save failed");
  const data = await res.json();
  return data.id;
}

export async function deleteSession(id: string): Promise<void> {
  await fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" });
}
