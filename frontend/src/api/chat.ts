export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  image?: string // base64 data URI for user messages
}

export interface ChatResponse {
  response: string
  session_id: string
}

export interface ContextUsageResponse {
  session_id: string
  estimated_tokens: number
  max_tokens: number
  usage_percent: number
  message_count: number
}

export interface CompactResponse {
  session_id: string
  original_messages: number
  compacted_messages: number
}

const SESSION_KEY = "qa-chat-session-id"

export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, id)
  }
  return id
}

export function resetSessionId(): string {
  const id = crypto.randomUUID()
  localStorage.setItem(SESSION_KEY, id)
  return id
}

export async function sendMessage(params: {
  sessionId: string
  message: string
  image?: string
  pageContext?: string
  signal?: AbortSignal
}): Promise<ChatResponse> {
  const { signal, ...rest } = params
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: rest.sessionId,
      message: rest.message,
      image: rest.image || undefined,
      page_context: rest.pageContext || undefined,
    }),
    signal,
  })

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error")
    throw new Error(`Chat failed (${response.status}): ${text}`)
  }

  return response.json()
}

export async function fetchContextUsage(
  sessionId: string
): Promise<ContextUsageResponse> {
  const response = await fetch(`/api/chat/context-usage/${sessionId}`)
  if (!response.ok) {
    throw new Error(`Context usage fetch failed (${response.status})`)
  }
  return response.json()
}

export async function compactHistory(
  sessionId: string
): Promise<CompactResponse> {
  const response = await fetch("/api/chat/compact", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  })

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error")
    throw new Error(`Compact failed (${response.status}): ${text}`)
  }

  return response.json()
}
