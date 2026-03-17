export interface ExecutionResult {
  stdout: string
  charts: string[]
  error: string | null
  execution_time_ms: number
}

export async function executeCode(params: {
  code: string
  mode: "quick" | "full"
  device: "cpu" | "mps"
}): Promise<ExecutionResult> {
  const response = await fetch("/api/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  })

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error")
    throw new Error(`Execution failed (${response.status}): ${text}`)
  }

  return response.json()
}
