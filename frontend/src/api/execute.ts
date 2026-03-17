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
  signal?: AbortSignal
}): Promise<ExecutionResult> {
  const { signal, ...body } = params
  const response = await fetch(`${import.meta.env.BASE_URL}api/execute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    const text = await response.text().catch(() => "Unknown error")
    throw new Error(`Execution failed (${response.status}): ${text}`)
  }

  return response.json()
}
