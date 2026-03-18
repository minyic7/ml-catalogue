import { createContext, useContext } from "react"

export interface QAToolsContextValue {
  onScreenshotClick: () => void
  onHighlightClick: () => void
  onAskClick: () => void
  onSettingsClick: () => void
  hasSelection: boolean
  screenshotActive: boolean
}

export const QAToolsContext = createContext<QAToolsContextValue | null>(null)

export function useQATools() {
  const ctx = useContext(QAToolsContext)
  if (!ctx) throw new Error("useQATools must be used within QAAssistant")
  return ctx
}
