import { createContext, useContext } from "react"
import type { InitialContext } from "./ChatDialog"

export interface QAToolsContextValue {
  onScreenshotClick: () => void
  onHighlightClick: () => void
  onAskClick: () => void
  onSettingsClick: () => void
  hasSelection: boolean
  screenshotActive: boolean
  /** Chat dialog state exposed so RootLayout can render inline */
  chatOpen: boolean
  onChatClose: () => void
  initialContext: InitialContext | null
  pageContext?: string
}

export const QAToolsContext = createContext<QAToolsContextValue | null>(null)

export function useQATools() {
  const ctx = useContext(QAToolsContext)
  if (!ctx) throw new Error("useQATools must be used within QAAssistant")
  return ctx
}
