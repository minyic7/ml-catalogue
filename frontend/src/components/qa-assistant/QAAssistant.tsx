import * as React from "react"
import { useLocation } from "react-router-dom"

import { MessageCircle } from "lucide-react"

import { ChatDialog, type InitialContext } from "./ChatDialog"
import { ScreenshotCapture } from "./ScreenshotCapture"
import { SettingsDialog } from "./SettingsDialog"
import { QAToolsContext } from "./QAToolsContext"

/**
 * QAAssistant — parent orchestrator that composes Toolbox, ChatDialog,
 * ScreenshotCapture, and highlight-to-ask logic.  Mounted once in the
 * app shell (RootLayout) so it is available on every page.
 *
 * QA tool button handlers and state are exposed via QAToolsContext
 * so the header bar can render the buttons inline.
 */
export function QAAssistant({ children }: { children?: React.ReactNode }) {
  const [chatOpen, setChatOpen] = React.useState(false)
  const [settingsOpen, setSettingsOpen] = React.useState(false)
  const [screenshotActive, setScreenshotActive] = React.useState(false)
  const [initialContext, setInitialContext] =
    React.useState<InitialContext | null>(null)
  const [hasSelection, setHasSelection] = React.useState(false)
  const [selectionPopup, setSelectionPopup] = React.useState<{
    x: number
    y: number
  } | null>(null)
  const location = useLocation()

  // ------- Track whether the user currently has text selected -------
  React.useEffect(() => {
    function checkSelection() {
      const sel = window.getSelection()
      const text = sel?.toString().trim() ?? ""
      setHasSelection(text.length > 0)

      if (text.length > 0 && sel && sel.rangeCount > 0) {
        const range = sel.getRangeAt(0)
        const rect = range.getBoundingClientRect()
        // Position popup above the selection, centered horizontally
        setSelectionPopup({
          x: rect.left + rect.width / 2,
          y: rect.top,
        })
      } else {
        setSelectionPopup(null)
      }
    }

    document.addEventListener("selectionchange", checkSelection)
    return () => document.removeEventListener("selectionchange", checkSelection)
  }, [])

  // ------- Escape key closes the chat dialog -------
  React.useEffect(() => {
    if (!chatOpen) return

    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setChatOpen(false)
      }
    }

    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [chatOpen])

  // ------- Build a human-readable page label from the URL path -------
  const pageLabel = React.useMemo(() => {
    // pathname like "/level-1/chapter-1/page-1" → "Level 1 > Chapter 1 > Page 1"
    const parts = location.pathname.split("/").filter(Boolean)
    if (parts.length === 0) return undefined
    return parts
      .map((s) =>
        s
          .replace(/-/g, " ")
          .replace(/\b\w/g, (c) => c.toUpperCase())
      )
      .join(" > ")
  }, [location.pathname])

  // ------- Extract page text content for richer context -------
  const pageContent = React.useMemo(() => {
    const main = document.querySelector("main")
    if (!main) return pageLabel
    const text = main.innerText?.trim()
    // Cap at a reasonable length to avoid blowing up token usage
    if (text && text.length > 0) {
      const maxChars = 4000
      return text.length > maxChars ? text.slice(0, maxChars) + "…" : text
    }
    return pageLabel
    // Re-derive when the route changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname, pageLabel])

  // ------- Highlight / Ask button handler -------
  const handleHighlightClick = React.useCallback(() => {
    const sel = window.getSelection()
    const selectedText = sel?.toString().trim() ?? ""

    if (selectedText) {
      setInitialContext({
        type: "text",
        data: selectedText,
        label: pageLabel,
      })
      // Clear the browser selection to keep things clean
      sel?.removeAllRanges()
    } else {
      // No selection — open with just the page context
      setInitialContext(
        pageLabel ? { type: "text", data: "", label: pageLabel } : null
      )
    }

    setChatOpen(true)
  }, [pageLabel])

  // ------- Ask button handler (no highlight/screenshot, just page context) -------
  const handleAskClick = React.useCallback(() => {
    setInitialContext(null)
    setChatOpen(true)
  }, [])

  // ------- Screenshot button handler -------
  const handleSettingsClick = React.useCallback(() => {
    setSettingsOpen(true)
  }, [])

  const handleScreenshotClick = React.useCallback(() => {
    setScreenshotActive(true)
  }, [])

  const handleScreenshotCapture = React.useCallback((dataUri: string) => {
    setScreenshotActive(false)
    setInitialContext({ type: "image", data: dataUri, label: "Screenshot" })
    setChatOpen(true)
  }, [])

  const handleScreenshotCancel = React.useCallback(() => {
    setScreenshotActive(false)
  }, [])

  // ------- Selection popup click handler -------
  const handleSelectionPopupClick = React.useCallback(() => {
    const sel = window.getSelection()
    const selectedText = sel?.toString().trim() ?? ""

    if (selectedText) {
      setInitialContext({
        type: "text",
        data: selectedText,
        label: pageLabel,
      })
      sel?.removeAllRanges()
    }

    setSelectionPopup(null)
    setChatOpen(true)
  }, [pageLabel])

  // ------- Close handler -------
  const handleClose = React.useCallback(() => {
    setChatOpen(false)
  }, [])

  const ctxValue = React.useMemo(
    () => ({
      onScreenshotClick: handleScreenshotClick,
      onHighlightClick: handleHighlightClick,
      onAskClick: handleAskClick,
      onSettingsClick: handleSettingsClick,
      hasSelection,
      screenshotActive,
    }),
    [handleScreenshotClick, handleHighlightClick, handleAskClick, handleSettingsClick, hasSelection, screenshotActive]
  )

  return (
    <QAToolsContext.Provider value={ctxValue}>
      {children}
      {/* Selection bubble popup */}
      {selectionPopup && !chatOpen && (
        <button
          type="button"
          className="fixed z-[60] flex -translate-x-1/2 -translate-y-full items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-sm font-medium text-foreground shadow-lg transition-colors hover:bg-accent dark:border-input dark:bg-input/30"
          style={{
            left: selectionPopup.x,
            top: selectionPopup.y - 8,
          }}
          onMouseDown={(e) => {
            // Prevent this click from clearing the selection
            e.preventDefault()
          }}
          onClick={handleSelectionPopupClick}
        >
          <MessageCircle className="size-3.5" />
          Ask about this
        </button>
      )}

      {screenshotActive && (
        <ScreenshotCapture
          onCapture={handleScreenshotCapture}
          onCancel={handleScreenshotCancel}
        />
      )}
      <ChatDialog
        isOpen={chatOpen}
        onClose={handleClose}
        initialContext={initialContext}
        pageContext={pageContent}
      />
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
    </QAToolsContext.Provider>
  )
}
