import * as React from "react"
import { useLocation } from "react-router-dom"

import { Toolbox } from "./Toolbox"
import { ChatDialog, type InitialContext } from "./ChatDialog"
import { ScreenshotCapture } from "./ScreenshotCapture"

/**
 * QAAssistant — parent orchestrator that composes Toolbox, ChatDialog,
 * ScreenshotCapture, and highlight-to-ask logic.  Mounted once in the
 * app shell (RootLayout) so it is available on every page.
 *
 * The Toolbox remains visible even while the chat panel is open so
 * users can capture additional screenshots or highlight more text
 * mid-conversation.
 */
export function QAAssistant() {
  const [chatOpen, setChatOpen] = React.useState(false)
  const [screenshotActive, setScreenshotActive] = React.useState(false)
  const [initialContext, setInitialContext] =
    React.useState<InitialContext | null>(null)
  const [hasSelection, setHasSelection] = React.useState(false)
  const location = useLocation()

  // ------- Track whether the user currently has text selected -------
  React.useEffect(() => {
    function checkSelection() {
      const sel = window.getSelection()
      const text = sel?.toString().trim() ?? ""
      setHasSelection(text.length > 0)
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

  // ------- Close handler -------
  const handleClose = React.useCallback(() => {
    setChatOpen(false)
  }, [])

  return (
    <>
      {/* Toolbox is always visible except during screenshot capture */}
      {!screenshotActive && (
        <Toolbox
          onScreenshotClick={handleScreenshotClick}
          onHighlightClick={handleHighlightClick}
          onAskClick={handleAskClick}
          hasSelection={hasSelection}
        />
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
    </>
  )
}
