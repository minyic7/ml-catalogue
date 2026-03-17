import * as React from "react"
import { useLocation } from "react-router-dom"

import { Toolbox } from "./Toolbox"
import { ChatDialog, type InitialContext } from "./ChatDialog"

/**
 * QAAssistant — parent component that composes Toolbox + ChatDialog
 * and wires up text-selection–based "highlight & ask" behaviour.
 */
export function QAAssistant() {
  const [chatOpen, setChatOpen] = React.useState(false)
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

  // ------- Build a human-readable page label from the URL path -------
  const pageLabel = React.useMemo(() => {
    // pathname like "/level-1/chapter-1/page-1" → "level-1 > chapter-1 > page-1"
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

  // ------- Screenshot button handler (placeholder) -------
  const handleScreenshotClick = React.useCallback(() => {
    setInitialContext(null)
    setChatOpen(true)
  }, [])

  // ------- Close handler -------
  const handleClose = React.useCallback(() => {
    setChatOpen(false)
  }, [])

  return (
    <>
      {!chatOpen && (
        <Toolbox
          onScreenshotClick={handleScreenshotClick}
          onHighlightClick={handleHighlightClick}
          hasSelection={hasSelection}
        />
      )}
      <ChatDialog
        isOpen={chatOpen}
        onClose={handleClose}
        initialContext={initialContext}
        pageContext={pageLabel}
      />
    </>
  )
}
