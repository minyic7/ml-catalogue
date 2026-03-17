import * as React from "react"
import html2canvas from "html2canvas"

export interface ScreenshotCaptureProps {
  /** Called with base64 data URI of the captured region */
  onCapture: (dataUri: string) => void
  /** Called when user cancels (Escape or click without drag) */
  onCancel: () => void
}

interface SelectionRect {
  startX: number
  startY: number
  currentX: number
  currentY: number
}

const MIN_DRAG_DISTANCE = 5

function ScreenshotCapture({ onCapture, onCancel }: ScreenshotCaptureProps) {
  const [selection, setSelection] = React.useState<SelectionRect | null>(null)
  const isDragging = React.useRef(false)
  const overlayRef = React.useRef<HTMLDivElement>(null)

  // Handle Escape key
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onCancel()
      }
    }
    document.addEventListener("keydown", handleKeyDown)
    return () => document.removeEventListener("keydown", handleKeyDown)
  }, [onCancel])

  const handleMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      // Only handle left click
      if (e.button !== 0) return
      isDragging.current = false
      setSelection({
        startX: e.clientX,
        startY: e.clientY,
        currentX: e.clientX,
        currentY: e.clientY,
      })
    },
    []
  )

  const handleMouseMove = React.useCallback(
    (e: React.MouseEvent) => {
      setSelection((prev) => {
        if (!prev) return null
        const dx = e.clientX - prev.startX
        const dy = e.clientY - prev.startY
        if (Math.abs(dx) > MIN_DRAG_DISTANCE || Math.abs(dy) > MIN_DRAG_DISTANCE) {
          isDragging.current = true
        }
        return { ...prev, currentX: e.clientX, currentY: e.clientY }
      })
    },
    []
  )

  const handleMouseUp = React.useCallback(async () => {
    if (!isDragging.current || !selection) {
      // Click without drag — cancel
      onCancel()
      return
    }

    const x = Math.min(selection.startX, selection.currentX)
    const y = Math.min(selection.startY, selection.currentY)
    const width = Math.abs(selection.currentX - selection.startX)
    const height = Math.abs(selection.currentY - selection.startY)

    if (width < MIN_DRAG_DISTANCE || height < MIN_DRAG_DISTANCE) {
      onCancel()
      return
    }

    // Hide overlay before capture so it doesn't appear in the screenshot
    if (overlayRef.current) {
      overlayRef.current.style.display = "none"
    }

    try {
      // Account for scroll position — html2canvas captures from the document
      const scrollX = window.scrollX
      const scrollY = window.scrollY

      const canvas = await html2canvas(document.body, {
        x: x + scrollX,
        y: y + scrollY,
        width,
        height,
        useCORS: true,
        scale: window.devicePixelRatio,
      })

      const dataUri = canvas.toDataURL("image/png")
      onCapture(dataUri)
    } catch {
      onCancel()
    }
  }, [selection, onCapture, onCancel])

  // Compute the visible selection rectangle
  const rect = selection
    ? {
        left: Math.min(selection.startX, selection.currentX),
        top: Math.min(selection.startY, selection.currentY),
        width: Math.abs(selection.currentX - selection.startX),
        height: Math.abs(selection.currentY - selection.startY),
      }
    : null

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-[9999]"
      style={{ cursor: "crosshair" }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Semi-transparent backdrop */}
      <div className="absolute inset-0 bg-black/30" />

      {/* Selection highlight — cut-out from backdrop */}
      {rect && isDragging.current && rect.width > 0 && rect.height > 0 && (
        <>
          {/* Clear region */}
          <div
            className="absolute border-2 border-primary bg-transparent"
            style={{
              left: rect.left,
              top: rect.top,
              width: rect.width,
              height: rect.height,
              boxShadow: "0 0 0 9999px rgba(0,0,0,0.3)",
              zIndex: 1,
            }}
          />
          {/* Dimension label */}
          <div
            className="absolute rounded bg-primary px-1.5 py-0.5 text-xs text-primary-foreground"
            style={{
              left: rect.left,
              top: rect.top + rect.height + 4,
              zIndex: 2,
            }}
          >
            {Math.round(rect.width)} &times; {Math.round(rect.height)}
          </div>
        </>
      )}
    </div>
  )
}

export { ScreenshotCapture }
