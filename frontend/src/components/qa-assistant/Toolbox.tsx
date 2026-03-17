import * as React from "react"
import { Camera, MessageCircle, TextSelect, X } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

const STORAGE_KEY = "qa-toolbox-position"
const DEFAULT_POSITION = { x: -80, y: -80 } // offset from bottom-right

interface ToolboxProps {
  onScreenshotClick?: () => void
  onHighlightClick?: () => void
  className?: string
}

function loadPosition(): { x: number; y: number } {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored) return JSON.parse(stored)
  } catch {
    // ignore
  }
  return DEFAULT_POSITION
}

function savePosition(pos: { x: number; y: number }) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(pos))
  } catch {
    // ignore
  }
}

/**
 * Clamp position so the toolbox stays within the viewport.
 * Positions are stored as offsets: negative = from right/bottom, positive = from left/top.
 */
function clampPosition(
  x: number,
  y: number,
  elWidth: number,
  elHeight: number
) {
  const maxX = window.innerWidth - elWidth
  const maxY = window.innerHeight - elHeight

  // Convert to left/top for clamping
  const left = x < 0 ? window.innerWidth + x : x
  const top = y < 0 ? window.innerHeight + y : y

  const clampedLeft = Math.max(0, Math.min(left, maxX))
  const clampedTop = Math.max(0, Math.min(top, maxY))

  return { left: clampedLeft, top: clampedTop }
}

function Toolbox({ onScreenshotClick, onHighlightClick, className }: ToolboxProps) {
  const [expanded, setExpanded] = React.useState(false)
  const [position, setPosition] = React.useState(loadPosition)
  const [isDragging, setIsDragging] = React.useState(false)
  const dragRef = React.useRef<{
    startX: number
    startY: number
    startLeft: number
    startTop: number
    moved: boolean
  } | null>(null)
  const toolboxRef = React.useRef<HTMLDivElement>(null)

  // Compute CSS left/top from stored position
  const style = React.useMemo(() => {
    const el = toolboxRef.current
    const w = el?.offsetWidth ?? 48
    const h = el?.offsetHeight ?? 48
    const { left, top } = clampPosition(position.x, position.y, w, h)
    return { left, top } as React.CSSProperties
  }, [position])

  // Re-clamp on window resize
  React.useEffect(() => {
    function handleResize() {
      setPosition((prev) => ({ ...prev })) // trigger recompute
    }
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  const handlePointerDown = React.useCallback(
    (e: React.PointerEvent) => {
      // Only drag on primary button
      if (e.button !== 0) return
      const el = toolboxRef.current
      if (!el) return

      el.setPointerCapture(e.pointerId)
      const rect = el.getBoundingClientRect()
      dragRef.current = {
        startX: e.clientX,
        startY: e.clientY,
        startLeft: rect.left,
        startTop: rect.top,
        moved: false,
      }
      setIsDragging(true)
    },
    []
  )

  const handlePointerMove = React.useCallback(
    (e: React.PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return

      const dx = e.clientX - drag.startX
      const dy = e.clientY - drag.startY

      if (!drag.moved && Math.abs(dx) < 4 && Math.abs(dy) < 4) return
      drag.moved = true

      const newLeft = drag.startLeft + dx
      const newTop = drag.startTop + dy

      setPosition({ x: newLeft, y: newTop })
    },
    []
  )

  const handlePointerUp = React.useCallback(
    (e: React.PointerEvent) => {
      const drag = dragRef.current
      if (!drag) return

      toolboxRef.current?.releasePointerCapture(e.pointerId)
      dragRef.current = null
      setIsDragging(false)

      if (drag.moved) {
        // Save as absolute left/top
        const el = toolboxRef.current
        const w = el?.offsetWidth ?? 48
        const h = el?.offsetHeight ?? 48
        const dx = e.clientX - drag.startX
        const dy = e.clientY - drag.startY
        const newLeft = drag.startLeft + dx
        const newTop = drag.startTop + dy
        const clamped = clampPosition(newLeft, newTop, w, h)
        setPosition({ x: clamped.left, y: clamped.top })
        savePosition({ x: clamped.left, y: clamped.top })
      }
    },
    []
  )

  const handleToggle = React.useCallback(() => {
    if (dragRef.current?.moved) return // don't toggle after drag
    setExpanded((prev) => !prev)
  }, [])

  return (
    <div
      ref={toolboxRef}
      className={cn(
        "fixed z-50 select-none",
        isDragging ? "cursor-grabbing" : "cursor-grab",
        className
      )}
      style={style}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
    >
      {/* Collapsed state — single circular button */}
      <div
        className={cn(
          "transition-all duration-200 ease-in-out",
          expanded ? "pointer-events-none scale-0 opacity-0" : "scale-100 opacity-100"
        )}
      >
        <Button
          variant="default"
          size="icon-lg"
          className="size-12 rounded-full shadow-lg"
          onClick={handleToggle}
          aria-label="Open QA assistant toolbox"
        >
          <MessageCircle className="size-5" />
        </Button>
      </div>

      {/* Expanded state — tool buttons panel */}
      <div
        className={cn(
          "absolute bottom-0 right-0 origin-bottom-right transition-all duration-200 ease-in-out",
          expanded
            ? "pointer-events-auto scale-100 opacity-100"
            : "pointer-events-none scale-75 opacity-0"
        )}
      >
        <div className="flex items-center gap-1.5 rounded-xl border border-border bg-background p-1.5 shadow-lg dark:border-input dark:bg-input/30">
          <Button
            variant="ghost"
            size="icon"
            onClick={onScreenshotClick}
            aria-label="Take screenshot"
          >
            <Camera className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={onHighlightClick}
            aria-label="Highlight and ask"
          >
            <TextSelect className="size-4" />
          </Button>
          <div className="mx-0.5 h-5 w-px bg-border dark:bg-input" />
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={handleToggle}
            aria-label="Close toolbox"
          >
            <X className="size-3.5" />
          </Button>
        </div>
      </div>
    </div>
  )
}

export { Toolbox }
export type { ToolboxProps }
