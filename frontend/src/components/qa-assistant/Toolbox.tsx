import * as React from "react"
import { Camera, MessageSquare, TextSelect, X } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"

const STORAGE_KEY = "qa-toolbox-position"
const SEEN_KEY = "qa-toolbox-seen"
const DEFAULT_POSITION = { x: -140, y: -80 } // offset from bottom-right

interface ToolboxProps {
  onScreenshotClick?: () => void
  onHighlightClick?: () => void
  onAskClick?: () => void
  /** When true, the highlight/ask button shows an active visual indicator */
  hasSelection?: boolean
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

function Toolbox({ onScreenshotClick, onHighlightClick, onAskClick, hasSelection, className }: ToolboxProps) {
  const [expanded, setExpanded] = React.useState(false)
  const [position, setPosition] = React.useState(loadPosition)
  const [isDragging, setIsDragging] = React.useState(false)
  const [showHint, setShowHint] = React.useState(false)
  const dragRef = React.useRef<{
    startX: number
    startY: number
    startLeft: number
    startTop: number
    moved: boolean
  } | null>(null)
  const toolboxRef = React.useRef<HTMLDivElement>(null)

  // On first visit, briefly auto-expand the toolbox so users discover the tools
  React.useEffect(() => {
    try {
      if (localStorage.getItem(SEEN_KEY)) return
      localStorage.setItem(SEEN_KEY, "1")
    } catch {
      return
    }

    // Brief delay so the page renders first, then auto-expand
    const showTimer = setTimeout(() => {
      setExpanded(true)
      setShowHint(true)
    }, 800)

    // Auto-collapse after a few seconds
    const hideTimer = setTimeout(() => {
      setExpanded(false)
      setShowHint(false)
    }, 4800)

    return () => {
      clearTimeout(showTimer)
      clearTimeout(hideTimer)
    }
  }, [])

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
      } else {
        // No drag occurred — treat as a click to toggle
        setExpanded((prev) => !prev)
      }
    },
    []
  )

  const handleToggle = React.useCallback(() => {
    if (dragRef.current?.moved) return // don't toggle after drag
    setShowHint(false)
    setExpanded((prev) => !prev)
  }, [])

  return (
    <TooltipProvider>
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
        {/* Collapsed state — labeled pill button */}
        <div
          className={cn(
            "transition-all duration-200 ease-in-out",
            expanded ? "pointer-events-none scale-0 opacity-0" : "scale-100 opacity-100"
          )}
        >
          <div className="relative">
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="default"
                  className={cn(
                    "h-10 gap-1.5 rounded-full px-3 shadow-lg",
                    hasSelection && "ring-2 ring-offset-2 ring-primary"
                  )}
                  onClick={handleToggle}
                  aria-label="Open QA assistant toolbox"
                >
                  <Camera className="size-4" />
                  <span className="text-xs font-medium">QA Tools</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent side="left">
                Click to open: Screenshot, Highlight, Ask
              </TooltipContent>
            </Tooltip>
            {hasSelection && (
              <span className="absolute -top-0.5 -right-0.5 flex size-3">
                <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-75" />
                <span className="relative inline-flex size-3 rounded-full bg-primary" />
              </span>
            )}
          </div>
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
          <div className="flex flex-col items-end gap-1.5">
            {showHint && (
              <div className="rounded-lg bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground shadow-md animate-in fade-in-0 slide-in-from-bottom-2">
                Capture screenshots or highlight text to ask about
              </div>
            )}
            <div className="flex items-center gap-1 rounded-xl border border-border bg-background p-1.5 shadow-lg dark:border-input dark:bg-input/30">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="gap-1.5 px-2"
                    onClick={onScreenshotClick}
                    aria-label="Capture screen region to ask about"
                  >
                    <Camera className="size-4" />
                    <span className="text-xs">Screenshot</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  Capture screen region to ask about
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className={cn(
                      "gap-1.5 px-2",
                      hasSelection &&
                        "ring-2 ring-primary bg-primary/10 text-primary"
                    )}
                    onClick={onHighlightClick}
                    aria-label="Highlight text and ask about it"
                  >
                    <TextSelect className="size-4" />
                    <span className="text-xs">Highlight</span>
                    {hasSelection && (
                      <span className="absolute -top-1 -right-1 flex size-2.5">
                        <span className="absolute inline-flex size-full animate-ping rounded-full bg-primary opacity-75" />
                        <span className="relative inline-flex size-2.5 rounded-full bg-primary" />
                      </span>
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  Highlight text and ask about it
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="gap-1.5 px-2"
                    onClick={onAskClick}
                    aria-label="Ask a question about this page"
                  >
                    <MessageSquare className="size-4" />
                    <span className="text-xs">Ask</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  Ask a question about this page
                </TooltipContent>
              </Tooltip>
              <div className="mx-0.5 h-5 w-px bg-border dark:bg-input" />
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={handleToggle}
                    aria-label="Close toolbox"
                  >
                    <X className="size-3.5" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="top">
                  Close toolbox
                </TooltipContent>
              </Tooltip>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

export { Toolbox }
export type { ToolboxProps }
