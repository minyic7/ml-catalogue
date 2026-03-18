import * as React from "react"
import {
  ArrowUp,
  Bot,
  GripVertical,
  ImageIcon,
  Loader2,
  Shrink,
  User,
  X,
} from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { MarkdownRenderer } from "@/components/MarkdownRenderer"
import {
  type ChatMessage,
  sendMessage,
  fetchContextUsage,
  compactHistory,
  getSessionId,
  resetSessionId,
} from "@/api/chat"
import { getStoredCustomBaseUrl, getStoredCustomApiKey, getStoredCustomModel } from "./SettingsDialog"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface InitialContext {
  type: "text" | "image"
  /** For text: the highlighted snippet. For image: base64 data URI. */
  data: string
  /** Human-readable label, e.g. "Linear Algebra > Vectors" */
  label?: string
}

export interface ChatDialogProps {
  isOpen: boolean
  onClose: () => void
  /** Context provided via highlight or screenshot from the Toolbox */
  initialContext?: InitialContext | null
  /** Current page markdown content sent as page_context with every request */
  pageContext?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const AUTO_COMPACT_THRESHOLD = 80 // percent
const PANEL_WIDTH_KEY = "qa-panel-width"
const DEFAULT_PANEL_WIDTH = 420
const MIN_PANEL_WIDTH = 300
const MAX_PANEL_WIDTH = 800

function getStoredPanelWidth(): number {
  try {
    const stored = localStorage.getItem(PANEL_WIDTH_KEY)
    if (stored) {
      const w = parseInt(stored, 10)
      if (w >= MIN_PANEL_WIDTH && w <= MAX_PANEL_WIDTH) return w
    }
  } catch { /* ignore */ }
  return DEFAULT_PANEL_WIDTH
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

function ChatDialog({
  isOpen,
  onClose,
  initialContext,
  pageContext,
}: ChatDialogProps) {
  const [messages, setMessages] = React.useState<ChatMessage[]>([])
  const [input, setInput] = React.useState("")
  const [pastedImage, setPastedImage] = React.useState<string | null>(null)
  const [isLoading, setIsLoading] = React.useState(false)
  const [contextUsage, setContextUsage] = React.useState<number>(0)
  const [isCompacting, setIsCompacting] = React.useState(false)
  const [sessionId, setSessionId] = React.useState(getSessionId)
  const [panelWidth, setPanelWidth] = React.useState(getStoredPanelWidth)

  const messagesEndRef = React.useRef<HTMLDivElement>(null)
  const inputRef = React.useRef<HTMLTextAreaElement>(null)
  const prevContextRef = React.useRef<InitialContext | null | undefined>(null)
  const autoSendPendingRef = React.useRef(false)

  // Persist panel width
  React.useEffect(() => {
    try { localStorage.setItem(PANEL_WIDTH_KEY, String(panelWidth)) } catch { /* ignore */ }
  }, [panelWidth])

  // Scroll to bottom when messages change
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isLoading])

  // Focus input when dialog opens
  React.useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }, [isOpen])

  // ------ Context usage polling ------
  const refreshContextUsage = React.useCallback(async () => {
    try {
      const usage = await fetchContextUsage(sessionId)
      setContextUsage(usage.usage_percent)

      if (usage.usage_percent >= AUTO_COMPACT_THRESHOLD) {
        setIsCompacting(true)
        await compactHistory(sessionId)
        const refreshed = await fetchContextUsage(sessionId)
        setContextUsage(refreshed.usage_percent)
        setIsCompacting(false)
      }
    } catch {
      // Session may not exist yet on first message
    }
  }, [sessionId])

  // ------ Send message ------
  const handleSend = React.useCallback(async (overrideText?: string, overrideImage?: string | null) => {
    const text = (overrideText ?? input).trim()
    const image = overrideImage !== undefined ? overrideImage : pastedImage
    if (!text && !image) return
    if (isLoading) return

    const userMessage: ChatMessage = {
      role: "user",
      content: text || "(image)",
      image: image ?? undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setPastedImage(null)
    setIsLoading(true)

    try {
      const customBaseUrl = getStoredCustomBaseUrl()
      const customApiKey = getStoredCustomApiKey()
      const customModel = getStoredCustomModel()
      const res = await sendMessage({
        sessionId,
        message: text || "Describe this image.",
        image: image ?? undefined,
        pageContext,
        model: customModel || undefined,
        customBaseUrl: customBaseUrl || undefined,
        customApiKey: customApiKey || undefined,
      })
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.response },
      ])
      await refreshContextUsage()
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err instanceof Error ? err.message : "Something went wrong."}`,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }, [input, pastedImage, isLoading, sessionId, pageContext, refreshContextUsage])

  // Handle new initial context arriving — auto-send
  React.useEffect(() => {
    if (!isOpen || !initialContext) return
    if (prevContextRef.current === initialContext) return
    prevContextRef.current = initialContext

    if (initialContext.type === "image") {
      // Auto-send the screenshot with a default prompt
      autoSendPendingRef.current = true
      setPastedImage(initialContext.data)
    }

    if (initialContext.type === "text" && initialContext.data) {
      // Auto-send a clean prompt — the highlighted text is already visible
      // in the context indicator card and available to the LLM via pageContext
      autoSendPendingRef.current = true
    }
  }, [isOpen, initialContext])

  // Auto-send when pending flag is set (after state updates settle)
  React.useEffect(() => {
    if (!autoSendPendingRef.current) return
    if (!isOpen) return

    autoSendPendingRef.current = false

    if (initialContext?.type === "image" && pastedImage) {
      handleSend("Describe this screenshot.", pastedImage)
    } else if (initialContext?.type === "text" && initialContext.data) {
      handleSend("Explain the highlighted text.", null)
    }
  }, [isOpen, pastedImage, initialContext, handleSend])

  // ------ Paste handler ------
  const handlePaste = React.useCallback(
    (e: React.ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          e.preventDefault()
          const file = item.getAsFile()
          if (!file) return
          const reader = new FileReader()
          reader.onload = () => {
            setPastedImage(reader.result as string)
          }
          reader.readAsDataURL(file)
          return
        }
      }
    },
    []
  )

  // ------ Keyboard shortcuts ------
  const handleKeyDown = React.useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend]
  )

  // ------ Manual compact ------
  const handleCompact = React.useCallback(async () => {
    setIsCompacting(true)
    try {
      await compactHistory(sessionId)
      const usage = await fetchContextUsage(sessionId)
      setContextUsage(usage.usage_percent)
    } catch {
      // ignore
    } finally {
      setIsCompacting(false)
    }
  }, [sessionId])

  // ------ New session ------
  const handleNewSession = React.useCallback(() => {
    const newId = resetSessionId()
    setSessionId(newId)
    setMessages([])
    setContextUsage(0)
    setInput("")
    setPastedImage(null)
  }, [])

  // ------ Resize drag handler ------
  const handleResizeMouseDown = React.useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      const startX = e.clientX
      const startWidth = panelWidth

      const onMouseMove = (ev: MouseEvent) => {
        const delta = startX - ev.clientX
        const newWidth = Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, startWidth + delta))
        setPanelWidth(newWidth)
      }

      const onMouseUp = () => {
        document.removeEventListener("mousemove", onMouseMove)
        document.removeEventListener("mouseup", onMouseUp)
        document.body.style.cursor = ""
        document.body.style.userSelect = ""
      }

      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"
      document.addEventListener("mousemove", onMouseMove)
      document.addEventListener("mouseup", onMouseUp)
    },
    [panelWidth]
  )

  // ------ Expandable screenshot state ------
  const [expandedImage, setExpandedImage] = React.useState(false)

  if (!isOpen) return null

  return (
    <div
      className="flex h-full shrink-0 flex-col border-l bg-background"
      style={{ width: panelWidth }}
    >
      {/* ---- Resize handle ---- */}
      <div
        className="absolute left-0 top-0 bottom-0 z-10 flex w-2 cursor-col-resize items-center justify-center hover:bg-primary/10 active:bg-primary/20"
        onMouseDown={handleResizeMouseDown}
        style={{ marginLeft: -4 }}
      >
        <GripVertical className="size-3 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100" />
      </div>

      {/* ---- Header ---- */}
      <div className="flex items-center justify-between border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <Bot className="size-5 text-muted-foreground" />
          <h2 className="text-sm font-semibold">QA Assistant</h2>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={handleNewSession}
            aria-label="New session"
            title="New session"
          >
            <span className="text-xs font-medium">New</span>
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onClose}
            aria-label="Close chat"
          >
            <X className="size-4" />
          </Button>
        </div>
      </div>

      {/* ---- Context usage bar ---- */}
      <div className="flex items-center gap-2 border-b px-4 py-2">
        <div className="flex-1">
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-300",
                contextUsage > 80
                  ? "bg-red-500"
                  : contextUsage > 50
                    ? "bg-yellow-500"
                    : "bg-green-500"
              )}
              style={{ width: `${Math.min(contextUsage, 100)}%` }}
            />
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            Context: {contextUsage.toFixed(1)}%
          </p>
        </div>
        <Button
          variant="outline"
          size="xs"
          onClick={handleCompact}
          disabled={isCompacting || messages.length === 0}
          className="gap-1"
        >
          {isCompacting ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Shrink className="size-3" />
          )}
          <span className="text-[10px]">Compact</span>
        </Button>
      </div>

      {/* ---- Source context indicator (redesigned) ---- */}
      {initialContext && (initialContext.data || initialContext.label) && (
        <div className="border-b px-4 py-3">
          <div className="rounded-lg border bg-muted/50 p-3">
            {initialContext.label && (
              <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                {initialContext.label}
              </p>
            )}
            {initialContext.type === "text" && initialContext.data && (
              <div className="border-l-2 border-primary pl-3">
                <p className="line-clamp-4 text-xs leading-relaxed text-foreground/80 italic">
                  {initialContext.data}
                </p>
              </div>
            )}
            {initialContext.type === "image" && (
              <button
                type="button"
                onClick={() => setExpandedImage(!expandedImage)}
                className="mt-1 block cursor-pointer"
              >
                <img
                  src={initialContext.data}
                  alt="Context screenshot"
                  className={cn(
                    "rounded border object-cover transition-all",
                    expandedImage ? "max-h-64 w-full object-contain" : "h-16"
                  )}
                />
                <span className="mt-1 block text-[10px] text-muted-foreground">
                  {expandedImage ? "Click to collapse" : "Click to expand"}
                </span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* ---- Messages ---- */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !isLoading && (
          <div className="flex h-full flex-col items-center justify-center text-center text-muted-foreground">
            <Bot className="mb-3 size-10 opacity-30" />
            <p className="text-sm">Ask a question about this page</p>
            <p className="mt-1 text-xs">
              Paste images with Ctrl+V / Cmd+V
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={cn(
              "mb-3 flex gap-2",
              msg.role === "user" ? "flex-row-reverse" : "flex-row"
            )}
          >
            {/* Avatar */}
            <div
              className={cn(
                "flex size-6 shrink-0 items-center justify-center rounded-full",
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground"
              )}
            >
              {msg.role === "user" ? (
                <User className="size-3.5" />
              ) : (
                <Bot className="size-3.5" />
              )}
            </div>

            {/* Bubble */}
            <div
              className={cn(
                "max-w-[85%] rounded-lg px-3 py-2 text-sm",
                msg.role === "user"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted"
              )}
            >
              {msg.image && (
                <img
                  src={msg.image}
                  alt="Pasted"
                  className="mb-2 max-h-40 rounded border object-contain"
                />
              )}
              {msg.role === "assistant" ? (
                <div className="[&_.prose]:text-sm [&_.prose]:leading-relaxed">
                  <MarkdownRenderer content={msg.content} />
                </div>
              ) : (
                <p className="whitespace-pre-wrap">{msg.content}</p>
              )}
            </div>
          </div>
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="mb-3 flex gap-2">
            <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
              <Bot className="size-3.5" />
            </div>
            <div className="rounded-lg bg-muted px-3 py-2">
              <Loader2 className="size-4 animate-spin text-muted-foreground" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* ---- Image preview ---- */}
      {pastedImage && (
        <div className="border-t px-4 py-2">
          <div className="relative inline-block">
            <img
              src={pastedImage}
              alt="Pasted preview"
              className="h-16 rounded border object-cover"
            />
            <button
              onClick={() => setPastedImage(null)}
              className="absolute -top-1.5 -right-1.5 flex size-4 items-center justify-center rounded-full bg-destructive text-destructive-foreground"
              aria-label="Remove image"
            >
              <X className="size-2.5" />
            </button>
          </div>
        </div>
      )}

      {/* ---- Input area ---- */}
      <div className="border-t px-4 py-3">
        <div className="flex items-end gap-2">
          <div className="relative flex-1">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onPaste={handlePaste}
              onKeyDown={handleKeyDown}
              placeholder="Ask a question…"
              rows={1}
              className="w-full resize-none rounded-lg border bg-background px-3 py-2 pr-8 text-sm outline-none ring-ring placeholder:text-muted-foreground focus:ring-1"
              style={{
                minHeight: "36px",
                maxHeight: "120px",
                height: "auto",
              }}
              onInput={(e) => {
                const target = e.target as HTMLTextAreaElement
                target.style.height = "auto"
                target.style.height = `${Math.min(target.scrollHeight, 120)}px`
              }}
            />
            <button
              onClick={() => {
                const fileInput = document.createElement("input")
                fileInput.type = "file"
                fileInput.accept = "image/*"
                fileInput.onchange = (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (!file) return
                  const reader = new FileReader()
                  reader.onload = () => setPastedImage(reader.result as string)
                  reader.readAsDataURL(file)
                }
                fileInput.click()
              }}
              className="absolute right-2 bottom-2 text-muted-foreground hover:text-foreground"
              aria-label="Attach image"
            >
              <ImageIcon className="size-4" />
            </button>
          </div>
          <Button
            size="icon"
            onClick={() => handleSend()}
            disabled={isLoading || (!input.trim() && !pastedImage)}
            aria-label="Send message"
          >
            {isLoading ? (
              <Loader2 className="size-4 animate-spin" />
            ) : (
              <ArrowUp className="size-4" />
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}

export { ChatDialog }
