import * as React from "react"
import {
  ArrowUp,
  Bot,
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

  const messagesEndRef = React.useRef<HTMLDivElement>(null)
  const inputRef = React.useRef<HTMLTextAreaElement>(null)
  const prevContextRef = React.useRef<InitialContext | null | undefined>(null)

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

  // Handle new initial context arriving (e.g. user highlights text then opens chat)
  React.useEffect(() => {
    if (!isOpen || !initialContext) return
    if (prevContextRef.current === initialContext) return
    prevContextRef.current = initialContext

    if (initialContext.type === "image") {
      setPastedImage(initialContext.data)
    }
    // For text, pre-fill input with a question about the highlighted snippet
    if (initialContext.type === "text") {
      setInput(`Explain this:\n\n> ${initialContext.data}`)
    }
  }, [isOpen, initialContext])

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
  const handleSend = React.useCallback(async () => {
    const text = input.trim()
    if (!text && !pastedImage) return
    if (isLoading) return

    const userMessage: ChatMessage = {
      role: "user",
      content: text || "(image)",
      image: pastedImage ?? undefined,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setPastedImage(null)
    setIsLoading(true)

    try {
      const res = await sendMessage({
        sessionId,
        message: text || "Describe this image.",
        image: pastedImage ?? undefined,
        pageContext,
      })
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.response },
      ])
      // Refresh context usage after every response
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

  if (!isOpen) return null

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full flex-col border-l bg-background shadow-xl sm:w-[420px]">
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

      {/* ---- Source context indicator ---- */}
      {initialContext && (
        <div className="border-b px-4 py-2">
          <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
            {initialContext.label ? `From: ${initialContext.label}` : "Context"}
          </p>
          {initialContext.type === "text" && (
            <p className="mt-0.5 line-clamp-2 text-xs text-muted-foreground italic">
              &ldquo;{initialContext.data}&rdquo;
            </p>
          )}
          {initialContext.type === "image" && (
            <img
              src={initialContext.data}
              alt="Context screenshot"
              className="mt-1 h-12 rounded border object-cover"
            />
          )}
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
                // Trigger file input for image attachment
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
            onClick={handleSend}
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
