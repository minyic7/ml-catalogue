import * as React from "react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"

const CUSTOM_BASE_URL_KEY = "qa-settings-custom-base-url"
const CUSTOM_API_KEY_KEY = "qa-settings-custom-api-key"
const CUSTOM_MODEL_KEY = "qa-settings-custom-model"

export function getStoredCustomBaseUrl(): string {
  try {
    return localStorage.getItem(CUSTOM_BASE_URL_KEY) ?? ""
  } catch {
    return ""
  }
}

export function getStoredCustomApiKey(): string {
  try {
    return localStorage.getItem(CUSTOM_API_KEY_KEY) ?? ""
  } catch {
    return ""
  }
}

export function getStoredCustomModel(): string {
  try {
    return localStorage.getItem(CUSTOM_MODEL_KEY) ?? ""
  } catch {
    return ""
  }
}

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const [baseUrl, setBaseUrl] = React.useState("")
  const [apiKey, setApiKey] = React.useState("")
  const [model, setModel] = React.useState("")

  // Load values when dialog opens
  React.useEffect(() => {
    if (open) {
      setBaseUrl(getStoredCustomBaseUrl())
      setApiKey(getStoredCustomApiKey())
      setModel(getStoredCustomModel())
    }
  }, [open])

  const handleSave = () => {
    try {
      const setOrRemove = (key: string, value: string) => {
        if (value) localStorage.setItem(key, value)
        else localStorage.removeItem(key)
      }
      setOrRemove(CUSTOM_BASE_URL_KEY, baseUrl.trim())
      setOrRemove(CUSTOM_API_KEY_KEY, apiKey.trim())
      setOrRemove(CUSTOM_MODEL_KEY, model.trim())
    } catch {
      // ignore storage errors
    }
    onOpenChange(false)
  }

  const handleClear = () => {
    setBaseUrl("")
    setApiKey("")
    setModel("")
    try {
      localStorage.removeItem(CUSTOM_BASE_URL_KEY)
      localStorage.removeItem(CUSTOM_API_KEY_KEY)
      localStorage.removeItem(CUSTOM_MODEL_KEY)
    } catch {
      // ignore
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>QA Assistant Settings</DialogTitle>
          <DialogDescription>
            Optionally connect to a custom OpenAI-compatible endpoint (e.g. company LLM gateway).
            Leave empty to use the default server-side model.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="qa-base-url" className="text-sm font-medium">
              Base URL
            </label>
            <input
              id="qa-base-url"
              type="url"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://your-company-llm.example.com/v1"
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none ring-ring placeholder:text-muted-foreground focus:ring-1"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="qa-custom-key" className="text-sm font-medium">
              API Key
            </label>
            <input
              id="qa-custom-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none ring-ring placeholder:text-muted-foreground focus:ring-1"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="qa-custom-model" className="text-sm font-medium">
              Model
            </label>
            <input
              id="qa-custom-model"
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gpt-4o, claude-3-sonnet, etc."
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none ring-ring placeholder:text-muted-foreground focus:ring-1"
            />
          </div>

          <p className="text-xs text-muted-foreground">
            Stored in your browser only. When set, requests go through your custom endpoint
            instead of the default server-side model.
          </p>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={handleClear} className="mr-auto text-muted-foreground">
            Clear all
          </Button>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export { SettingsDialog }
export type { SettingsDialogProps }
