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

const API_KEY_STORAGE_KEY = "qa-settings-api-key"
const MODEL_STORAGE_KEY = "qa-settings-model"

const MODEL_OPTIONS = [
  { value: "claude-opus-4-6", label: "Opus 4.6" },
  { value: "claude-sonnet-4-6", label: "Sonnet 4.6" },
  { value: "claude-haiku-4-5-20251001", label: "Haiku 4.5" },
] as const

export function getStoredApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) ?? ""
  } catch {
    return ""
  }
}

export function getStoredModel(): string {
  try {
    return localStorage.getItem(MODEL_STORAGE_KEY) ?? ""
  } catch {
    return ""
  }
}

interface SettingsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

function SettingsDialog({ open, onOpenChange }: SettingsDialogProps) {
  const [apiKey, setApiKey] = React.useState("")
  const [model, setModel] = React.useState("")

  // Load values when dialog opens
  React.useEffect(() => {
    if (open) {
      setApiKey(getStoredApiKey())
      setModel(getStoredModel())
    }
  }, [open])

  const handleSave = () => {
    try {
      if (apiKey) {
        localStorage.setItem(API_KEY_STORAGE_KEY, apiKey)
      } else {
        localStorage.removeItem(API_KEY_STORAGE_KEY)
      }
      if (model) {
        localStorage.setItem(MODEL_STORAGE_KEY, model)
      } else {
        localStorage.removeItem(MODEL_STORAGE_KEY)
      }
    } catch {
      // ignore storage errors
    }
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>QA Assistant Settings</DialogTitle>
          <DialogDescription>
            Configure your Anthropic API key and preferred model.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="qa-api-key"
              className="text-sm font-medium"
            >
              Anthropic API Key
            </label>
            <input
              id="qa-api-key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-ant-..."
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none ring-ring placeholder:text-muted-foreground focus:ring-1"
            />
            <p className="text-xs text-muted-foreground">
              Stored in your browser. Sent to the server for API calls but never persisted server-side.
            </p>
          </div>

          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="qa-model"
              className="text-sm font-medium"
            >
              Model
            </label>
            <select
              id="qa-model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm outline-none ring-ring focus:ring-1"
            >
              <option value="">Default (server)</option>
              {MODEL_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        <DialogFooter>
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
