import { useState } from "react"
import { Loader2, Play, Square } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

export type RunMode = "quick" | "full"
export type DeviceType = "cpu" | "mps"

interface RunButtonProps {
  onRun: (mode: RunMode, device: DeviceType) => void
  onCancel: () => void
  isLoading: boolean
  elapsedSeconds: number
  timeoutWarning: boolean
  showDeviceToggle?: boolean
}

export function RunButton({
  onRun,
  onCancel,
  isLoading,
  elapsedSeconds,
  timeoutWarning,
  showDeviceToggle = false,
}: RunButtonProps) {
  const [mode, setMode] = useState<RunMode>("quick")
  const [device, setDevice] = useState<DeviceType>("cpu")

  return (
    <div className="flex items-center gap-2">
      <Button
        onClick={() => onRun(mode, device)}
        disabled={isLoading}
        size="default"
      >
        {isLoading ? (
          <Loader2 className="animate-spin" />
        ) : (
          <Play className="size-4" />
        )}
        {isLoading ? `Running… ${elapsedSeconds}s` : "Run"}
      </Button>

      {isLoading && (
        <Button
          onClick={onCancel}
          variant="outline"
          size="default"
        >
          <Square className="size-3" />
          Cancel
        </Button>
      )}

      {isLoading && timeoutWarning && (
        <span className="text-sm text-amber-600 dark:text-amber-400">
          Execution is taking longer than usual…
        </span>
      )}

      <ToggleGroup
        type="single"
        value={mode}
        onValueChange={(value) => {
          if (value) setMode(value as RunMode)
        }}
        variant="outline"
        size="sm"
      >
        <ToggleGroupItem value="quick" aria-label="Quick mode">
          ⚡ Quick
        </ToggleGroupItem>
        <ToggleGroupItem value="full" aria-label="Full mode">
          🔬 Full
        </ToggleGroupItem>
      </ToggleGroup>

      {showDeviceToggle && (
        <ToggleGroup
          type="single"
          value={device}
          onValueChange={(value) => {
            if (value) setDevice(value as DeviceType)
          }}
          variant="outline"
          size="sm"
        >
          <ToggleGroupItem value="cpu" aria-label="CPU device">
            CPU
          </ToggleGroupItem>
          <ToggleGroupItem value="mps" aria-label="MPS device">
            MPS
          </ToggleGroupItem>
        </ToggleGroup>
      )}
    </div>
  )
}
