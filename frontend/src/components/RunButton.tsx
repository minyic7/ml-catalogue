import { useState } from "react"
import { Loader2, Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

export type RunMode = "quick" | "full"
export type DeviceType = "cpu" | "mps"

interface RunButtonProps {
  onRun: (mode: RunMode, device: DeviceType) => void
  isLoading: boolean
  showDeviceToggle?: boolean
}

export function RunButton({ onRun, isLoading, showDeviceToggle = false }: RunButtonProps) {
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
        {isLoading ? "Running…" : "Run"}
      </Button>

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
