import { useState } from "react"
import { Loader2, Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"

export type RunMode = "quick" | "full"

interface RunButtonProps {
  onRun: (mode: RunMode) => void
  isLoading: boolean
}

export function RunButton({ onRun, isLoading }: RunButtonProps) {
  const [mode, setMode] = useState<RunMode>("quick")

  return (
    <div className="flex items-center gap-2">
      <Button
        onClick={() => onRun(mode)}
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
    </div>
  )
}
