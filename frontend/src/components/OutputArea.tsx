import { cn } from "@/lib/utils"

export interface OutputData {
  stdout?: string
  charts?: string[]
  error?: string
  warning?: string
}

interface OutputAreaProps {
  output: OutputData | null
}

export function OutputArea({ output }: OutputAreaProps) {
  if (!output) {
    return (
      <div className="rounded-lg border border-border bg-muted/30 p-6 text-center text-sm text-muted-foreground">
        Run your code to see output here.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {output.stdout && (
        <pre
          className={cn(
            "overflow-x-auto rounded-lg border border-border bg-muted/30 p-4",
            "font-mono text-sm text-foreground"
          )}
        >
          {output.stdout}
        </pre>
      )}

      {output.charts && output.charts.length > 0 && (
        <div className="space-y-2">
          {output.charts.map((chart, index) => (
            <img
              key={index}
              src={`data:image/png;base64,${chart}`}
              alt={`Chart ${index + 1}`}
              className="max-w-full rounded-lg border border-border"
            />
          ))}
        </div>
      )}

      {output.warning && (
        <pre
          className={cn(
            "overflow-x-auto rounded-lg border border-amber-400/50 bg-amber-50 p-4 dark:bg-amber-950/20",
            "font-mono text-sm text-amber-700 dark:text-amber-400"
          )}
        >
          {output.warning}
        </pre>
      )}

      {output.error && (
        <pre
          className={cn(
            "overflow-x-auto rounded-lg border border-destructive/50 bg-destructive/5 p-4",
            "font-mono text-sm text-destructive"
          )}
        >
          {output.error}
        </pre>
      )}
    </div>
  )
}
