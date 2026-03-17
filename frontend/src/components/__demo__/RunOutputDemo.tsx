import { useState } from "react"

import { RunButton, type RunMode } from "@/components/RunButton"
import { OutputArea, type OutputData } from "@/components/OutputArea"

const MOCK_OUTPUT: OutputData = {
  stdout: `Loading dataset... done.
Processing 1,247 records across 3 categories.
Category A: 423 records (33.9%)
Category B: 512 records (41.1%)
Category C: 312 records (25.0%)

Summary statistics:
  Mean:   42.7
  Median: 38.2
  Std:    12.4`,
  charts: [
    "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjE1MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMzAwIiBoZWlnaHQ9IjE1MCIgZmlsbD0iI2YwZjBmMCIgcng9IjgiLz48dGV4dCB4PSIxNTAiIHk9Ijc1IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZpbGw9IiM4ODgiIGZvbnQtc2l6ZT0iMTQiPkNoYXJ0IFBsYWNlaG9sZGVyPC90ZXh0Pjwvc3ZnPg==",
  ],
}

const MOCK_ERROR_OUTPUT: OutputData = {
  stdout: "Loading dataset... done.\nProcessing records...",
  error: `Traceback (most recent call last):
  File "analysis.py", line 42, in <module>
    result = process(data)
  File "analysis.py", line 28, in process
    raise ValueError("Column 'revenue' contains null values")
ValueError: Column 'revenue' contains null values`,
}

export function RunOutputDemo() {
  const [output, setOutput] = useState<OutputData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [runCount, setRunCount] = useState(0)

  const handleRun = (mode: RunMode) => {
    setIsLoading(true)
    setOutput(null)

    setTimeout(() => {
      const nextCount = runCount + 1
      setRunCount(nextCount)

      // Alternate between success and error output
      if (nextCount % 3 === 0) {
        setOutput(MOCK_ERROR_OUTPUT)
      } else {
        setOutput({
          ...MOCK_OUTPUT,
          stdout: `[${mode} mode]\n${MOCK_OUTPUT.stdout}`,
        })
      }

      setIsLoading(false)
    }, 1000)
  }

  return (
    <div className="mx-auto max-w-2xl space-y-4 p-6">
      <h2 className="text-lg font-semibold">Run & Output Demo</h2>
      <RunButton onRun={handleRun} isLoading={isLoading} />
      <OutputArea output={output} />
    </div>
  )
}
