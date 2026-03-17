import { useProgress } from "@/hooks/useProgress";
import { Progress } from "@/components/ui/progress";

export default function HomePage() {
  const { totalPages, completedPages } = useProgress();
  const percentage = totalPages > 0 ? Math.round((completedPages / totalPages) * 100) : 0;

  return (
    <div>
      <h2 className="text-2xl font-bold">Welcome to ML Catalogue</h2>
      <p className="mt-2 text-muted-foreground">
        Select an item from the sidebar to get started.
      </p>

      <div className="mt-6 max-w-md space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="font-medium">Learning Progress</span>
          <span className="text-muted-foreground">
            {completedPages} / {totalPages} pages completed ({percentage}%)
          </span>
        </div>
        <Progress value={completedPages} max={totalPages} />
      </div>
    </div>
  );
}
