import { Link } from "react-router-dom";
import { SearchIcon, HomeIcon, BookOpenIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLayoutContext } from "@/layouts/RootLayout";

export default function NotFound() {
  const { openSearch } = useLayoutContext();

  return (
    <div className="flex flex-col items-center justify-center gap-6 py-20 text-center">
      <div className="text-6xl font-bold text-muted-foreground/50">404</div>
      <div className="space-y-2">
        <h2 className="text-xl font-semibold">Page not found</h2>
        <p className="max-w-sm text-sm text-muted-foreground">
          The page you're looking for doesn't exist or may have been moved. Try
          searching or browse the sidebar to find what you need.
        </p>
      </div>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button variant="default" size="sm" className="gap-2" onClick={openSearch}>
          <SearchIcon className="size-4" />
          Search pages
        </Button>
        <Button variant="outline" size="sm" className="gap-2" asChild>
          <Link to="/">
            <HomeIcon className="size-4" />
            Go home
          </Link>
        </Button>
        <Button variant="ghost" size="sm" className="gap-2" asChild>
          <Link to="/foundational">
            <BookOpenIcon className="size-4" />
            Browse content
          </Link>
        </Button>
      </div>
    </div>
  );
}
