import { Moon, Sun, Monitor } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useTheme } from '@/components/ThemeProvider';

const icons = {
  light: Sun,
  dark: Moon,
  system: Monitor,
} as const;

const next: Record<string, 'light' | 'dark' | 'system'> = {
  light: 'dark',
  dark: 'system',
  system: 'light',
};

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const Icon = icons[theme];

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={() => setTheme(next[theme])}
      aria-label={`Theme: ${theme}. Click to switch.`}
    >
      <Icon className="size-4" />
    </Button>
  );
}
