import { MarkdownRenderer } from "../MarkdownRenderer";

const sampleMarkdown = `
# Markdown Rendering Demo

## Text Formatting

This paragraph demonstrates **bold text**, *italic text*, and \`inline code\`.
You can also use ~~strikethrough~~ for deleted content.

## Lists

### Unordered List
- First item
- Second item
  - Nested item
  - Another nested item
- Third item

### Ordered List
1. Step one
2. Step two
3. Step three

## Links and Images

Visit [GitHub](https://github.com) for more information.

## Code Fences

\`\`\`typescript
function fibonacci(n: number): number {
  if (n <= 1) return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}
\`\`\`

## Tables

| Feature       | Status    |
|---------------|-----------|
| Markdown      | ✅ Ready  |
| Math (KaTeX)  | ✅ Ready  |
| GFM           | ✅ Ready  |

## Blockquotes

> Mathematics is the queen of the sciences and number theory is the queen of mathematics.
> — Carl Friedrich Gauss

## Math Rendering

### Inline Math

Einstein's famous equation: $E = mc^2$. The quadratic formula is $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$.

### Block Math

The sum of a series:

$$\\sum_{i=1}^{n} x_i = x_1 + x_2 + \\cdots + x_n$$

The Gaussian integral:

$$\\int_{-\\infty}^{\\infty} e^{-x^2} \\, dx = \\sqrt{\\pi}$$

Euler's identity:

$$e^{i\\pi} + 1 = 0$$
`;

export function MarkdownDemo() {
  return (
    <div className="mx-auto max-w-3xl p-8">
      <h1 className="mb-6 text-2xl font-bold">MarkdownRenderer Demo</h1>
      <MarkdownRenderer content={sampleMarkdown} />
    </div>
  );
}
