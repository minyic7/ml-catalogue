import { CodeBlock } from "../CodeBlock";

const samplePython = `\
import numpy as np
from dataclasses import dataclass

@dataclass
class Particle:
    """A simple particle with position and velocity."""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0

    def step(self, dt: float = 0.01) -> None:
        # Update position based on velocity
        self.x += self.vx * dt
        self.y += self.vy * dt

def simulate(n_particles: int = 100, steps: int = 1000) -> list[Particle]:
    particles = [
        Particle(
            x=np.random.uniform(-10, 10),
            y=np.random.uniform(-10, 10),
            vx=np.random.normal(0, 1),
            vy=np.random.normal(0, 1),
        )
        for _ in range(n_particles)
    ]

    for _ in range(steps):
        for p in particles:
            p.step()

    return particles

if __name__ == "__main__":
    results = simulate(n_particles=50, steps=500)
    print(f"Simulated {len(results)} particles")`;

export function CodeBlockDemo() {
  return (
    <div className="max-w-3xl mx-auto p-8 space-y-6">
      <h1 className="text-2xl font-bold text-zinc-100">
        CodeBlock Component Demo
      </h1>
      <CodeBlock code={samplePython} language="python" />
    </div>
  );
}
