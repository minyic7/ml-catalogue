import type { Chapter } from "../types";

const mdpBellmanMarkdown = `
# MDP & Bellman Equations

**Reinforcement Learning (RL)** is a framework where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, the agent receives no labeled examples — only reward signals that indicate how good its actions were.

## Markov Decision Process (MDP)

The mathematical foundation of RL is the **Markov Decision Process**, defined by the tuple $(S, A, P, R, \\gamma)$:

- $S$ — a finite set of **states**
- $A$ — a finite set of **actions**
- $P(s' \\mid s, a)$ — **transition probability** of reaching state $s'$ from state $s$ by taking action $a$
- $R(s, a, s')$ — **reward** received after transitioning from $s$ to $s'$ via action $a$
- $\\gamma \\in [0, 1)$ — **discount factor** that controls how much the agent values future rewards

The **Markov property** states that the future depends only on the current state, not on the history of past states:

$$P(s_{t+1} \\mid s_t, a_t, s_{t-1}, a_{t-1}, \\dots) = P(s_{t+1} \\mid s_t, a_t)$$

## Policy and Value Functions

A **policy** $\\pi(a \\mid s)$ defines the agent's behaviour — the probability of taking action $a$ in state $s$.

The **state-value function** $V^\\pi(s)$ measures how good it is to be in state $s$ under policy $\\pi$:

$$V^\\pi(s) = \\mathbb{E}_\\pi\\!\\left[\\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1} \\mid S_t = s\\right]$$

The **action-value function** $Q^\\pi(s, a)$ measures how good it is to take action $a$ in state $s$ and then follow $\\pi$:

$$Q^\\pi(s, a) = \\mathbb{E}_\\pi\\!\\left[\\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1} \\mid S_t = s, A_t = a\\right]$$

## Bellman Equations

The Bellman equation expresses a recursive relationship between the value of a state and the values of its successor states.

**Bellman Expectation Equation** for $V^\\pi$:

$$V^\\pi(s) = \\sum_a \\pi(a \\mid s) \\sum_{s'} P(s' \\mid s, a)\\bigl[R(s, a, s') + \\gamma\\, V^\\pi(s')\\bigr]$$

**Bellman Optimality Equation** for the optimal value function $V^*$:

$$V^*(s) = \\max_a \\sum_{s'} P(s' \\mid s, a)\\bigl[R(s, a, s') + \\gamma\\, V^*(s')\\bigr]$$

The optimal policy $\\pi^*$ can be derived from $V^*$ by choosing the action that maximises the expected value at each state.

## Value Iteration

**Value iteration** is a dynamic programming algorithm that computes $V^*$ by repeatedly applying the Bellman optimality update:

$$V_{k+1}(s) = \\max_a \\sum_{s'} P(s' \\mid s, a)\\bigl[R(s, a, s') + \\gamma\\, V_k(s')\\bigr]$$

This converges to $V^*$ as $k \\to \\infty$. Once we have $V^*$, we extract the optimal policy greedily.

## Why This Matters

MDPs and Bellman equations are the theoretical backbone of all RL algorithms. Run the code to see value iteration solve a grid world — watch the value function converge and the optimal policy emerge.
`;

const mdpBellmanCode = `import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

QUICK = os.environ.get("DATASET_MODE", "quick") == "quick"

# ============================================================
# Grid World Environment
# ============================================================
# 5x5 grid, agent starts top-left, goal at bottom-right
# Walls block movement, stepping into a wall keeps agent in place
# Reward: +1 at goal, -0.04 per step (encourages shortest path)

ROWS, COLS = 5, 5
GOAL = (4, 4)
WALLS = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)}
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ['U', 'D', 'L', 'R']
GAMMA = 0.95
STEP_REWARD = -0.04
GOAL_REWARD = 1.0

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in WALLS

def step(s, a_idx):
    """Deterministic transition: returns (next_state, reward)."""
    dr, dc = ACTIONS[a_idx]
    nr, nc = s[0] + dr, s[1] + dc
    if is_valid(nr, nc):
        ns = (nr, nc)
    else:
        ns = s  # stay in place
    reward = GOAL_REWARD if ns == GOAL else STEP_REWARD
    return ns, reward

# --- Value Iteration ---
print("=== Value Iteration on 5x5 Grid World ===\\n")
V = np.zeros((ROWS, COLS))
policy = np.full((ROWS, COLS), -1, dtype=int)
max_iters = 50 if QUICK else 200
theta = 1e-6  # convergence threshold

history = []
for iteration in range(max_iters):
    delta = 0.0
    V_new = V.copy()
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == GOAL or (r, c) in WALLS:
                continue
            values = []
            for a_idx in range(len(ACTIONS)):
                ns, reward = step((r, c), a_idx)
                values.append(reward + GAMMA * V[ns[0], ns[1]])
            best = max(values)
            V_new[r, c] = best
            policy[r, c] = int(np.argmax(values))
            delta = max(delta, abs(best - V[r, c]))
    V = V_new
    history.append(delta)
    if delta < theta:
        print(f"Converged after {iteration + 1} iterations (delta={delta:.2e})")
        break

# Display the value function
print(f"\\nOptimal Value Function V*:")
for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        if (r, c) in WALLS:
            row_str += "  WALL "
        elif (r, c) == GOAL:
            row_str += " GOAL  "
        else:
            row_str += f" {V[r, c]:5.2f} "
    print(row_str)

# Display optimal policy
print(f"\\nOptimal Policy:")
for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        if (r, c) in WALLS:
            row_str += "  #  "
        elif (r, c) == GOAL:
            row_str += "  G  "
        else:
            row_str += f"  {ACTION_NAMES[policy[r, c]]}  "
    print(row_str)

# Trace optimal path from start
print(f"\\nOptimal path from (0,0) to goal:")
s = (0, 0)
path = [s]
for _ in range(25):
    if s == GOAL:
        break
    a = policy[s[0], s[1]]
    s, _ = step(s, a)
    path.append(s)
print(f"  {' -> '.join(str(p) for p in path)}")
print(f"  Path length: {len(path) - 1} steps")

# --- Visualisation ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) Value function heatmap
ax = axes[0]
V_display = V.copy()
for (r, c) in WALLS:
    V_display[r, c] = np.nan
im = ax.imshow(V_display, cmap='YlOrRd', interpolation='nearest')
for r in range(ROWS):
    for c in range(COLS):
        if (r, c) in WALLS:
            ax.text(c, r, '#', ha='center', va='center', fontsize=14, fontweight='bold')
        elif (r, c) == GOAL:
            ax.text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            ax.text(c, r, f'{V[r,c]:.2f}', ha='center', va='center', fontsize=10)
ax.set_title('Optimal Value Function V*', fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)

# 2) Policy arrows
ax = axes[1]
ax.set_xlim(-0.5, COLS - 0.5)
ax.set_ylim(ROWS - 0.5, -0.5)
arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
for r in range(ROWS):
    for c in range(COLS):
        if (r, c) in WALLS:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='gray'))
        elif (r, c) == GOAL:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='gold'))
            ax.text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            dx, dy = arrow_map[policy[r, c]]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.1, fc='steelblue', ec='steelblue')
# Draw optimal path
for i in range(len(path) - 1):
    r1, c1 = path[i]
    r2, c2 = path[i + 1]
    ax.plot([c1, c2], [r1, r2], 'r-', linewidth=2.5, alpha=0.7)
ax.set_title('Optimal Policy + Path', fontsize=12)
ax.set_xticks(range(COLS))
ax.set_yticks(range(ROWS))
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 3) Convergence curve
ax = axes[2]
ax.plot(history, 'b-', linewidth=1.5)
ax.set_xlabel('Iteration')
ax.set_ylabel('Max Value Change (delta)')
ax.set_title('Value Iteration Convergence', fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nGrid world visualisation saved to output.png")
`;

const qLearningMarkdown = `
# Q-Learning

While value iteration requires complete knowledge of the environment (transition probabilities), **Q-learning** is a **model-free** algorithm that learns directly from experience. The agent explores the environment, observes rewards, and incrementally builds a Q-table mapping state-action pairs to expected returns.

## The Q-Table

The Q-function $Q(s, a)$ estimates the expected cumulative reward of taking action $a$ in state $s$ and then following the optimal policy. We store these values in a table with one entry per state-action pair.

## Q-Learning Update Rule

After taking action $a$ in state $s$, observing reward $r$ and next state $s'$, we update:

$$Q(s, a) \\leftarrow Q(s, a) + \\alpha \\bigl[r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)\\bigr]$$

where:
- $\\alpha \\in (0, 1]$ is the **learning rate** — how much we trust new experience over old estimates
- $\\gamma$ is the **discount factor** — how much we value future rewards
- $r + \\gamma \\max_{a'} Q(s', a')$ is the **TD target** — the best estimate of the true value
- $r + \\gamma \\max_{a'} Q(s', a') - Q(s, a)$ is the **temporal difference (TD) error**

The key insight is that Q-learning is **off-policy**: it always updates toward the greedy best action $\\max_{a'} Q(s', a')$, even if the agent chose a different action during exploration.

## Exploration vs. Exploitation

The agent faces a fundamental dilemma: should it **exploit** what it already knows (choose the action with the highest Q-value) or **explore** new actions that might lead to better outcomes?

The **$\\varepsilon$-greedy** strategy balances this:

$$a = \\begin{cases} \\text{random action} & \\text{with probability } \\varepsilon \\\\ \\arg\\max_a Q(s, a) & \\text{with probability } 1 - \\varepsilon \\end{cases}$$

Typically $\\varepsilon$ starts high (e.g., 1.0) and decays over time, shifting from exploration to exploitation as the agent learns.

## Convergence Guarantees

Q-learning converges to the optimal Q-function $Q^*$ under two conditions:
1. Every state-action pair is visited infinitely often
2. The learning rate $\\alpha$ satisfies: $\\sum_t \\alpha_t = \\infty$ and $\\sum_t \\alpha_t^2 < \\infty$

In practice, a fixed small $\\alpha$ works well for finite environments.

## Why This Matters

Q-learning is the foundation of modern RL algorithms. Its model-free, off-policy nature makes it practical for environments where transition dynamics are unknown. Run the code to see Q-learning solve a grid world with obstacles — watch the Q-values converge and the agent discover the optimal path.
`;

const qLearningCode = `import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

QUICK = os.environ.get("DATASET_MODE", "quick") == "quick"

# ============================================================
# Grid World for Q-Learning
# ============================================================
ROWS, COLS = 5, 5
START = (0, 0)
GOAL = (4, 4)
WALLS = {(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)}
TRAPS = {(3, 3): -1.0, (0, 4): -0.5}  # penalty states

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U, D, L, R
ACTION_NAMES = ['U', 'D', 'L', 'R']
N_ACTIONS = len(ACTIONS)

def is_valid(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in WALLS

def env_step(state, action_idx):
    dr, dc = ACTIONS[action_idx]
    nr, nc = state[0] + dr, state[1] + dc
    if is_valid(nr, nc):
        ns = (nr, nc)
    else:
        ns = state
    if ns == GOAL:
        return ns, 1.0, True
    elif ns in TRAPS:
        return ns, TRAPS[ns], False
    else:
        return ns, -0.04, False

# --- Q-Learning ---
np.random.seed(42)
Q = np.zeros((ROWS, COLS, N_ACTIONS))

alpha = 0.1        # learning rate
gamma = 0.95       # discount factor
epsilon_start = 1.0
epsilon_end = 0.01
n_episodes = 300 if QUICK else 2000
epsilon_decay = epsilon_start / (n_episodes * 0.8)

episode_rewards = []
episode_lengths = []

print("=== Q-Learning on Grid World ===\\n")

for ep in range(n_episodes):
    state = START
    total_reward = 0
    steps = 0
    epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)

    for _ in range(100):  # max steps per episode
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(N_ACTIONS)
        else:
            action = int(np.argmax(Q[state[0], state[1]]))

        next_state, reward, done = env_step(state, action)

        # Q-learning update
        td_target = reward + gamma * np.max(Q[next_state[0], next_state[1]]) * (1 - done)
        td_error = td_target - Q[state[0], state[1], action]
        Q[state[0], state[1], action] += alpha * td_error

        state = next_state
        total_reward += reward
        steps += 1
        if done:
            break

    episode_rewards.append(total_reward)
    episode_lengths.append(steps)

    if (ep + 1) % (n_episodes // 5) == 0:
        avg_r = np.mean(episode_rewards[-50:])
        avg_l = np.mean(episode_lengths[-50:])
        print(f"  Episode {ep+1:>5}/{n_episodes}  eps={epsilon:.3f}  "
              f"avg_reward={avg_r:.2f}  avg_steps={avg_l:.1f}")

# Extract learned policy
policy = np.argmax(Q, axis=2)

print(f"\\nLearned Q-values (max over actions):")
for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        if (r, c) in WALLS:
            row_str += "  WALL "
        elif (r, c) == GOAL:
            row_str += " GOAL  "
        else:
            row_str += f" {np.max(Q[r, c]):5.2f} "
    print(row_str)

print(f"\\nLearned Policy:")
for r in range(ROWS):
    row_str = ""
    for c in range(COLS):
        if (r, c) in WALLS:
            row_str += "  #  "
        elif (r, c) == GOAL:
            row_str += "  G  "
        else:
            row_str += f"  {ACTION_NAMES[policy[r, c]]}  "
    print(row_str)

# Trace learned path
s = START
path = [s]
for _ in range(25):
    if s == GOAL:
        break
    a = int(np.argmax(Q[s[0], s[1]]))
    s, _, done = env_step(s, a)
    path.append(s)
    if done:
        break
print(f"\\nLearned path: {' -> '.join(str(p) for p in path)}")
print(f"Path length: {len(path) - 1} steps")

# --- Visualisation ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) Q-value heatmap
ax = axes[0]
V = np.max(Q, axis=2)
V_display = V.copy()
for (r, c) in WALLS:
    V_display[r, c] = np.nan
im = ax.imshow(V_display, cmap='YlOrRd', interpolation='nearest')
for r in range(ROWS):
    for c in range(COLS):
        if (r, c) in WALLS:
            ax.text(c, r, '#', ha='center', va='center', fontsize=14, fontweight='bold')
        elif (r, c) == GOAL:
            ax.text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
        elif (r, c) in TRAPS:
            ax.text(c, r, f'{V[r,c]:.2f}\\nTRAP', ha='center', va='center', fontsize=8, color='red')
        else:
            ax.text(c, r, f'{V[r,c]:.2f}', ha='center', va='center', fontsize=10)
ax.set_title('Learned Q-values (max)', fontsize=12)
plt.colorbar(im, ax=ax, shrink=0.8)

# 2) Learned policy with path
ax = axes[1]
ax.set_xlim(-0.5, COLS - 0.5)
ax.set_ylim(ROWS - 0.5, -0.5)
arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
for r in range(ROWS):
    for c in range(COLS):
        if (r, c) in WALLS:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='gray'))
        elif (r, c) == GOAL:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='gold'))
            ax.text(c, r, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
        elif (r, c) in TRAPS:
            ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, color='lightsalmon', alpha=0.5))
            dx, dy = arrow_map[policy[r, c]]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.1, fc='darkred', ec='darkred')
        else:
            dx, dy = arrow_map[policy[r, c]]
            ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.1, fc='steelblue', ec='steelblue')
for i in range(len(path) - 1):
    r1, c1 = path[i]
    r2, c2 = path[i + 1]
    ax.plot([c1, c2], [r1, r2], 'r-', linewidth=2.5, alpha=0.7)
ax.set_title('Learned Policy + Path', fontsize=12)
ax.set_xticks(range(COLS))
ax.set_yticks(range(ROWS))
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# 3) Training curves
ax = axes[2]
window = 20
smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
smoothed_lengths = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
ax.plot(smoothed_rewards, 'b-', linewidth=1.5, label='Reward')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward (smoothed)', color='b')
ax2 = ax.twinx()
ax2.plot(smoothed_lengths, 'r-', linewidth=1.5, alpha=0.7, label='Steps')
ax2.set_ylabel('Steps (smoothed)', color='r')
ax.set_title('Q-Learning Training Progress', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nQ-learning visualisation saved to output.png")
`;

const policyGradientMarkdown = `
# Policy Gradient (REINFORCE)

Q-learning works by learning a value function and deriving a policy from it. **Policy gradient** methods take a fundamentally different approach: they directly parameterise and optimise the policy itself.

## Why Policy Gradients?

Value-based methods like Q-learning have limitations:
- They struggle with **continuous action spaces** (the $\\max$ over actions becomes intractable)
- They learn **deterministic** policies, which can't represent optimal stochastic strategies
- Small changes in Q-values can cause large, abrupt policy changes

Policy gradient methods address all of these by optimising a parameterised policy $\\pi_\\theta(a \\mid s)$ directly.

## Policy Parameterisation

We represent the policy as a function with learnable parameters $\\theta$. For discrete actions, we typically use a **softmax** over action scores:

$$\\pi_\\theta(a \\mid s) = \\frac{\\exp(h_\\theta(s, a))}{\\sum_{a'} \\exp(h_\\theta(s, a'))}$$

where $h_\\theta(s, a)$ is a score function — either a linear model or a neural network.

## The Policy Gradient Theorem

We want to maximise the expected return:

$$J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[\\sum_{t=0}^{T} \\gamma^t r_t\\right]$$

The **policy gradient theorem** (Sutton et al., 2000) gives us the gradient:

$$\\nabla_\\theta J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta}\\!\\left[\\sum_{t=0}^{T} \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t) \\cdot G_t\\right]$$

where $G_t = \\sum_{k=t}^{T} \\gamma^{k-t} r_k$ is the **return** from time step $t$.

This is remarkable: we can compute the gradient of the expected return using only samples from the policy, without knowing the environment dynamics.

## The REINFORCE Algorithm

**REINFORCE** (Williams, 1992) is the simplest policy gradient algorithm:

1. Sample a full episode $\\tau = (s_0, a_0, r_0, \\dots, s_T)$ using the current policy $\\pi_\\theta$
2. For each time step, compute the return $G_t$
3. Update parameters: $\\theta \\leftarrow \\theta + \\alpha \\sum_t \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t) \\cdot G_t$

## Variance Reduction with Baselines

Raw REINFORCE has **high variance** because returns $G_t$ can vary wildly between episodes. A common fix is to subtract a **baseline** $b(s_t)$ from the return:

$$\\nabla_\\theta J(\\theta) \\approx \\sum_t \\nabla_\\theta \\log \\pi_\\theta(a_t \\mid s_t) \\cdot (G_t - b(s_t))$$

The baseline does not introduce bias (as long as it doesn't depend on $a_t$) but significantly reduces variance. A common choice is $b(s_t) = \\bar{G}$, the average return.

## Why This Matters

Policy gradient methods are the basis for modern RL algorithms like PPO, A3C, and SAC. Run the code to see REINFORCE learn to balance a simple CartPole-like environment — watch the training curve as the agent improves.
`;

const policyGradientCode = `import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

QUICK = os.environ.get("DATASET_MODE", "quick") == "quick"

# ============================================================
# Simple CartPole Environment (no gym dependency)
# ============================================================
# State: [x, x_dot, theta, theta_dot]
# Action: 0 (push left) or 1 (push right)

class SimpleCartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # time step
        self.max_steps = 200

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sin_th) / self.total_mass
        theta_acc = (self.gravity * sin_th - cos_th * temp) / (
            self.length * (4.0/3.0 - self.masspole * cos_th**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * cos_th / self.total_mass

        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1

        done = (abs(x) > 2.4 or abs(theta) > 0.21 or self.steps >= self.max_steps)
        reward = 1.0 if not done or self.steps >= self.max_steps else 0.0
        return self.state.copy(), reward, done

# ============================================================
# REINFORCE with Softmax Linear Policy
# ============================================================
np.random.seed(42)

n_features = 4
n_actions = 2
theta = np.random.randn(n_features, n_actions) * 0.01

alpha = 0.005 if QUICK else 0.002
gamma = 0.99
n_episodes = 500 if QUICK else 2000

def softmax(logits):
    logits = logits - np.max(logits)  # numerical stability
    exp_l = np.exp(logits)
    return exp_l / exp_l.sum()

def select_action(state, theta):
    logits = state @ theta
    probs = softmax(logits)
    action = np.random.choice(n_actions, p=probs)
    return action, probs

env = SimpleCartPole()
episode_rewards = []

print("=== REINFORCE Policy Gradient on CartPole ===\\n")

for ep in range(n_episodes):
    state = env.reset()
    states, actions, rewards = [], [], []

    # Collect episode
    done = False
    while not done:
        action, probs = select_action(state, theta)
        next_state, reward, done = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    total_reward = sum(rewards)
    episode_rewards.append(total_reward)

    # Compute returns with baseline
    T = len(rewards)
    returns = np.zeros(T)
    G = 0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    baseline = returns.mean()
    advantages = returns - baseline

    # Policy gradient update
    for t in range(T):
        s = states[t]
        a = actions[t]
        logits = s @ theta
        probs = softmax(logits)

        # Gradient of log pi(a|s) for softmax linear policy
        grad_log = np.outer(s, -probs)
        grad_log[:, a] += s

        theta += alpha * (gamma ** t) * advantages[t] * grad_log

    if (ep + 1) % (n_episodes // 5) == 0:
        avg_r = np.mean(episode_rewards[-50:])
        print(f"  Episode {ep+1:>5}/{n_episodes}  avg_reward={avg_r:.1f}")

# --- Results ---
print(f"\\nFinal average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.1f}")
print(f"Best episode reward: {max(episode_rewards):.1f}")

# Test the learned policy
test_rewards = []
for _ in range(20):
    state = env.reset()
    done = False
    total = 0
    while not done:
        logits = state @ theta
        action = int(np.argmax(softmax(logits)))
        state, r, done = env.step(action)
        total += r
    test_rewards.append(total)
print(f"Test performance (20 episodes): {np.mean(test_rewards):.1f} +/- {np.std(test_rewards):.1f}")

# --- Visualisation ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1) Training curve
ax = axes[0]
window = 30
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax.plot(episode_rewards, alpha=0.3, color='steelblue', linewidth=0.5)
ax.plot(range(window-1, len(episode_rewards)), smoothed, 'b-', linewidth=2, label=f'Smoothed ({window}-ep)')
ax.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Max reward (200)')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('REINFORCE Training Curve', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2) Policy visualisation (theta vs angle)
ax = axes[1]
angles = np.linspace(-0.25, 0.25, 200)
prob_right = []
for ang in angles:
    s = np.array([0.0, 0.0, ang, 0.0])
    logits = s @ theta
    probs = softmax(logits)
    prob_right.append(probs[1])
ax.plot(angles, prob_right, 'b-', linewidth=2)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(angles, prob_right, 0.5, where=np.array(prob_right) > 0.5, alpha=0.2, color='green', label='Push right')
ax.fill_between(angles, prob_right, 0.5, where=np.array(prob_right) < 0.5, alpha=0.2, color='red', label='Push left')
ax.set_xlabel('Pole Angle (radians)')
ax.set_ylabel('P(push right)')
ax.set_title('Learned Policy: Action Probability vs Pole Angle', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nTraining curve and policy plot saved to output.png")
`;

const deepRLMarkdown = `
# Deep RL (DQN)

When the state space is large or continuous, storing a Q-table becomes impractical. **Deep Q-Networks (DQN)** replace the table with a neural network that approximates $Q(s, a; \\theta)$.

## From Q-Table to Neural Network

Instead of maintaining a table $Q(s, a)$, we train a neural network $Q(s, a; \\theta)$ that takes a state $s$ as input and outputs Q-values for all actions. The network generalises across similar states — nearby states get similar Q-values without visiting each one individually.

## The DQN Loss Function

DQN minimises the mean squared TD error:

$$L(\\theta) = \\mathbb{E}\\!\\left[\\bigl(r + \\gamma \\max_{a'} Q(s', a'; \\theta^{-}) - Q(s, a; \\theta)\\bigr)^2\\right]$$

where $\\theta^{-}$ are the parameters of a separate **target network** (explained below).

## Experience Replay

Naively training on sequential experience causes two problems:
1. **Correlation**: consecutive samples are highly correlated, violating the i.i.d. assumption of SGD
2. **Catastrophic forgetting**: the network overfits to recent experience and forgets earlier lessons

**Experience replay** stores transitions $(s, a, r, s', \\text{done})$ in a replay buffer $\\mathcal{D}$. During training, we sample random mini-batches from $\\mathcal{D}$:

$$\\theta \\leftarrow \\theta - \\alpha\\, \\nabla_\\theta L(\\theta), \\quad \\text{where } (s, a, r, s') \\sim \\text{Uniform}(\\mathcal{D})$$

This breaks correlation and allows the network to learn from diverse experiences.

## Target Network

Using the same network $Q(s, a; \\theta)$ for both the current estimate and the TD target creates a moving target problem — the target shifts with every update, causing instability.

The **target network** $Q(s, a; \\theta^{-})$ is a copy of the main network whose parameters $\\theta^{-}$ are updated less frequently:

$$\\theta^{-} \\leftarrow \\theta \\quad \\text{every } C \\text{ steps}$$

This stabilises training by keeping the target fixed between updates.

## The DQN Algorithm

1. Initialise replay buffer $\\mathcal{D}$, Q-network $\\theta$, target network $\\theta^{-} = \\theta$
2. For each episode:
   - Select action using $\\varepsilon$-greedy on $Q(s, \\cdot; \\theta)$
   - Execute action, observe $(r, s')$, store $(s, a, r, s', \\text{done})$ in $\\mathcal{D}$
   - Sample mini-batch from $\\mathcal{D}$
   - Compute targets: $y = r + \\gamma \\max_{a'} Q(s', a'; \\theta^{-}) \\cdot (1 - \\text{done})$
   - Update $\\theta$ by minimising $(y - Q(s, a; \\theta))^2$
   - Periodically update $\\theta^{-} \\leftarrow \\theta$

## DQN Variants

Several improvements have been proposed:
- **Double DQN**: uses the online network to select actions and the target network to evaluate them, reducing overestimation bias
- **Dueling DQN**: separates value and advantage streams in the network architecture
- **Prioritised replay**: samples important transitions more frequently

## Why This Matters

DQN was the breakthrough that demonstrated deep RL could learn complex behaviours directly from high-dimensional inputs. Run the code to see a DQN agent learn to balance a CartPole — compare the training stability and sample efficiency with REINFORCE.
`;

const deepRLCode = `import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device(os.environ.get("ML_CATALOGUE_DEVICE", "cpu"))
QUICK = os.environ.get("DATASET_MODE", "quick") == "quick"

# ============================================================
# Simple CartPole Environment (no gym dependency)
# ============================================================
class SimpleCartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.max_steps = 200

    def reset(self):
        self.state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sin_th) / self.total_mass
        theta_acc = (self.gravity * sin_th - cos_th * temp) / (
            self.length * (4.0/3.0 - self.masspole * cos_th**2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * cos_th / self.total_mass

        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        done = (abs(x) > 2.4 or abs(theta) > 0.21 or self.steps >= self.max_steps)
        reward = 1.0 if not done or self.steps >= self.max_steps else 0.0
        return self.state.copy(), reward, done

# ============================================================
# DQN Components
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

# ============================================================
# DQN Training
# ============================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

env = SimpleCartPole()
q_net = QNetwork().to(device)
target_net = QNetwork().to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

replay = ReplayBuffer(capacity=10000)
batch_size = 64
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
n_episodes = 300 if QUICK else 1000
epsilon_decay = epsilon_start / (n_episodes * 0.7)
target_update_freq = 10  # episodes

episode_rewards = []
losses = []

print("=== DQN on CartPole ===\\n")

for ep in range(n_episodes):
    state = env.reset()
    total_reward = 0
    epsilon = max(epsilon_end, epsilon_start - ep * epsilon_decay)

    for step_count in range(200):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_net(s_tensor)
                action = int(q_values.argmax(dim=1).item())

        next_state, reward, done = env.step(action)
        replay.push(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward

        # Train on mini-batch
        if len(replay) >= batch_size:
            s_batch, a_batch, r_batch, ns_batch, d_batch = replay.sample(batch_size)

            s_t = torch.tensor(s_batch, dtype=torch.float32).to(device)
            a_t = torch.tensor(a_batch, dtype=torch.long).to(device)
            r_t = torch.tensor(r_batch, dtype=torch.float32).to(device)
            ns_t = torch.tensor(ns_batch, dtype=torch.float32).to(device)
            d_t = torch.tensor(d_batch, dtype=torch.float32).to(device)

            # Current Q-values
            current_q = q_net(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)

            # Target Q-values (from target network)
            with torch.no_grad():
                next_q = target_net(ns_t).max(dim=1)[0]
                target_q = r_t + gamma * next_q * (1 - d_t)

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if done:
            break

    episode_rewards.append(total_reward)

    # Update target network
    if (ep + 1) % target_update_freq == 0:
        target_net.load_state_dict(q_net.state_dict())

    if (ep + 1) % (n_episodes // 5) == 0:
        avg_r = np.mean(episode_rewards[-50:])
        avg_loss = np.mean(losses[-100:]) if losses else 0
        print(f"  Episode {ep+1:>5}/{n_episodes}  eps={epsilon:.3f}  "
              f"avg_reward={avg_r:.1f}  avg_loss={avg_loss:.4f}")

# --- Results ---
print(f"\\nFinal average reward (last 50 episodes): {np.mean(episode_rewards[-50:]):.1f}")

# Test the learned policy
test_rewards = []
for _ in range(20):
    state = env.reset()
    done = False
    total = 0
    while not done:
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = int(q_net(s_tensor).argmax(dim=1).item())
        state, r, done = env.step(action)
        total += r
    test_rewards.append(total)
print(f"Test performance (20 episodes): {np.mean(test_rewards):.1f} +/- {np.std(test_rewards):.1f}")

# --- Visualisation ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) Reward training curve
ax = axes[0]
window = 30
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax.plot(episode_rewards, alpha=0.3, color='steelblue', linewidth=0.5)
ax.plot(range(window-1, len(episode_rewards)), smoothed, 'b-', linewidth=2, label=f'Smoothed ({window}-ep)')
ax.axhline(y=200, color='r', linestyle='--', alpha=0.5, label='Max reward (200)')
ax.set_xlabel('Episode')
ax.set_ylabel('Total Reward')
ax.set_title('DQN Training Curve', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 2) Loss curve
ax = axes[1]
if losses:
    loss_window = 50
    smoothed_loss = np.convolve(losses, np.ones(loss_window)/loss_window, mode='valid')
    ax.plot(smoothed_loss, 'orange', linewidth=1.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('DQN Loss (MSE TD Error)', fontsize=12)
    ax.grid(True, alpha=0.3)

# 3) Q-value landscape
ax = axes[2]
angles = np.linspace(-0.25, 0.25, 100)
q_left = []
q_right = []
for ang in angles:
    s = torch.tensor([0.0, 0.0, ang, 0.0], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        qvals = q_net(s).cpu().numpy()[0]
    q_left.append(qvals[0])
    q_right.append(qvals[1])
ax.plot(angles, q_left, 'r-', linewidth=2, label='Q(push left)')
ax.plot(angles, q_right, 'b-', linewidth=2, label='Q(push right)')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Pole Angle (radians)')
ax.set_ylabel('Q-value')
ax.set_title('Learned Q-values vs Pole Angle', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output.png', dpi=100)
print("\\nDQN training visualisation saved to output.png")
`;

export const reinforcementLearning: Chapter = {
  title: "Reinforcement Learning",
  slug: "reinforcement-learning",
  pages: [
    {
      title: "MDP & Bellman Equations",
      slug: "mdp-bellman-equations",
      description:
        "Markov Decision Process definition, state/action/reward, Bellman optimality equation, and value iteration on a grid world",
      markdownContent: mdpBellmanMarkdown,
      codeSnippet: mdpBellmanCode,
      codeLanguage: "python",
    },
    {
      title: "Q-Learning",
      slug: "q-learning",
      description:
        "Q-table, epsilon-greedy exploration, Q-learning algorithm on a grid world with convergence visualization",
      markdownContent: qLearningMarkdown,
      codeSnippet: qLearningCode,
      codeLanguage: "python",
    },
    {
      title: "Policy Gradient (REINFORCE)",
      slug: "policy-gradient",
      description:
        "REINFORCE algorithm, policy parameterization, gradient estimation, and CartPole training curve",
      isDeepLearning: true,
      markdownContent: policyGradientMarkdown,
      codeSnippet: policyGradientCode,
      codeLanguage: "python",
    },
    {
      title: "Deep RL (DQN)",
      slug: "deep-rl-dqn",
      description:
        "Deep Q-Network architecture, experience replay, target network, and DQN on CartPole with reward plot",
      isDeepLearning: true,
      markdownContent: deepRLMarkdown,
      codeSnippet: deepRLCode,
      codeLanguage: "python",
    },
  ],
};
