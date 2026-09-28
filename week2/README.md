# Week 2 – Healthcare MDP

## Assignment Idea

For this assignment, I continued with the healthcare domain from Week 1 and reimplemented the Week 2 policy iteration notebook using a simple healthcare MDP.

My main idea was to keep the RL and SQL structure from the lecture notebook, but replace the parts that were specifically designed for GridWorld.

Instead of creating a completely different implementation, I first separated the original notebook into two parts:

- general MDP / policy iteration logic that can be reused
- GridWorld-specific logic that needs to be changed

This helped me keep the main structure of the lecture example while applying it to a different domain.

---

## 1. Designing the Healthcare MDP

I wanted to keep the MDP small because the main goal of the assignment is to understand policy evaluation and policy improvement, not to build a realistic healthcare simulation.

I used four states:

- `Stable`
- `At_Risk`
- `Critical`
- `Recovered`

`Recovered` is the terminal state.

I used three actions:

- `Monitor`
- `Treat`
- `Escalate`

Then I designed one deterministic transition for every state-action pair.

| Current State | Action | Next State | Reward |
|---|---|---|---:|
| Stable | Monitor | Recovered | +10 |
| Stable | Treat | Recovered | +7 |
| Stable | Escalate | Recovered | +4 |
| At_Risk | Monitor | Critical | -8 |
| At_Risk | Treat | Stable | +5 |
| At_Risk | Escalate | Stable | +2 |
| Critical | Monitor | Critical | -10 |
| Critical | Treat | At_Risk | -2 |
| Critical | Escalate | Stable | +4 |
| Recovered | Monitor | Recovered | 0 |
| Recovered | Treat | Recovered | 0 |
| Recovered | Escalate | Recovered | 0 |

The basic idea behind the rewards is simple: moving toward recovery should generally be better, while staying in or moving toward a more serious state should be worse.

These numbers are only designed for this MDP exercise. They are not intended to represent real medical decisions or treatment outcomes.

---

## 2. What I Kept from the Original Notebook

I tried to keep the parts that were not tied to GridWorld.

The main structure I kept includes:

- SQL as the main data model
- SQLAlchemy ORM classes
- `Run`
- `Snapshot`
- `Policy`
- `PolicyProbability`
- `StateValue`
- `ActionValue`
- Bellman backups
- iterative policy evaluation
- exact policy evaluation
- greedy policy improvement
- policy iteration
- convergence checks

I also kept the idea of starting with a uniform policy.

Since there are three actions, the initial policy assigns:

```text
Monitor   = 1/3
Treat     = 1/3
Escalate  = 1/3
```

to each state.

This gives policy improvement a neutral starting point.

---

## 3. What I Changed

The biggest changes were the parts of the original notebook that depended on the GridWorld environment.

### States

The original notebook represented states as cells with row and column coordinates.

That does not make sense for healthcare, so I removed the grid coordinates and used named states instead:

```text
Stable
At_Risk
Critical
Recovered
```

### Actions

Grid movement actions were replaced with:

```text
Monitor
Treat
Escalate
```

### Transitions

The original notebook could generate transitions based on movement inside a grid.

There is no geometry in this environment, so I explicitly defined the transition for each state-action pair.

For example:

```text
At_Risk + Monitor  -> Critical
At_Risk + Treat    -> Stable
Critical + Escalate -> Stable
```

These transitions are then stored in the same SQL transition table used by the rest of the notebook.

### Terminal State

Instead of terminal grid cells, I use:

```text
Recovered
```

as the terminal state.

### Visualization

The original notebook used a heatmap because every state had a location in a grid.

My states do not have spatial positions, so a heatmap would not be meaningful.

I changed the visualization to bar charts where each bar represents one healthcare state. This makes it easier to compare the value of `Stable`, `At_Risk`, `Critical`, and `Recovered` across different snapshots.

### Manhattan Distance

The original notebook also used Manhattan distance to create an independent reference for the GridWorld.

That idea depends completely on grid coordinates and distance to terminal cells.

There is no equivalent distance in my healthcare MDP, so I removed this part instead of trying to force it into the new domain.

---

## 4. Why I Changed Gamma to 0.9

This was one important difference I found while adapting the original notebook.

My healthcare MDP contains this transition:

```text
Critical + Monitor -> Critical
Reward = -10
```

So `Critical` has a self-loop.

If I use:

```text
γ = 1
```

and a policy keeps choosing `Monitor`, the Bellman equation would be:

```text
V(Critical) = -10 + V(Critical)
```

There is no finite solution to this equation.

Because the agent can remain in `Critical` indefinitely, I changed the discount factor to:

```text
γ = 0.9
```

Now the equation becomes:

```text
V(Critical) = -10 + 0.9 × V(Critical)
```

and the value is finite:

```text
V(Critical) = -100
```

So I did not choose `0.9` just as a different hyperparameter. I needed a discount factor below 1 because the healthcare environment I designed contains a possible infinite self-loop.

After changing gamma, I also needed to make sure it was used consistently.

The Bellman calculation becomes:

```text
Q(s, a) = reward + γ × V(next_state)
```

The iterative evaluator already works through these discounted Bellman backups.

I also updated the exact evaluator so that gamma is included when constructing the linear equations. Otherwise the exact and iterative evaluators would actually be solving two different versions of the MDP.

---

## 5. Policy Evaluation

After building the environment, I first evaluated the uniform policy.

I kept both methods from the lecture notebook:

### Exact Evaluation

The exact evaluator creates the Bellman equations and solves them as a linear system.

### Iterative Evaluation

The iterative evaluator starts with initial state values and repeatedly applies Bellman expectation backups until the values converge.

I kept both because it gives me a useful way to check the implementation: the iterative values should approach the exact values.

The results were:

| State | Exact | Iterative |
|---|---:|---:|
| Stable | 7.000000 | 7.000000 |
| At_Risk | 4.158470 | 4.158470 |
| Critical | 0.972678 | 0.972678 |
| Recovered | 0.000000 | 0.000000 |

The iterative version converged after 31 sweeps.

The differences between the two methods were around `10^-9`, so they reached essentially the same result.

I also kept several intermediate snapshots so I could see how the values changed during iteration instead of only looking at the final answer.

---

## 6. Policy Improvement

After evaluating the uniform policy, I used the action values to create a greedy policy.

For each nonterminal state, policy improvement compares the available actions and selects the action with the highest value.

The resulting greedy actions were:

| State | Action |
|---|---|
| Stable | Monitor |
| At_Risk | Treat |
| Critical | Escalate |

For `Recovered`, all three actions lead back to `Recovered` with reward `0`, so they remain tied.

This result also matches the reward and transition structure I designed.

For example, when the patient is `Critical`, `Monitor` keeps the system in `Critical` with a negative reward, while `Escalate` moves it to `Stable`.

---

## 7. Policy Iteration

Finally, I kept the full policy iteration process instead of assuming that one greedy improvement would always be enough.

The process is:

```text
Initial uniform policy
        ↓
Evaluate policy
        ↓
Create greedy policy
        ↓
Evaluate new policy
        ↓
Check whether the policy changes
        ↓
Repeat if necessary
```

For this small MDP, the first improvement changed all three nonterminal states:

```text
Changed states: 3
```

After evaluating the improved policy and improving again:

```text
Changed states: 0
```

The optimality residual was also `0.0`, so the policy was stable and policy iteration converged.

The final policy was:

```text
Stable   -> Monitor
At_Risk  -> Treat
Critical -> Escalate
```

---

## 8. Overall Approach

The main thing I wanted to do in this assignment was not just rename GridWorld variables to healthcare terms.

I tried to identify which parts of the notebook represent general RL concepts and which parts only work because the original environment is a grid.

### Kept

```text
SQL experiment structure
Run / Snapshot
Policies and policy probabilities
State values and action values
Bellman backups
Exact policy evaluation
Iterative policy evaluation
Greedy improvement
Policy iteration
Convergence checking
```

### Reworked

```text
Grid cells        -> Healthcare states
Grid coordinates  -> State names / ordinals
Movement actions  -> Monitor / Treat / Escalate
Grid transitions  -> Healthcare transitions
Terminal cells    -> Recovered state
Heatmaps          -> Bar charts
γ = 1             -> γ = 0.9
```

### Removed

```text
Grid dimensions
Rows and columns
Grid movement logic
Manhattan-distance reference
Grid-specific plotting
Directional arrows
```

The final notebook therefore keeps the policy evaluation and policy iteration structure from the lecture example, but the environment itself is redesigned around the healthcare MDP.