---
name: roadmap
description: Create a multi-phase roadmap document for a feature or refactoring effort. Use when the user asks to "create a plan", "write a plan", or "plan out" a feature.
argument-hint: <topic>
---

# Roadmap Skill

Create a high-level multi-phase roadmap document at `docs/plans/<topic>.md`.

## Behavior

1. If `$ARGUMENTS` is provided, use it as the topic name (kebab-case for the filename).
2. Explore the codebase to understand the current state relevant to the topic.
3. Design phases that break the work into independently implementable increments.
4. Write the roadmap to `docs/plans/<topic>.md`.

## Roadmap format

Follow this structure (based on existing roadmaps in `docs/plans/`):

```markdown
# <Title> Roadmap

<1-2 paragraph summary of what this roadmap achieves and why.>

## Status

| Phase | Status |
|-------|--------|
| 1 — <short title> | not started |
| 2 — <short title> | not started |

## Phase 1: <Short title>

<Brief description of what this phase does.>

### Context

<Why this phase exists — what's the current state and what's wrong with it.>

### Steps

1. <Concrete implementation step>
2. <Concrete implementation step>
3. ...

### Acceptance criteria

- <Testable criterion>
- <Testable criterion>

## Phase 2: <Short title>

...

## Ordering

<Describe dependencies between phases and suggested priority.>
```

## Rules

- Each phase must be independently implementable and mergeable.
- Steps should be concrete enough to guide implementation but not so detailed that they prescribe every line of code.
- Include acceptance criteria for every phase.
- Include a Status table after the summary, listing every phase with its current status (`not started`, `in progress`, `done (<date>)`, or `blocked on <reason>`). For ML/quantitative phases, record go/no-go outcomes in the status when completed.
- End with an ordering section noting dependencies and suggested priority.
- Do NOT start implementing code. This is a planning artifact only.
