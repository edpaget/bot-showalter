---
user-invocable: true
description: Create a high-level multi-phase plan document in docs/plans/
---

# Skill: Roadmap

Create a high-level, multi-phase plan for a feature or initiative. The output is a markdown document in `docs/plans/` — do NOT start implementing or enter plan mode.

## Workflow

1. **Understand scope**: Read any referenced documents (IMPLEMENTATION.md, docs/active/, docs/proposals/) for context on the topic.
2. **Explore the codebase**: Search relevant source files and tests to understand the current architecture and what exists today.
3. **Write the roadmap**: Create `docs/plans/<topic>.md` with the structure below.
4. **Stop**: Present a summary of the phases. Do not implement anything.

## Roadmap Document Structure

```markdown
# <Topic> Roadmap

## Overview
One paragraph describing the goal and why it matters.

## Current State
Brief summary of what exists today in the codebase.

## Phases

### Phase 1: <Title>
**Goal:** What this phase achieves.
**Scope:**
- Bullet points of what's included
**Key files:** List of files likely to be touched.
**Dependencies:** Any prerequisites or external data needed.

### Phase 2: <Title>
...

## Open Questions
Any unresolved decisions or trade-offs to discuss.
```

## Rules

- Each phase should be a coherent, independently-committable unit of work.
- Order phases so earlier ones unblock later ones.
- Keep phases small enough to complete in a single session where possible.
- Do NOT write implementation-level detail (specific function signatures, test cases, etc.) — that belongs in phase-level planning when it's time to implement.
- If the user specifies a topic, use it. If not, look for the highest-priority unfinished item in IMPLEMENTATION.md or docs/active/.
