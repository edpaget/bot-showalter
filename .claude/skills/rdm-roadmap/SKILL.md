---
name: rdm-roadmap
description: Create an rdm roadmap with phases for a topic
allowed-tools:
  - Read
  - Glob
  - Grep
  - mcp__rdm__rdm_roadmap_create
  - mcp__rdm__rdm_roadmap_show
  - mcp__rdm__rdm_phase_create
---

Create an rdm roadmap with phases for the topic described in `$ARGUMENTS`.

## Principles

Read `docs/principles.md` before starting. It contains project conventions that should guide your work.
## Steps

1. **Explore the codebase** to understand the current state relevant to `$ARGUMENTS`. Read key files, search for related code, and build context.
2. **Design phases** that break the work into independently deliverable increments. Each phase should produce a working, testable result.
3. **Create the roadmap** using the `mcp__rdm__rdm_roadmap_create` tool with `project: "fbm"`, a descriptive `slug`, `title`, and `body` (summary).
4. **Create each phase** using the `mcp__rdm__rdm_phase_create` tool with `project: "fbm"`, `roadmap: <slug>`, a phase `slug`, `title`, `number`, and a `body` containing Context, Steps, and Acceptance Criteria in Markdown:
   ```markdown
   ## Context
   Why this phase exists and what it builds on.

   ## Steps
   1. First step
   2. Second step

   ## Acceptance Criteria
   - [ ] Criterion one
   - [ ] Criterion two
   ```
5. **Verify** the roadmap looks correct using the `mcp__rdm__rdm_roadmap_show` tool with `project: "fbm"` and `roadmap: <slug>`.

## Guidelines

- Aim for 2-6 phases per roadmap
- Each phase should be independently deliverable and testable
- Include Context, Steps, and Acceptance Criteria in every phase body
- Order phases so each builds on the previous one
- Use clear, descriptive slugs (e.g., `add-caching`, `migrate-auth`)
