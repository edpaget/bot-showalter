---
name: review
description: Review recent changes against the architectural principles in docs/principles.md. Flags semantic violations that static analysis cannot catch. Use when you want to check a commit, diff, or set of changes for principle violations.
argument-hint: [<diff-range>]
---

# Architecture Review Skill

Review code changes against the project's architectural principles (`docs/principles.md`). This catches semantic violations that linting and architecture tests cannot — model-specific code, business logic in interaction layers, complex protocols, and similar design-intent issues.

## Behavior

1. **Determine the diff to review.**
   - If `$ARGUMENTS` specifies a diff range (e.g., `HEAD~3`, `main..HEAD`, a commit SHA), use `git diff $ARGUMENTS`.
   - If no arguments, default to the last commit: `git diff HEAD~1`.
   - If the repo has no commits yet or HEAD~1 fails, use `git diff --cached` (staged changes).

2. **Read the principles.** Read `docs/principles.md` in full.

3. **Analyze the diff against every principle.** For each changed file and hunk, check whether the change violates any of the 11 principles. Focus on the semantic violations that static analysis misses:

   | Principle | What to check for |
   |-----------|-------------------|
   | 1 — Module Decoupling | Service importing a concrete repo class instead of a protocol. Non-composition-root code importing from multiple unrelated packages. |
   | 2 — All Code Must Be Tested | New public functions/classes with no corresponding test. Test files that mock the database instead of using in-memory SQLite. |
   | 3 — Secrets via Env Vars | Hardcoded API keys, tokens, passwords, or credentials anywhere in source. URLs with embedded credentials. |
   | 4 — Model Generality | Code that names a specific model (e.g., `if model == "marcel"`) outside that model's own package. CLI commands that hardcode a model name instead of accepting it as an argument. |
   | 5 — Interaction Mode Generality | Business logic inside CLI commands, agent tools, or HTTP handlers instead of in services. A service importing from `cli/`, `agent/`, `tools/`, or HTTP code. |
   | 6 — Module Public API via Re-exports | New public symbols added to a package without updating `__init__.py` and `__all__`. Consumers importing from a submodule when a re-export exists. |
   | 7 — Domain Models Are Pure Data | Domain dataclass with custom methods (beyond `__post_init__`). Domain model importing infrastructure (repos, HTTP clients). Non-frozen dataclass in `domain/`. |
   | 8 — Unidirectional Layer Dependencies | Lower layer importing a higher layer (e.g., repo importing a service, domain importing a repo). |
   | 9 — Raw Parameterized SQL | String interpolation of values into SQL (f-strings with user data). Missing `?` placeholders. |
   | 10 — Result Types for Recoverable Errors | Raising exceptions for expected, recoverable failures instead of returning `Result[T, E]`. Bare `except` clauses that swallow errors. |
   | 11 — Simple Functional Interfaces | Protocol with many methods when a `Callable` or single-method protocol would suffice. New multi-method protocol where the methods aren't cohesive. |

4. **Report findings.** For each violation found, output:

   ```
   ### [SEVERITY] Principle N — <principle name>
   **File:** `path/to/file.py`, lines X–Y
   **Description:** <what the violation is and why it matters>
   **Suggestion:** <how to fix it>
   ```

   Severity levels:
   - **ERROR** — Clear violation that should be fixed before merging.
   - **WARNING** — Potential concern worth reviewing; may be acceptable with justification.

5. **Summary.** End with a summary line:
   - If violations found: `**Summary:** N error(s), M warning(s) found across K file(s).`
   - If no violations: `**No architectural violations detected.** The changes conform to all 11 principles.`

## Rules

- **Be adversarial but fair.** Flag real violations, not style preferences. Every finding must reference a specific principle number and explain why the code violates it.
- **Ignore test files for most principles.** Tests legitimately import concrete classes, use hardcoded values, etc. Only flag tests for principle 2 violations (e.g., mocking the database instead of using in-memory SQLite).
- **Ignore composition roots.** Files in `cli/`, `agent/`, `tools/`, and factory modules are allowed to import from any layer — that's their job.
- **Ignore `scripts/` directory.** Scripts are standalone utilities, not part of the layered architecture.
- **Don't flag existing code.** Only flag violations in the *changed* lines (lines with `+` prefix in the diff). If existing code around the change has issues, mention it as context but don't count it as a finding.
- **Read surrounding context when needed.** If a diff hunk is ambiguous, use the Read tool to check the full file for context (e.g., to verify whether an import is used at runtime or only in `TYPE_CHECKING`).
- **Be specific.** Always include the file path, line numbers, and a concrete description. Never give vague findings like "this might violate principle 1."
