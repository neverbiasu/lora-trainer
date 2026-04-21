# Competitor-driven Design and Implementation SOP

Purpose: provide a repeatable, evidence-based workflow for feature design and delivery using competitor references.

## Inputs
- Feature scope and MVP constraints
- Repository references in docs/references
- Design docs in docs/design

## Output artifacts
- Updated design doc with links to concrete references
- Implemented interfaces aligned with references
- Validation notes (tests or manual checks)

## SOP (step-by-step)
1. Define scope and non-goals
   - Write a one-paragraph scope statement and explicit non-goals.
   - Confirm target models, formats, and MVP exclusions.

2. Collect competitor references
   - Identify 3–6 high-quality implementations.
   - Record file paths and line ranges for each relevant method.
   - Capture key behaviors (API shape, data flow, edge cases).

3. Draft initial design
   - Specify interfaces, inputs/outputs, and lifecycle.
   - Provide a minimal end-to-end flow (e.g., validate → train → export).

4. Map references to design decisions
   - For each method, add a reference table:
     - Project, file path, line range, behavior summary.
   - Highlight deviations and rationale.

5. Implement interfaces
   - Implement public methods first.
   - Keep logic close to referenced behavior.
   - Avoid new entry points; extend via adapters.

6. Enrich design with concrete references
   - Add links to exact line ranges used.
   - Document any simplified MVP behavior.

7. Validate behavior
   - Run unit tests or manual checks.
   - Confirm outputs match reference behavior.
   - Log any known gaps and planned follow-ups.

8. Final review checklist
   - English-only in non-doc files.
   - Public APIs have PEP 257 docstrings.
   - Reproducibility metadata recorded.
   - CLI and config remain single-source.

## Prompt template
Use this prompt to execute the SOP:

"""
Task: <feature>
MVP scope: <scope>
Non-goals: <non-goals>
References: <repo list>
Deliverables:
- Update design doc with exact file/line links
- Implement interfaces aligned with references
- Provide a validation summary
"""
