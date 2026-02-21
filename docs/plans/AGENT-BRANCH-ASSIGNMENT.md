# Agent branch assignment – Temporal Topic Analysis

Use **one branch per stream of work**. Check out the branch for the agent you’re using, then do only that branch’s work.

| Branch | Purpose | Primary file(s) | Agent |
|--------|--------|------------------|--------|
| **topic-analysis/design** | Own and edit the **design** doc | `docs/plans/2026-02-21-temporal-topic-analysis-design.md` | Claude (or design-focused agent) |
| **topic-analysis/plan** | Own and edit the **implementation plan** | `docs/plans/2026-02-21-temporal-topic-analysis.md` | Claude (or plan-focused agent) |
| **topic-analysis/execute** | **Execute** the analysis: implement and run the pipeline from the design | `config.yaml`, `scripts/*.py`, `requirements.txt`, `output/` | **Cursor** |

## How to use

1. **Design work**  
   - Check out: `git checkout topic-analysis/design`  
   - Edit only: `docs/plans/2026-02-21-temporal-topic-analysis-design.md`  
   - Use the design-focused agent (e.g. Claude) on this branch.

2. **Plan work**  
   - Check out: `git checkout topic-analysis/plan`  
   - Edit only: `docs/plans/2026-02-21-temporal-topic-analysis.md`  
   - Use the plan-focused agent (e.g. Claude) on this branch.

3. **Execution (Cursor)**  
   - Check out: `git checkout topic-analysis/execute`  
   - Implement and run the pipeline (scripts, config, outputs) per the design.  
   - Use **Cursor** on this branch to execute the analysis.

Merge branches back to `main` when each stream is ready (e.g. design approved → merge design; plan approved → merge plan; pipeline working → merge execute).
