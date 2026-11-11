# CLAUDE.md

All instructions for AI agents working in this repository are in AGENTS.md:

@AGENTS.md

Summary, in case the import above is not loaded:
- Read `AGENTS.md`, then `docs/PLAN.md` Sections 1-4 and the section of the person you are helping.
- Ask the user which person (1-5) they are; edit only that person's files.
- Never change the function signatures in `src/`; they are shared between five people.
- Rates are annual continuous decimals; dt = 1/252; seed 42 via `np.random.default_rng`.
- Run `pytest` before finishing. Explain the maths of each step to the user: there is a viva.
