# Working requirements

Imagine I am a student completing this homework. Everything produced in this repo should read that way — written by a competent student who understands the material, not by a polished research lab or a senior engineer.

## Code

- **Student style, but correct.** Straightforward scripts. No over-engineering: no deep class hierarchies, no config frameworks, no abstract base classes, no premature abstractions.
- **Readable over clever.** Plain variable names. No one-liners that hide what's happening. Loops are fine when they make the logic obvious.
- **Self-contained per task.** Each task should run on its own. Some duplication between tasks is acceptable and expected — do not build a shared "framework" file just to dedupe a few lines.
- **Comments only where the reasoning is non-obvious.** Do not narrate every line. No multi-line docstrings restating what a function literally does.
- **Reproducible.** Set seeds. Save anything (weights, sampled indices, intermediate arrays) needed to regenerate the plots without retraining.
- **One language/stack, picked once.** Do not mix frameworks across tasks.

## Explanations / write-up

- **Concise.** Short paragraphs, direct sentences. No throat-clearing ("In this section, we will discuss…", "It is important to note that…").
- **Interpret, don't just report.** Every plot or table should be followed by a few sentences explaining *why* the result looks the way it does, not just *what* the numbers are.
- **Own reasoning.** All conceptual arguments, derivations, and interpretations must be mine. LLM assistance is for coding help only.
- **No filler.** Cut anything that doesn't either (a) state a result, (b) explain a result, or (c) connect two results.

## Plots

- Label axes, units, and any parameter that varies between plots (sample size, noise level, method, etc.).
- Use consistent colormaps and axis ranges across comparable plots so they can be read side by side.
- Title each figure with what it shows, not just the variable name.

## What "student style but excellent" means

- A reader should believe a strong student wrote this in a week, not that a lab spent a month on it.
- Correctness and clarity matter more than polish or generality.
- If choosing between "obviously right" and "elegantly abstract," choose obviously right.