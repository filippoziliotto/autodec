# docs

## Purpose

`gendec/docs/` stores the human-facing design, implementation, and model-reference documents for Phase 1.

## Maintenance Contract

If implementation, architecture, or runtime behavior changes in a way that invalidates these documents, both the affected document and this index must be updated.

## Files

### `IMPLEMENTATION.md`

- Implementation-oriented design specification for Phase 1.
- Describes the intended objective, teacher data pipeline, tokenization, model family, training objective, and runtime structure.
- Functions as the implementation blueprint for the scaffold prior.

### `MODEL.md`

- Detailed implementation-reference document for the current codebase.
- Explains token layout, dataset schema, model submodules, tensor shapes, loss construction, backpropagation paths, training-time validation and sampling diagnostics, sampling behavior, and evaluation behavior.
- Now also documents the optional eval-only frozen AutoDec zero-residual coarse-decode path.
- This is the most detailed “how the code works today” reference in the folder.

### `PHASE1.md`

- Project concept note for Gen-Phase 1.
- Explains why the scaffold prior exists, what the phase includes, and what it explicitly leaves for later phases.
- Frames the scope as scaffold generation, not dense reconstruction.
