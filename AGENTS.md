## Environment

Set up a test environment with `pip install -e ".[dev]"`.

## Code Design Principles

Prioritize elegant, minimal implementations over feature-complete first drafts.

**Parsimony**: Prefer fewer concepts, fewer lines, fewer moving parts. Before adding an abstraction or indirection layer, justify why the simpler approach doesn't work.

**Extensibility through simplicity**: Don't add plugin architectures, factory patterns, or configuration layers until there's a concrete second use case. Readable code is extensible code.

**Aesthetic quality**: Treat code as prose. Names should reveal intent, related logic should be visually grouped, and structure should mirror the problem. If it's awkward to read, it's awkward to extend.

**Concise comments**: Don't add overly verbose multi-line comments; be concise. Especially avoid comments that merely describe responses to historical pivots.

Flag design tensions explicitly when proposing implementations.

## Cross-fold validation

We use cross-fold validation, training multiple models on different data splits. Scripts ending in \_folds.py are wrappers around base scripts (e.g. hound_snp_folds.py wraps hound_snp.py) — CLI changes to a base script must be propagated to its \_folds.py counterpart. Inference scripts ensemble predictions across folds.

## Gotchas

**Reverse complement**: Gene regulation is strand-agnostic. We use reverse complement as data augmentation: half of training steps reverse complement the DNA and reverse the tracks. Some tracks are stranded (5'/3') and paired via `strand_pair`—these pairs must be swapped when reverse complementing. At inference, generally ensemble forward predictions with reversed reverse-complement predictions.
