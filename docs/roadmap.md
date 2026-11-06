# Research status

Snapshot date: 2026-11-06.

Completed: the 22M TinyStories prototype; block-size and prediction-target
studies; generated-history and cycle-readout analyses; 12-language byte-AE
training and character-coverage analysis; verified-path and filtered-BPE tools.

In progress: 77.6M multilingual priors comparing grid8 and filtered BPE, using
mixed B1/4/8 training and a matched one-pass natural-byte clock.

Next evidence to report: the new priors' generated text and fixed evaluation,
including quality per byte and inference cost. The longer-term question is how
to align reconstructable representations with the distribution learned by a
continuous generator. Weight release and a standalone small-data reproduction
recipe remain separate release tasks.
