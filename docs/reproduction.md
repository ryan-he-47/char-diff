# Reproduction and asset status

## CPU package check

Install PyTorch and `python -m pip install -e .`, then run
`python scripts/check_release.py`. It uses synthetic latent tensors, no network,
no dataset, no pretrained codec and no training updates. Development tests in
`tests/` may need caches or checkpoints; they are not all source-only tests.

## Training and evaluation

Historical TinyStories entry points are `scripts/prepare_tinystories.py`,
`scripts/train.py`, `scripts/evaluate.py`, `scripts/sample.py`, and
`scripts/teacher_forced_em.py`. Their original recipe files are in `configs/`.
The 22M runs require the matching `assets/ae_best.pt` and tokenizer plus a
prepared latent cache. The fingerprint in a checkpoint must match its codec.

Multilingual preparation and AE training use
`scripts/prepare_multilingual_stories.py` and `scripts/train_utf8_ae.py`.
`scripts/prepare_full_ae.py` materializes the pinned source manifest;
`scripts/prepare_full_grid8.py` builds verified span paths. The new prior uses
`scripts/configure_multilingual_prior.py` and `scripts/train_multilingual.py`.
Read each command's `--help` and provide paths to prepared inputs before running.

The pinned dataset is `Gabrui/multilingual_TinyStories`, revision
`7fb5f327b907318dd95877a366e288f1971b64c1`. Exact document hashes separate
splits; the upstream data has no original-story/translation-group identity,
so exact-content exclusion does not guarantee translation-family disjointness.

The committed recipes preserve recorded research settings. Batch sizes and
storage guards were tuned for an existing environment; users must select their
own paths and capacity before starting a run. For CPU source checks, no such
environment is needed. Model/optimizer hyperparameters were not changed for
this release.

## Weights

Training corpora, cached latents, tokenizer files and checkpoint binaries are
not included. `assets/manifest.json` records identities and release status.
No public download is currently claimed. The release reproduces the source
package check; independent reproduction of the reported trained outputs needs
the exact checkpoints and inputs or a new training run. Checkpoint publication
is tracked as a separate research-release item.
