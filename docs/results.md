# Recorded measurements

## TinyStories block size

64 fixed dev windows, 1,845 distinct anchors, seeds 42 and 123, 3,690
predictions. Each prediction uses true history, starts from Gaussian noise,
and uses EMA weights, FP32 evaluation and 32 Euler denoising steps. EM requires
all bytes of the first predicted AE unit to match the reference unit.

| Block | Correct units | Unit EM | Byte-weighted whole-unit EM |
|---|---:|---:|---:|
| 16 | 1,501/3,690 | 40.68% | 35.20% |
| 4 | 1,568/3,690 | 42.49% | 36.78% |
| 1 | 1,620/3,690 | 43.90% | 38.46% |

B1/B4 branch from the 1.6-pass control checkpoint and reset optimizer/noise;
B16 reuses the existing control with its optimizer. All end at 28,989 updates.
The results therefore describe these continuation recipes. B1's difference
versus B16 is +3.22 pp (paired window-bootstrap interval +1.86 to +4.68 pp).

On the same Buzz prefix, 128-unit generation took 8.97/33.26/133.03 seconds
for B16/B4/B1 on an RTX 4060 Laptop, FP16 model and FP32 byte decoding. Smaller
blocks increase denoiser calls per output unit. Free text still has pseudo-words,
reference drift and repetitive endings. [Preserved paired texts](samples.md).

## Multilingual byte reconstruction

The full-corpus continuation consumes 19,044,237,273 natural source bytes across
12 languages and 20,404,922 documents, in addition to the preceding 8 GiB
stable training phase. It is not a from-scratch one-pass result. On an identical
2,048-string natural-span calibration panel, exact byte-and-EOS EM improves
from 2,007/2,048 to 2,046/2,048 (97.9980% to 99.9023%). All 5,017 observed
training codepoints reconstruct individually. A fixed 25-document local check
also conserves the original bytes across 843 reconstructed units.

An out-of-domain handwritten probe containing U+FF0C fails coverage: that
character is absent from the prepared training character set. These measurements
do not establish arbitrary-Unicode reconstruction or language-generation quality.

## Current stage

The 77.6M multilingual BPE/grid8 training was submitted and the BPE arm had
started updating in the latest local record dated 2026-11-06. Capability results
from these arms are not yet part of this release.

Data and recipe identities are in [the evidence record](evidence/results.json).
