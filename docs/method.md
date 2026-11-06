# Model and segmentation

The local byte autoencoder accepts UTF-8 spans up to 32 bytes and emits a
48-dimensional vector. The current AE has 3,015,537 parameters. Natural spans,
character replacements, and character coverage exercises train byte-and-EOS
reconstruction. The encoder and decoder are frozen for prior modeling.

The original TinyStories prior is a 21,845,808-parameter denoiser. A clean
history stream is causal; each noisy block sees earlier clean blocks and its
own noisy positions. Time is sampled independently per block. The x0 target
uses scalar-standardized coordinates, with Xt=(1-t)X+t*epsilon and t=0 clean.
Euler denoising generates blocks from left to right, retaining clean KV state.

Current multilingual experiments compare two ways to choose spans: a sparse
grid8 candidate graph with exact byte-and-EOS feasibility and cost K+10 sum NLL,
and filtered BPE packing with a total-NLL threshold of .03. Grid8 is a sparse
candidate strategy, not an exhaustive search over every possible edge. Both
arms use a 77,587,504-parameter denoiser, 512 width, 16 layers, 8 heads, and
SwiGLU FFN 1408. B1/4/8 are sampled equally per update. The recipe uses a
one-pass raw-byte clock and a WSD schedule with 1% warmup and 10% final decay.

Byte reconstruction EM, next-unit prediction EM, and free-running text quality
measure different behavior. A well-decoded latent can still represent an
incorrect or incoherent continuation.
