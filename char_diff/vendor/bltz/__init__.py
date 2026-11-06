"""bltz: vocab-free byte-level LM.

Set Transformer patch encoder -> causal transformer backbone over patch latents
-> conditional query head (h, delta) -> byte, trained with patch-prior-weighted
multi-token prediction. See docs/01-design.md.
"""
