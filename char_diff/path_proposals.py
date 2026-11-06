"""UTF-8-safe coarse candidate boundaries, retaining exact original bytes."""
from __future__ import annotations
from bisect import bisect_left,bisect_right
from .ae_paths import boundary_offsets


def proposed_boundaries(raw, mode, *, tokenizer=None, max_bytes=32):
    full=boundary_offsets(raw)
    if mode.startswith('grid'):
        stride=int(mode[4:])
        if not 1<=stride<=max_bytes-4:raise ValueError('Invalid grid stride')
        chosen={0,len(raw)}|{full[bisect_left(full,b)] for b in range(stride,len(raw),stride)}
    elif mode in ('bpe','bpe8'):
        if tokenizer is None:raise ValueError('BPE proposals need a tokenizer')
        text=raw.decode('utf8');encoded=tokenizer.encode(text,add_special_tokens=False)
        chosen={0,len(raw)}
        for start,end in encoded.offsets:
            if not 0<=start<=end<len(full):raise ValueError('Tokenizer offset outside original codepoints')
            chosen.update((full[start],full[end]))
        if mode=='bpe8':
            bpe=sorted(chosen)
            chosen={0,len(raw)}|{bpe[bisect_left(bpe,b)] for b in range(8,len(raw),8)}
    else:raise ValueError('Unknown proposal mode')
    # Long tokens receive interior UTF-8 nodes, so every adjacent span fits AE.
    previous=0
    for end in sorted(chosen)[1:]:
        while end-previous>max_bytes:
            point=full[bisect_right(full,previous+max_bytes)-1]
            if point<=previous:raise ValueError('Codepoint exceeds max_bytes')
            chosen.add(point);previous=point
        previous=end
    return tuple(sorted(chosen))
