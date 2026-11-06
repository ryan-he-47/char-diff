"""Targeted checks for shared byte scoring, histories, and the two new batch adapters."""
import json
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from compare_byte_em import (BASELINE, MODELS, ROOT, LatentCache, Denoiser, generate,
                             vocabulary_bytes, byte_score, aggregate, diffusion_batch,
                             diffusion_predict, ar_predict, write_json)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    tick = time.perf_counter()
    folder = ROOT / 'runs/cross_model_byte_em_20261102'
    cases = json.loads((folder / 'anchors.json').read_text(encoding='utf-8'))['cases']
    cache = LatentCache(ROOT / 'data/tinystories', 'dev')
    tokenizer = AutoTokenizer.from_pretrained(BASELINE, local_files_only=True)
    byte_vocab = vocabulary_bytes(tokenizer)
    rows = [dict(seed=42, **byte_score(b'cat ', b'cat tail', b'cat tail', b'cat ')),
            dict(seed=42, **byte_score(b'apple ', b'apple hi', b'apple!', b'apple!'))]
    metrics = aggregate(rows)
    assert metrics['span_em'] == .5 and metrics['byte_weighted_em'] == .4 and metrics['em4'] == 1
    offsets = np.fromfile(cache.root / 'unit_offsets.u64', dtype='<u8')
    blob = (cache.root / 'units.bin').read_bytes()
    for case in cases:
        history = b''.join(blob[int(offsets[i]):int(offsets[i + 1])] for i in case['history_ids'])
        assert history == bytes.fromhex(case['history_hex'])
        assert history == b''.join(byte_vocab[i] for i in case['input_ids'])
        assert len(case['input_ids']) <= 192
    full = [c for c in cases if c['block_length'] == 16]
    chosen = [min(full, key=lambda c: len(c['history_ids'])), max(full, key=lambda c: len(c['history_ids'])),
              next(c for c in cases if c['block_length'] < 16)]
    packed = diffusion_batch(cache, chosen)
    noise = torch.zeros(len(chosen), 16, 48, device='cuda')
    for i, case in enumerate(chosen):
        rng = torch.Generator(device='cuda').manual_seed(42 + i)
        noise[i, :case['block_length']] = torch.randn(case['block_length'], 48, device='cuda', generator=rng)
    differences = {}
    for name in ['x0', 'gmm_early']:
        state = torch.load(ROOT / MODELS[name], map_location='cpu', weights_only=False)
        model = Denoiser(state['config']['model']).cuda().eval(); model.load_state_dict(state['ema'])
        for sampler in (['euler'] if name == 'x0' else ['euler', 'gm_ode']):
            actual = diffusion_predict(model, packed, noise.clone(), sampler, steps=2)
            diffs = []
            for i, case in enumerate(chosen):
                length = case['block_length']
                reference = generate(model, cache.vectors[case['history_ids']], length,
                                     steps=2, seed=42 + i, sampler=sampler)
                torch.testing.assert_close(actual[i, :length], reference, atol=3e-4, rtol=3e-4)
                diffs.append(float((actual[i, :length] - reference).abs().max()))
            differences[name + '_' + sampler] = diffs
        del model, state
        torch.cuda.empty_cache()
    # Variable-length left-padded prompts and a trailing space must produce the
    # same greedy bytes as running the established AR forward one case at a time.
    ar_cases = [min((c for c in cases if c['input_ids'] and bytes.fromhex(c['history_hex']).endswith(b' ')),
                    key=lambda c: len(c['input_ids'])),
                max(cases, key=lambda c: len(c['input_ids']))]
    model = AutoModelForCausalLM.from_pretrained(BASELINE, local_files_only=True, dtype=torch.float32).cuda().eval()
    batched, _, healing = ar_predict(model, tokenizer, byte_vocab, ar_cases, 42, greedy=True)
    singles = [ar_predict(model, tokenizer, byte_vocab, [c], 42, greedy=True)[0][0] for c in ar_cases]
    assert batched == singles
    result = {'checks': ['byte-weighted arithmetic', 'all shared histories and byte alphabet',
                         'padded diffusion adapter vs generate, x0/Euler/GMM-ODE',
                         'AR padded batch vs single, known-whitespace healing'],
              'history_cases': len(cases), 'diffusion_max_abs_differences': differences,
              'ar_case_ids': [c['id'] for c in ar_cases], 'ar_healed_bytes': healing,
              'ar_generated_hex': [v.hex() for v in batched], 'seconds': time.perf_counter() - tick}
    write_json(folder / 'targeted-checks.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
