"""Prepare all pinned training documents for the AE, one checksummed Parquet at a time."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from char_diff.full_corpus import CorpusIndex, fetch_source, owned_unlink
from char_diff.storage import StorageGuard
from char_diff.utils import sha256, write_json


def prepare(root, source_path, holdout, guard):
    source = json.loads(source_path.read_text(encoding='utf8'))
    index = CorpusIndex(root, source, holdout.read_bytes())
    try:
        if not index.get('prepare_complete'):
            index.prepare_exclusions()
        parts = index.get('ae_parts') or []
        for i, entry in enumerate(source['train_files']):
            output = root / f'raw-{i:03d}.parquet'
            if i < len(parts):
                if sha256(output) != parts[i]['sha256']:
                    raise ValueError(f'Committed raw corpus changed: {output}')
                continue
            guard.check(int(entry['bytes']) * 2 + 128 * 1024**2)
            path, owned = fetch_source(root, source, entry)
            pending = output.with_suffix('.pending')
            counts = Counter()
            inventory = set()
            index.connection.execute('BEGIN')
            with pq.ParquetFile(path) as original, pq.ParquetWriter(
                    pending, pa.schema([('story', pa.string())]), compression='zstd') as writer:
                for batch in original.iter_batches(columns=['story'], batch_size=2048):
                    stories = []
                    for story in batch.column(0).to_pylist():
                        raw = story.encode('utf8')
                        counts['scanned'] += 1
                        reason = index.accept(raw) if raw else 'empty'
                        counts[reason] += 1
                        if reason == 'accepted':
                            stories.append(story)
                            inventory.update(story)
                            counts['raw_bytes'] += len(raw)
                    if stories:
                        writer.write_table(pa.table({'story': stories}))
            pending.replace(output)
            sizes = []
            with pq.ParquetFile(output) as saved:
                sizes = [saved.metadata.row_group(j).num_rows for j in range(saved.num_row_groups)]
            part = dict(file=output.name, sha256=sha256(output), language=entry['language'],
                        source=entry['path'], counts=dict(counts), groups=sizes,
                        characters=''.join(sorted(inventory)))
            assert sum(sizes) == counts['accepted']
            parts.append(part)
            index.set('ae_parts', parts)
            index.connection.commit()
            if owned:
                owned_unlink(path, root/'input_cache')
            print(json.dumps({'file':i+1,'total_files':len(source['train_files']),**counts}),flush=True)
        scanned = Counter()
        for part in parts:
            scanned[part['language']] += part['counts']['scanned']
        if dict(scanned) != source['expected_train_documents']:
            raise ValueError('Full-source document counts differ from pinned inventory')
        write_json(root/'manifest.json',dict(format='full-raw-ae-v1',status='complete',
                   source_sha256=sha256(source_path),holdout_sha256=sha256(holdout),parts=parts,
                   raw_bytes=sum(p['counts']['raw_bytes'] for p in parts),
                   documents=sum(p['counts']['accepted'] for p in parts)))
    finally:
        index.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for key in ('out','source','holdout','project'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    prepare(args.out,args.source,args.holdout,StorageGuard(args.project,32,1,True))
