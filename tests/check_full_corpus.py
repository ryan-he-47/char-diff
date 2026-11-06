"""Targeted heldout/dedup, Parquet cursor and commit-gap recovery fixture."""
from pathlib import Path
import hashlib,json,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pyarrow.parquet as pq
from char_diff.full_corpus import CorpusIndex,segment_corpus
from char_diff.ae_paths import CandidateScore
from char_diff.segmented_cache import read_segmented_shard
from char_diff.coarse_process import ProcessCoarseSolver


if __name__=='__main__':
    with tempfile.TemporaryDirectory(dir='runs',prefix='full_corpus_check_') as folder:
        root=Path(folder)
        texts=['a story 苹果 '*5,'second tale 🙂 '*5,'held out','old eval','a story 苹果 '*5,'', '尾巴'*22]
        for name,rows in [('train',texts),('train2',['尾巴'*22,'last story '*8]),('test',['held out'])]:
            pq.write_table(pa.table({'story':rows}),root/f'{name}.parquet',row_group_size=2)
        source={'repo':'fixture','revision':'fixed','languages':['english'],
                'train_files':[{'language':'english','fixture_path':str(root/f'{s}.parquet')} for s in ('train','train2')],
                'test_files':[{'language':'english','fixture_path':str(root/'test.parquet')}],
                'expected_train_documents':{'english':9}}
        holdout=hashlib.sha256(b'old eval').digest();provenance={'languages':['english'],'scoring':{'fixture':1}}
        def scorer(units):return [CandidateScore(True,.01) for _ in units]
        reference=CorpusIndex(root/'reference',source,holdout);assert reference.prepare_exclusions()==2
        full=segment_corpus(reference,scorer,provenance,shard_bytes=100,group_documents=1,cache_capacity=100)
        reference.close()
        interrupted=CorpusIndex(root/'resumed',source,holdout);interrupted.prepare_exclusions()
        class InjectedStop(Exception):pass
        def crash():raise InjectedStop('after shard publication, before SQLite cursor commit')
        try:segment_corpus(interrupted,scorer,provenance,shard_bytes=100,group_documents=1,
                           cache_capacity=100,after_publish=crash)
        except InjectedStop:pass
        else:raise AssertionError('Missing injected interruption')
        interrupted.close()
        restored=CorpusIndex(root/'resumed',source,holdout)
        upgraded={**provenance,'scoring':{'fixture':1,'batch_size':32,'cuda_streams':2},'cpu_processes':2}
        pool=ProcessCoarseSolver(2)
        try:
            resumed=segment_corpus(restored,scorer,upgraded,shard_bytes=100,group_documents=1,cache_capacity=100,
                                  process_solver=pool,upgrade_runtime=True)
        finally:pool.close()
        assert resumed==full and full['complete'];restored.close()
        invalid=CorpusIndex(root/'resumed',source,holdout)
        try:segment_corpus(invalid,scorer,{**upgraded,'quality_weight':99},upgrade_runtime=True)
        except ValueError:pass
        else:raise AssertionError('Runtime upgrade changed the path objective')
        invalid.close()
        actual=[];coordinates=[]
        for path in sorted((root/'resumed/segments').glob('shard-*')):
            shard=read_segmented_shard(path)
            for i in range(shard['manifest']['documents']):
                a,b=map(int,shard['byte_offsets'][i:i+2]);actual.append(shard['text'][a:b].decode())
                coordinates.append((int(shard['source_files'][i]),int(shard['source_rows'][i])))
        assert actual==[texts[0],texts[1],texts[6],'last story '*8]
        assert coordinates==[(0,0),(0,1),(0,6),(1,1)]
        assert full['counts']['english']['duplicate']==2 and full['counts']['english']['excluded']==2
        print(json.dumps({'recovery':'identical full state','accepted_documents':len(actual),
                          'heldout_and_dedup':True,'source_coordinates':coordinates}))
