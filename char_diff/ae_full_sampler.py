"""Reproducible UTF-8 windows covering every admitted story in each full-data pass."""
import json
from pathlib import Path
import random

import pyarrow.parquet as pq

from .ae_windows import UTF8WindowSampler, WindowSample
from .utils import sha256


class FullCorpusSampler(UTF8WindowSampler):
    def __init__(self, manifest, *, seed, max_bytes, lanes=16, **options):
        manifest = Path(manifest)
        spec = json.loads(manifest.read_text(encoding="utf8"))
        if spec["format"] != "full-raw-ae-v1" or spec["status"] != "complete":
            raise ValueError("Expected a complete full-data manifest")
        self.spec = spec
        self.directory = manifest.parent
        self.groups = []
        self.checksums = {p["file"]:p["sha256"] for p in spec["parts"]}
        self.verified_files = set()
        inventory = {}
        for part in spec["parts"]:
            language = part["language"]
            inventory.setdefault(language, set()).update(part["characters"])
            self.groups.extend((part["file"], i, size, language)
                               for i, size in enumerate(part["groups"]) if size)
        # Reuse the existing character replacement and accounting machinery.
        inventory_records = [dict(split="train", language=lang, id=lang, text="".join(sorted(chars)))
                             for lang, chars in inventory.items()]
        super().__init__(inventory_records, seed=seed, max_bytes=max_bytes, **options)
        self.source_sha256 = sha256(manifest)
        self.seed = seed
        self.lane_count = min(lanes,len(self.groups))
        if not self.lane_count:
            raise ValueError("Empty full training source")
        self.epoch = 0
        self._start_epoch()

    @property
    def protocol(self):
        return {**super().protocol, "version":3, "sampling":"shuffled_row_groups_round_robin/document_tiling",
                "lanes":self.lane_count, "seed":self.seed, "single_character_language":"uniform"}

    def _order(self):
        order = list(range(len(self.groups)))
        random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def _start_epoch(self):
        self.order = self._order()
        self.next_group = self.lane_count
        self.lanes = [[idx,0,0] for idx in self.order[:self.lane_count]]
        self.lane_turn = 0
        self.loaded = {}

    @property
    def completed_epochs(self):
        return self.epoch + int(all(lane is None for lane in self.lanes))

    def natural(self):
        if all(lane is None for lane in self.lanes):
            self.epoch += 1
            self._start_epoch()
        while self.lanes[self.lane_turn] is None:
            self.lane_turn = (self.lane_turn+1) % self.lane_count
        lane_id = self.lane_turn
        self.lane_turn = (lane_id+1) % self.lane_count
        index, row, pos = self.lanes[lane_id]
        name, group, size, language = self.groups[index]
        if index not in self.loaded:
            if name not in self.verified_files:
                if sha256(self.directory/name) != self.checksums[name]:
                    raise ValueError(f"Full AE raw file checksum mismatch: {name}")
                self.verified_files.add(name)
            with pq.ParquetFile(self.directory/name) as reader:
                self.loaded[index] = reader.read_row_group(group,columns=["story"]).column(0).to_pylist()
            if len(self.loaded[index]) != size:
                raise ValueError("Row group size mismatch")
        story = self.loaded[index][row]
        # UTF-8 continuation bytes distinguish codepoint boundaries directly.
        prefix = story[pos:pos+self.max_bytes].encode("utf8")[:self.max_bytes].decode("utf8",errors="ignore")
        count = self.rng.randint(1,len(prefix))
        original = prefix[:count]
        self.lanes[lane_id][2] += count
        if pos+count == len(story):
            self.lanes[lane_id][1:] = [row+1,0]
            if row+1 == size:
                del self.loaded[index]
                self.lanes[lane_id] = None
                if self.next_group<len(self.order):
                    self.lanes[lane_id] = [self.order[self.next_group],0,0]
                    self.next_group += 1
        augmented = self.rng.random()<self.augment_probability
        result,subs,rejected = self._replace(original) if augmented else (original,0,0)
        return WindowSample(result.encode(),original.encode(),language,None,None,None,augmented,subs,rejected)

    def sample(self):
        if self.rng.random()<self.single_character_probability:
            lang=self.rng.choice(self.languages)
            chars=self._language_characters[lang]
            cursor=self._character_cursors[lang]
            raw=chars[cursor%len(chars)].encode()
            self._character_cursors[lang]+=1
            sample=WindowSample(raw,raw,lang,None,None,None,practice=True)
        else:
            sample=self.natural()
            self.stats["natural_source_bytes"] = self.stats.get("natural_source_bytes",0)+len(sample.original)
        self.stats["completed_epochs"]=self.completed_epochs
        return self._record_sample(sample)

    def state_dict(self):
        return {**super().state_dict(),"epoch":self.epoch,"next_group":self.next_group,
                "lanes":[list(x) if x is not None else None for x in self.lanes],"turn":self.lane_turn}

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.epoch=state["epoch"]
        self.order=self._order()
        self.next_group=state["next_group"]
        self.lanes=[list(x) if x is not None else None for x in state["lanes"]]
        self.lane_turn=state["turn"]
        self.loaded.clear()
