import random,unittest
from scripts.run_x369_geometry_training_gpu import ARMS,RELATIONS,groups,training_batch
from scripts.run_x369_phase_trajectory_gpu import schedule
from scripts.audit_x369_ordered_signed_task import endpoint,TRIADS

class GeometryTrainingTest(unittest.TestCase):
    def test_matched_draws(self):
        train=groups([dict(word=w,id=i) for i,w in enumerate(('a','b','ab','j','n'))])
        streams=[]
        for arm in ARMS:
            rng=random.Random(41)
            streams.append([[r['id'] for r in training_batch(train,rng,128)] for _ in range(20)])
        self.assertEqual(streams[0],streams[1])
        self.assertEqual(streams[0],streams[2])
        self.assertTrue(all(len(set(len(train_word) for train_word in [ ('a','b','ab','j','n')[i] for i in draw]))==1 for draw in streams[0]))

    def test_schedule(self):
        self.assertEqual(schedule(2000,250),[0,1,250,500,750,1000,1250,1500,1750,2000])

    def test_independent_relations(self):
        for l,r in RELATIONS:
            for d in TRIADS:
                self.assertEqual(endpoint(l,(1,2,1,1),(1,0,0),d),endpoint(r,(1,2,1,1),(1,0,0),d))

if __name__=='__main__':unittest.main()
