import contextlib,io,json,tempfile,unittest
from pathlib import Path
from scripts.x369_campaign_progress import CampaignProgress

class CampaignProgressTest(unittest.TestCase):
    def test_resume_and_phase_estimates(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            now=[0.]
            p=CampaignProgress([('fresh',41),('warm',41)],100,3,Path(d),10,clock=lambda:now[0])
            k=('fresh',41);w=('warm',41)
            p.seed(k,50);p.begin(k,'train')
            now[0]=10.;p.update(k,'train',75,force=True)
            def status():return json.loads((Path(d)/'progress.json').read_text())
            self.assertEqual(status()['training_percent'],37.5)
            self.assertEqual(status()['arm_training_eta_seconds'],10.)
            self.assertIsNone(status()['campaign_eta_seconds'])
            now[0]=20.;p.update(k,'train',100,force=True)
            p.begin(k,'audit');now[0]=24.;p.update(k,'audit',1,force=True)
            self.assertEqual(status()['campaign_eta_seconds'],68.)
            now[0]=36.;p.update(k,'audit',4,force=True);p.complete(k)
            self.assertEqual(status()['arms_completed'],1)
            p.begin(w,'train');now[0]=76.;p.update(w,'train',100,force=True)
            p.begin(w,'audit');now[0]=92.;p.update(w,'audit',4,force=True);p.complete(w)
            self.assertEqual(status()['campaign_eta_seconds'],0.)
            self.assertEqual(status()['training_percent'],100.)
            self.assertEqual(status()['arms_completed'],2)
if __name__=='__main__':unittest.main()
