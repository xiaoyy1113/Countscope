"""Focused checks for spatial mapping, ROI semantics and API validation."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import app


class SpatialTests(unittest.TestCase):
    def test_area_overlap_and_rectangular_pool(self):
        # ROI x=[10,20], y=[0,14]: first two x tokens receive 4/10,6/10.
        f=torch.tensor([[[1.,0.],[0.,1.]],[[0.,0.],[0.,0.]]])
        actual=app.descriptors(f,[10],[0],10,14,14)[0,0]
        expected=torch.tensor([.4,.6]);expected/=expected.norm()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))

    def test_query_matches_identical_dense_map(self):
        rng=np.random.default_rng(17)
        f=rng.normal(size=(6,6,12)).astype(np.float32)
        q=app.descriptors(torch.tensor(f,device=app.DEVICE),[28],[28],14,14,14)[0,0]
        score,x,y=app.best_match(f,84,84,14,14,1,q)
        self.assertGreater(score,.99999)
        self.assertEqual((x,y),(28,28))

    def test_multiple_matches_respect_threshold_limit_and_spacing(self):
        f=np.zeros((6,6,4),dtype=np.float32)
        f[...,3]=1
        f[1,1]=[1,0,0,0]
        f[4,4]=[1,0,0,0]
        q=torch.tensor([1.,0.,0.,0.],device=app.DEVICE)
        found=app.multiple_matches(f,84,84,14,14,1,q,threshold=.99,max_candidates=5,nms_distance=14)
        self.assertEqual({(x,y) for _,x,y in found},{(14,14),(56,56)})
        self.assertTrue(all(score>.99 for score,_,_ in found))

    def test_2x_coordinate_mapping(self):
        f=torch.eye(4).reshape(1,4,4)
        q=app.descriptors(f,[7],[0],7,7,7)[0,0]
        self.assertTrue(torch.allclose(q,torch.tensor([0.,1.,0.,0.])))

    def test_windows_fit_edges(self):
        starts=app.axis_starts(512,10,14)
        self.assertEqual(starts[0],0)
        self.assertEqual(starts[-1],502)
        self.assertTrue(all(0<=x<=502 for x in starts))

    def test_local_api_validation(self):
        client=app.app.test_client()
        self.assertEqual(client.get('/api/state',base_url='http://127.0.0.1:8765').status_code,200)
        self.assertEqual(client.post('/api/folder',json={'folder':'no'},base_url='http://127.0.0.1:8765').status_code,403)
        self.assertEqual(client.get('/api/state',base_url='http://evil.example').status_code,403)
        response=client.post('/api/start',json={'generation':app.dataset['generation'],'width':0},
            headers={'X-Scope-Token':app.TOKEN},base_url='http://127.0.0.1:8765')
        self.assertEqual(response.status_code,400)

    def test_annotation_round_trip(self):
        original_database=app.ANNOTATIONS
        try:
            with tempfile.TemporaryDirectory() as directory:
                folder=Path(directory)
                Image.new('RGB',(32,32),'white').save(folder/'x000100_y000200.png')
                app.ANNOTATIONS=folder/'annotations.sqlite3'
                app.open_folder(folder)
                client=app.app.test_client()
                payload=dict(generation=app.dataset['generation'],id=0,x=4,y=5,width=14,height=14,
                             score=.9,review_label='positive')
                response=client.post('/api/annotation',json=payload,headers={'X-Scope-Token':app.TOKEN},
                                     base_url='http://127.0.0.1:8765')
                self.assertEqual(response.status_code,200)
                rows=client.get('/api/annotations',base_url='http://127.0.0.1:8765').get_json()['annotations']
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['review_label'],'positive')
                self.assertEqual((rows[0]['global_center_x'],rows[0]['global_center_y']),(111,212))
                manual=client.post('/api/manual-point',json=dict(generation=app.dataset['generation'],id=0,
                    center_x=20,center_y=20,width=14,height=14),headers={'X-Scope-Token':app.TOKEN},
                    base_url='http://127.0.0.1:8765')
                self.assertEqual(manual.status_code,200)
                reviewed=client.post('/api/patch-review',json=dict(generation=app.dataset['generation'],id=0,
                    review_status='complete'),headers={'X-Scope-Token':app.TOKEN},
                    base_url='http://127.0.0.1:8765')
                self.assertEqual(reviewed.status_code,200)
                exported=client.get('/api/review-export',base_url='http://127.0.0.1:8765').get_json()
                self.assertEqual(len(exported['manual_points']),1)
                self.assertEqual(exported['patches'][0]['gold_count'],2)
        finally:
            app.ANNOTATIONS=original_database


if __name__=='__main__':unittest.main()
