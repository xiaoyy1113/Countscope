import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from color_filter import summarize_rgb, region_color


def red_count(rgb, threshold=80):
    return sum(summarize_rgb(rgb)['hist'][threshold:])


class ColorTests(unittest.TestCase):
    def test_red_pink_and_magenta_survive(self):
        for color in ((190, 40, 50), (245, 180, 200), (200, 50, 170)):
            self.assertEqual(red_count([[color]]), 1)

    def test_brown_blue_and_neutral_are_excluded(self):
        for color in ((140, 90, 40), (180, 110, 70), (70, 60, 180), (250, 248, 248), (80, 80, 80)):
            self.assertEqual(red_count([[color]]), 0)

    def test_threshold_can_recover_orange_brown(self):
        sample=[[(180, 100, 60)]]
        self.assertEqual(red_count(sample, 80), 0)
        self.assertEqual(red_count(sample, 60), 1)

    def test_small_dot_on_white_keeps_exact_area(self):
        rgb=np.full((10,10,3), 255, dtype=np.uint8)
        rgb[3:5,4:6]=[200,40,60]
        stats=summarize_rgb(rgb)
        self.assertEqual(stats['pixels'],100)
        self.assertEqual(sum(stats['hist'][80:]),4)

    def test_only_requested_original_region_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sample.png'
            rgb=np.full((20,20,3), [200,30,30], dtype=np.uint8)
            rgb[5:15,5:15]=[140,90,40]
            Image.fromarray(rgb).save(path)
            stats=region_color(path,5,5,10,10)
            self.assertEqual(stats['pixels'],100)
            self.assertEqual(sum(stats['hist'][80:]),0)
            with self.assertRaises(ValueError):region_color(path,19,19,10,10)


if __name__=='__main__':unittest.main()
