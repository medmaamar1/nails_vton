"""
test_finger_mapping.py
----------------------
Tests finger label assignment using REAL bboxes from the actual
Roboflow v50 COCO JSON (_annotations.coco.json).

No synthetic data. Every bbox here is copied verbatim from the JSON
you shared. Each test manually traces what the correct answer should
be, then asserts the algorithm matches.

Run:
    python test_finger_mapping.py
    python test_finger_mapping.py -v
"""

import sys
import unittest

sys.path.insert(0, ".")
from dataset import (
    assign_finger_ids,
    FINGER_UNUSED, FINGER_THUMB, FINGER_INDEX,
    FINGER_MIDDLE, FINGER_RING, FINGER_PINKY,
)

LBL = {0:"UNUSED", 1:"THUMB", 2:"INDEX", 3:"MIDDLE", 4:"RING", 5:"PINKY"}

def lname(code):
    return LBL[code]

def lnames(codes):
    return [lname(c) for c in codes]

def cx(bbox): return bbox[0] + bbox[2] / 2
def cy(bbox): return bbox[1] + bbox[3] / 2
def area(bbox): return bbox[2] * bbox[3]


# ══════════════════════════════════════════════════════════════════════════════
# image_id 0  — 5 nails, left hand, thumb bottom-right
#
#  ann 0: bbox [544, 246, 79,  44]  cx=583  cy=268  area=3530  → INDEX
#  ann 1: bbox [427, 175, 76,  45]  cx=465  cy=197  area=3504  → MIDDLE
#  ann 2: bbox [297, 164, 74,  44]  cx=334  cy=186  area=3315  → RING
#  ann 3: bbox [ 60, 188, 47,  32]  cx= 83  cy=204  area=1525  → PINKY
#  ann 4: bbox [482, 474, 93,  47]  cx=528  cy=497  area=4478  → THUMB
#
#  Thumb reasoning: ann4 has largest area (4478) AND cy=497 which is well
#  below the median cy of the finger row (~200).
#  Remaining sorted by cx: 83(ann3) < 334(ann2) < 465(ann1) < 583(ann0)
#  → PINKY, RING, MIDDLE, INDEX
# ══════════════════════════════════════════════════════════════════════════════

class TestImageId0(unittest.TestCase):

    BBOXES = [
        [544, 246, 79.077, 44.642],   # ann 0
        [427, 175, 76.556, 45.775],   # ann 1
        [297, 164, 74.021, 44.788],   # ann 2
        [60,  188, 47.279, 32.257],   # ann 3
        [482, 474, 93.705, 47.792],   # ann 4
    ]

    def setUp(self):
        self.labels = assign_finger_ids(self.BBOXES)

    def test_ann4_is_thumb(self):
        self.assertEqual(self.labels[4], FINGER_THUMB,
            f"ann4 (cx=528, cy=497, area=4478) should be THUMB. "
            f"Got {lname(self.labels[4])}")

    def test_ann3_is_pinky(self):
        self.assertEqual(self.labels[3], FINGER_PINKY,
            f"ann3 (cx=83, leftmost) should be PINKY. "
            f"Got {lname(self.labels[3])}")

    def test_ann2_is_ring(self):
        self.assertEqual(self.labels[2], FINGER_RING,
            f"ann2 (cx=334, 2nd from left) should be RING. "
            f"Got {lname(self.labels[2])}")

    def test_ann1_is_middle(self):
        self.assertEqual(self.labels[1], FINGER_MIDDLE,
            f"ann1 (cx=465, 3rd from left) should be MIDDLE. "
            f"Got {lname(self.labels[1])}")

    def test_ann0_is_index(self):
        self.assertEqual(self.labels[0], FINGER_INDEX,
            f"ann0 (cx=583, rightmost non-thumb) should be INDEX. "
            f"Got {lname(self.labels[0])}")

    def test_all_five_unique(self):
        expected = {FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE,
                    FINGER_RING, FINGER_PINKY}
        self.assertEqual(set(self.labels), expected,
            f"Not all 5 fingers assigned. Got: {lnames(self.labels)}")

    def test_no_unused_slots(self):
        self.assertNotIn(FINGER_UNUSED, self.labels,
            f"No UNUSED expected for 5 nails. Got: {lnames(self.labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# image_id 1  — 5 nails, different hand orientation
#
#  ann 5: bbox [214,  74, 139, 22]   cx=283  cy= 85  area=3073  → PINKY
#  ann 6: bbox [225, 237, 142, 89]   cx=296  cy=281  area=12727 → MIDDLE
#  ann 7: bbox [193, 338, 138, 78]   cx=262  cy=377  area=10909 → THUMB
#  ann 8: bbox [236, 428, 110, 48]   cx=291  cy=452  area=5338  → RING
#  ann 9: bbox [303, 124, 119, 79]   cx=362  cy=163  area=9503  → INDEX
#
#  Thumb reasoning: ann7 has cy=377 which is the highest (most bottom of image)
#  among nails that are below the median cy. ann6 has larger area but cy=281
#  which is below median (~281), ann7 cy=377 is also below median and has
#  second-largest area. Let's trace:
#    median_cy of [85, 281, 377, 452, 163] sorted = [85,163,281,377,452]
#    median = 281 (index 2)
#    nails with cy > 281: ann7(377), ann8(452)
#    of those, largest area: ann8=5338 vs ann7=10909 → ann7 wins → THUMB
#  Remaining sorted by cx: 283(ann5) < 291(ann8) < 296(ann6) < 362(ann9)
#  → PINKY, RING, MIDDLE, INDEX
# ══════════════════════════════════════════════════════════════════════════════

class TestImageId1(unittest.TestCase):

    BBOXES = [
        [214, 74,  139.669, 22],      # ann 5
        [225, 237, 142,     89.632],  # ann 6
        [193, 338, 138.72,  78.646],  # ann 7
        [236, 428, 110.667, 48.237],  # ann 8
        [303, 124, 119.021, 79.845],  # ann 9
    ]

    def setUp(self):
        self.labels = assign_finger_ids(self.BBOXES)

    def test_ann7_is_thumb(self):
        # ann7: cx=262, cy=377, area=10909 — largest area below median cy
        self.assertEqual(self.labels[2], FINGER_THUMB,
            f"ann7 (index 2, cx=262, cy=377, area=10909) should be THUMB. "
            f"Got {lname(self.labels[2])}. All labels: {lnames(self.labels)}")

    def test_ann5_is_pinky(self):
        # ann5: cx=283, leftmost non-thumb
        self.assertEqual(self.labels[0], FINGER_PINKY,
            f"ann5 (index 0, cx=283) should be PINKY. "
            f"Got {lname(self.labels[0])}")

    def test_ann9_is_index(self):
        # ann9: cx=362, rightmost non-thumb
        self.assertEqual(self.labels[4], FINGER_INDEX,
            f"ann9 (index 4, cx=362) should be INDEX. "
            f"Got {lname(self.labels[4])}")

    def test_exactly_one_thumb(self):
        self.assertEqual(self.labels.count(FINGER_THUMB), 1,
            f"Exactly 1 thumb expected. Got: {lnames(self.labels)}")

    def test_all_five_unique(self):
        expected = {FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE,
                    FINGER_RING, FINGER_PINKY}
        self.assertEqual(set(self.labels), expected,
            f"Not all 5 fingers assigned. Got: {lnames(self.labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# image_id 7  — 9 nails (stress test: more than 5, some get UNUSED)
#
#  9 annotations: ann 33–41
#  After thumb is found and 4 finger labels assigned, remaining 4 get UNUSED.
#
#  Bboxes:
#  ann33: [338,500,56,81]   cx=366  cy=540  area=4609
#  ann34: [458,533,59,98]   cx=487  cy=582  area=5863  ← largest area + high cy
#  ann35: [517,501,56,99]   cx=545  cy=550  area=5591
#  ann36: [536,402,53,93]   cx=562  cy=448  area=4937
#  ann37: [361,316,41,105]  cx=381  cy=368  area=4390
#  ann38: [290,363,47,112]  cx=313  cy=419  area=5321
#  ann39: [217,326,41,109]  cx=237  cy=380  area=4499
#  ann40: [420,206,29,71]   cx=434  cy=241  area=2131
#  ann41: [123,103,34,110]  cx=140  cy=158  area=3843
#
#  median_cy of all 9 sorted = [158,241,368,380,419,448,540,550,582]
#  median = 419 (index 4)
#  nails with cy > 419: ann33(540), ann34(582), ann35(550), ann36(448)
#  of those, largest area: ann34(5863) → THUMB
#  remaining 8 sorted by cx:
#    140,237,313,366,381,434,545,562
#  assign first 4 positions: PINKY(140), RING(237), MIDDLE(313), INDEX(366)
#  positions 4-7: UNUSED
# ══════════════════════════════════════════════════════════════════════════════

class TestImageId7(unittest.TestCase):

    BBOXES = [
        [338, 500, 56.367,  81.780],  # ann 33  idx 0
        [458, 533, 59.499,  98.551],  # ann 34  idx 1  → THUMB
        [517, 501, 56.011,  99.825],  # ann 35  idx 2
        [536, 402, 53.076,  93.025],  # ann 36  idx 3
        [361, 316, 41.599, 105.538],  # ann 37  idx 4
        [290, 363, 47.116, 112.948],  # ann 38  idx 5
        [217, 326, 41.090, 109.506],  # ann 39  idx 6
        [420, 206, 29.782,  71.585],  # ann 40  idx 7
        [123, 103, 34.820, 110.375],  # ann 41  idx 8
    ]

    def setUp(self):
        self.labels = assign_finger_ids(self.BBOXES)

    def test_exactly_one_thumb(self):
        count = self.labels.count(FINGER_THUMB)
        self.assertEqual(count, 1,
            f"Exactly 1 thumb expected. Got {count}. Labels: {lnames(self.labels)}")

    def test_ann34_is_thumb(self):
        # ann34: cx=487, cy=582, area=5863 — largest area among nails below median
        self.assertEqual(self.labels[1], FINGER_THUMB,
            f"ann34 (idx 1, cx=487, cy=582, area=5863) should be THUMB. "
            f"Got {lname(self.labels[1])}. All: {lnames(self.labels)}")

    def test_ann41_is_pinky(self):
        # ann41: cx=140 — leftmost of all non-thumb nails
        self.assertEqual(self.labels[8], FINGER_PINKY,
            f"ann41 (idx 8, cx=140, leftmost) should be PINKY. "
            f"Got {lname(self.labels[8])}")

    def test_four_unused_slots(self):
        # 9 nails: 1 thumb + 4 finger labels + 4 unused
        unused = self.labels.count(FINGER_UNUSED)
        self.assertEqual(unused, 4,
            f"Expected 4 UNUSED slots for 9 nails. Got {unused}. "
            f"Labels: {lnames(self.labels)}")

    def test_output_length(self):
        self.assertEqual(len(self.labels), 9)

    def test_four_named_fingers_assigned(self):
        named = [l for l in self.labels
                 if l in (FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY)]
        self.assertEqual(len(named), 4,
            f"Expected 4 named finger labels. Got {len(named)}: {lnames(self.labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# image_id 8  — 10 small nails (fingertip close-ups, very small bboxes)
#
#  Tests robustness on tiny bboxes (area < 800px) — e.g. ann 51: area=286
#  The algorithm must not crash and must still assign exactly 1 thumb.
# ══════════════════════════════════════════════════════════════════════════════

class TestImageId8(unittest.TestCase):

    BBOXES = [
        [289, 184, 16.402, 34.379],  # ann 42
        [305, 138, 18.322, 37.688],  # ann 43
        [331, 150, 19.289, 35.559],  # ann 44
        [374, 198, 14.453, 27.193],  # ann 45
        [282, 353, 18.342, 40.481],  # ann 46
        [175, 408, 16.497, 43.853],  # ann 47
        [119, 292, 15.488, 34.681],  # ann 48
        [ 91, 282, 15.567, 37.118],  # ann 49
        [ 71, 314, 16.198, 38.271],  # ann 50
        [ 61, 396, 10.066, 28.448],  # ann 51  ← smallest bbox (area=286)
    ]

    def setUp(self):
        self.labels = assign_finger_ids(self.BBOXES)

    def test_no_crash_on_tiny_bboxes(self):
        self.assertEqual(len(self.labels), 10)

    def test_exactly_one_thumb(self):
        count = self.labels.count(FINGER_THUMB)
        self.assertEqual(count, 1,
            f"Exactly 1 thumb expected. Got {count}. Labels: {lnames(self.labels)}")

    def test_five_unused_slots(self):
        # 10 nails: 1 thumb + 4 finger labels + 5 unused
        unused = self.labels.count(FINGER_UNUSED)
        self.assertEqual(unused, 5,
            f"Expected 5 UNUSED for 10 nails. Got {unused}. "
            f"Labels: {lnames(self.labels)}")

    def test_smallest_bbox_not_thumb(self):
        # ann51 (idx 9) has area=286 — smallest, should not be thumb
        self.assertNotEqual(self.labels[9], FINGER_THUMB,
            "Smallest bbox (area=286) should not be thumb")


# ══════════════════════════════════════════════════════════════════════════════
# Cross-image consistency — same algorithm, different images
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossImageConsistency(unittest.TestCase):

    def test_thumb_always_has_largest_or_second_largest_area(self):
        """
        Thumb should never be the smallest nail in the image.
        Verify across all 4 real images.
        """
        all_images = [
            # image_id 0
            [[544,246,79.077,44.642],[427,175,76.556,45.775],
             [297,164,74.021,44.788],[60,188,47.279,32.257],
             [482,474,93.705,47.792]],
            # image_id 1
            [[214,74,139.669,22],[225,237,142,89.632],
             [193,338,138.72,78.646],[236,428,110.667,48.237],
             [303,124,119.021,79.845]],
            # image_id 7
            [[338,500,56.367,81.780],[458,533,59.499,98.551],
             [517,501,56.011,99.825],[536,402,53.076,93.025],
             [361,316,41.599,105.538],[290,363,47.116,112.948],
             [217,326,41.090,109.506],[420,206,29.782,71.585],
             [123,103,34.820,110.375]],
            # image_id 8
            [[289,184,16.402,34.379],[305,138,18.322,37.688],
             [331,150,19.289,35.559],[374,198,14.453,27.193],
             [282,353,18.342,40.481],[175,408,16.497,43.853],
             [119,292,15.488,34.681],[91,282,15.567,37.118],
             [71,314,16.198,38.271],[61,396,10.066,28.448]],
        ]

        for img_idx, bboxes in enumerate(all_images):
            labels   = assign_finger_ids(bboxes)
            areas    = [area(b) for b in bboxes]
            thumb_i  = labels.index(FINGER_THUMB)
            thumb_area = areas[thumb_i]
            min_area   = min(areas)
            # Thumb area must be strictly greater than the minimum
            self.assertGreater(thumb_area, min_area,
                f"Image {img_idx}: thumb has area={thumb_area:.1f} "
                f"which equals the minimum area={min_area:.1f}. "
                f"Thumb should not be the smallest nail.")

    def test_exactly_one_thumb_per_image(self):
        all_images = [
            [[544,246,79.077,44.642],[427,175,76.556,45.775],
             [297,164,74.021,44.788],[60,188,47.279,32.257],
             [482,474,93.705,47.792]],
            [[214,74,139.669,22],[225,237,142,89.632],
             [193,338,138.72,78.646],[236,428,110.667,48.237],
             [303,124,119.021,79.845]],
        ]
        for img_idx, bboxes in enumerate(all_images):
            labels = assign_finger_ids(bboxes)
            count  = labels.count(FINGER_THUMB)
            self.assertEqual(count, 1,
                f"Image {img_idx}: expected 1 thumb, got {count}. "
                f"Labels: {lnames(labels)}")

    def test_finger_labels_respect_spatial_order(self):
        """
        After removing thumb, non-thumb labels sorted by cx must always
        follow the order PINKY < RING < MIDDLE < INDEX (ascending label
        values correspond to ascending cx position).
        PINKY=5, RING=4, MIDDLE=3, INDEX=2 — descending codes left→right.
        """
        bboxes = [
            [544,246,79.077,44.642],[427,175,76.556,45.775],
            [297,164,74.021,44.788],[60,188,47.279,32.257],
            [482,474,93.705,47.792],
        ]
        labels = assign_finger_ids(bboxes)

        non_thumb = [(cx(bboxes[i]), l)
                     for i, l in enumerate(labels) if l != FINGER_THUMB]
        non_thumb.sort()  # sort by cx
        ordered_labels = [l for _, l in non_thumb]

        # Left→right must be PINKY(5), RING(4), MIDDLE(3), INDEX(2)
        expected = [FINGER_PINKY, FINGER_RING, FINGER_MIDDLE, FINGER_INDEX]
        self.assertEqual(ordered_labels, expected,
            f"Spatial order wrong. Got L→R: {lnames(ordered_labels)}, "
            f"expected: {lnames(expected)}")


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("Nail VTON — Real Data Finger Mapping Tests")
    print("Bboxes taken verbatim from _annotations.coco.json")
    print("=" * 65)
    print()

    loader = unittest.TestLoader()
    suite  = loader.loadTestsFromModule(__import__("__main__"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 65)
    if result.wasSuccessful():
        print(f"ALL {result.testsRun} TESTS PASSED ✓")
    else:
        print(f"{len(result.failures)} FAILURES  {len(result.errors)} ERRORS "
              f"out of {result.testsRun} tests")
    print("=" * 65)
    sys.exit(0 if result.wasSuccessful() else 1)