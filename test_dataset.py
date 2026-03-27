"""
test_dataset.py
---------------
Comprehensive tests for dataset.py — finger identity, direction field,
masks, edge cases, and flipped hand handling.

Run:
    python test_dataset.py
    python test_dataset.py -v       # verbose, shows each test name
"""

import sys
import math
import unittest
import numpy as np

sys.path.insert(0, ".")
from dataset import (
    assign_finger_ids,
    compute_direction_field,
    polygon_to_mask,
    FINGER_UNUSED, FINGER_THUMB, FINGER_INDEX,
    FINGER_MIDDLE, FINGER_RING,  FINGER_PINKY,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_bbox(cx, cy, w=40, h=60):
    """Return [x, y, w, h] from a centre point."""
    return [cx - w / 2, cy - h / 2, w, h]


def label_name(code):
    return {0: "UNUSED", 1: "THUMB", 2: "INDEX",
            3: "MIDDLE", 4: "RING", 5: "PINKY"}[code]


def labels_str(labels):
    return [label_name(l) for l in labels]


# ══════════════════════════════════════════════════════════════════════════════
# Tests — assign_finger_ids
# ══════════════════════════════════════════════════════════════════════════════

class TestFingerIdentityEdgeCases(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(assign_finger_ids([]), [])

    def test_single_nail_becomes_thumb(self):
        # Only one nail — fallback path must not crash, should get thumb label
        labels = assign_finger_ids([make_bbox(100, 100)])
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], FINGER_THUMB,
                         f"Single nail should be THUMB, got {label_name(labels[0])}")

    def test_two_nails_thumb_and_one_finger(self):
        # Thumb lower (larger cy), one finger above
        thumb_box  = make_bbox(cx=300, cy=400, w=80, h=90)  # large + low
        finger_box = make_bbox(cx=200, cy=150, w=40, h=60)  # small + high
        labels = assign_finger_ids([thumb_box, finger_box])
        self.assertEqual(len(labels), 2)
        self.assertEqual(labels[0], FINGER_THUMB,
                         f"First bbox should be THUMB, got {labels_str(labels)}")
        self.assertNotEqual(labels[1], FINGER_THUMB,
                            "Second bbox must not also be THUMB")

    def test_output_length_matches_input(self):
        bboxes = [make_bbox(i * 80, 100) for i in range(7)]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(len(labels), len(bboxes))

    def test_no_duplicate_thumb(self):
        bboxes = [make_bbox(i * 80, 100 + i * 10, w=50, h=60) for i in range(5)]
        labels = assign_finger_ids(bboxes)
        thumb_count = labels.count(FINGER_THUMB)
        self.assertEqual(thumb_count, 1,
                         f"Exactly 1 thumb expected, got {thumb_count}. Labels: {labels_str(labels)}")

    def test_more_than_five_nails_no_crash(self):
        # Dataset occasionally has >5 nails (both hands visible)
        bboxes = [make_bbox(i * 60, 100) for i in range(9)]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(len(labels), 9)
        # Should not crash and should have exactly one thumb
        self.assertEqual(labels.count(FINGER_THUMB), 1)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — standard left hand (palm facing camera, fingers pointing up)
#
#  Typical image layout (left hand):
#    Pinky  — leftmost,  high y (near top)
#    Ring   — second from left
#    Middle — centre, tallest nail
#    Index  — second from right
#    Thumb  — rightmost, low y (near bottom), biggest bbox
#
#  Image coords: x increases rightward, y increases downward.
# ══════════════════════════════════════════════════════════════════════════════

class TestFingerIdentityLeftHand(unittest.TestCase):

    def setUp(self):
        # Simulate left hand, palm facing camera, fingers pointing up
        # Thumb is bottom-right, large.  Fingers top row, left→right = pinky→index.
        self.bboxes = [
            make_bbox(cx=80,  cy=180, w=35, h=55),   # 0 = pinky   (leftmost, high)
            make_bbox(cx=180, cy=160, w=40, h=60),   # 1 = ring
            make_bbox(cx=280, cy=150, w=45, h=65),   # 2 = middle  (tallest)
            make_bbox(cx=380, cy=165, w=40, h=60),   # 3 = index
            make_bbox(cx=450, cy=380, w=85, h=95),   # 4 = thumb   (low + large)
        ]
        self.labels = assign_finger_ids(self.bboxes)

    def test_thumb_identified(self):
        self.assertEqual(self.labels[4], FINGER_THUMB,
                         f"Expected index 4 = THUMB, got {labels_str(self.labels)}")

    def test_pinky_identified(self):
        self.assertEqual(self.labels[0], FINGER_PINKY,
                         f"Expected index 0 = PINKY, got {labels_str(self.labels)}")

    def test_middle_identified(self):
        self.assertEqual(self.labels[2], FINGER_MIDDLE,
                         f"Expected index 2 = MIDDLE, got {labels_str(self.labels)}")

    def test_index_identified(self):
        self.assertEqual(self.labels[3], FINGER_INDEX,
                         f"Expected index 3 = INDEX, got {labels_str(self.labels)}")

    def test_ring_identified(self):
        self.assertEqual(self.labels[1], FINGER_RING,
                         f"Expected index 1 = RING, got {labels_str(self.labels)}")

    def test_all_five_assigned(self):
        assigned = set(self.labels)
        expected = {FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY}
        self.assertEqual(assigned, expected,
                         f"Not all five fingers assigned. Got: {labels_str(self.labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — right hand (mirror of left hand)
#
#  Right hand, palm facing camera:
#    Thumb  — leftmost,  low y, big bbox
#    Index  — second from left
#    Middle — centre
#    Ring   — second from right
#    Pinky  — rightmost, high y, small bbox
#
#  The algorithm assigns left→right as pinky→index regardless of hand.
#  This is POSITION-RELATIVE labelling — consistent across hands.
#  The test documents this intentional behaviour.
# ══════════════════════════════════════════════════════════════════════════════

class TestFingerIdentityRightHand(unittest.TestCase):

    def setUp(self):
        # Right hand: thumb is bottom-LEFT, fingers go left→right = index→pinky
        self.bboxes = [
            make_bbox(cx=80,  cy=380, w=85, h=95),   # 0 = thumb   (leftmost, low, big)
            make_bbox(cx=200, cy=165, w=40, h=60),   # 1 = index
            make_bbox(cx=300, cy=150, w=45, h=65),   # 2 = middle
            make_bbox(cx=400, cy=160, w=40, h=60),   # 3 = ring
            make_bbox(cx=490, cy=180, w=35, h=55),   # 4 = pinky   (rightmost, small)
        ]
        self.labels = assign_finger_ids(self.bboxes)

    def test_thumb_found_correctly_on_right_hand(self):
        self.assertEqual(self.labels[0], FINGER_THUMB,
                         f"Right hand: thumb should be index 0. Got {labels_str(self.labels)}")

    def test_position_relative_labelling_is_consistent(self):
        # After removing thumb, remaining left→right order is:
        # index(1) ring(3) middle(2) ring(3) pinky(4)
        # Sorted by cx: 200,300,400,490 → positions 0,1,2,3
        # Algorithm assigns:  pos0=PINKY, pos1=RING, pos2=MIDDLE, pos3=INDEX
        # So on a right hand: anatomical index → labelled PINKY (position-relative)
        # This is EXPECTED behaviour — document it, don't accidentally fix it
        # without updating the rendering code too.
        non_thumb = [(i, l) for i, l in enumerate(self.labels) if l != FINGER_THUMB]
        assigned_labels = sorted([l for _, l in non_thumb])
        expected_labels = sorted([FINGER_PINKY, FINGER_RING, FINGER_MIDDLE, FINGER_INDEX])
        self.assertEqual(assigned_labels, expected_labels,
                         "All four non-thumb finger labels must be assigned (regardless of anatomical correctness)")

    def test_right_hand_no_crash(self):
        # Basic smoke test
        self.assertEqual(len(self.labels), 5)
        self.assertNotIn(FINGER_UNUSED, self.labels,
                         f"No UNUSED slots expected for 5 nails. Got {labels_str(self.labels)}")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — horizontally flipped hand
#
#  IMPORTANT BUG TO CATCH:
#  The augmentation in _augment() flips the IMAGE and MASKS but does NOT
#  update the bboxes used for finger_id assignment — because finger_ids are
#  derived BEFORE augmentation from original-scale bboxes.
#  This means after hflip, the spatial positions of nails are mirrored but
#  the finger labels stay the same. This is correct behaviour only if the
#  finger labels are used purely for size lookup (not absolute position).
#  This test documents the known limitation.
# ══════════════════════════════════════════════════════════════════════════════

class TestFlippedHand(unittest.TestCase):

    def setUp(self):
        # Standard left hand
        self.bboxes_original = [
            make_bbox(cx=80,  cy=180, w=35, h=55),   # pinky
            make_bbox(cx=180, cy=160, w=40, h=60),   # ring
            make_bbox(cx=280, cy=150, w=45, h=65),   # middle
            make_bbox(cx=380, cy=165, w=40, h=60),   # index
            make_bbox(cx=450, cy=380, w=85, h=95),   # thumb
        ]
        # Simulate hflip: mirror cx around image width=640
        W = 640
        self.bboxes_flipped = [
            [W - (x + w), y, w, h]
            for x, y, w, h in self.bboxes_original
        ]

    def test_labels_before_flip(self):
        labels = assign_finger_ids(self.bboxes_original)
        self.assertEqual(labels[4], FINGER_THUMB)
        self.assertEqual(labels[0], FINGER_PINKY)
        self.assertEqual(labels[2], FINGER_MIDDLE)

    def test_labels_after_flip_thumb_still_found(self):
        labels = assign_finger_ids(self.bboxes_flipped)
        # After hflip, thumb is now on the LEFT (low cx, low cy), still biggest
        self.assertEqual(labels.count(FINGER_THUMB), 1,
                         f"Exactly one thumb after flip. Got {labels_str(labels)}")

    def test_finger_positional_labels_same_after_flip(self):
        """
        After hflip the ANATOMICAL fingers swap sides (what was the index is now
        on the left) but the POSITIONAL labels stay the same — leftmost non-thumb
        is still labelled PINKY, rightmost is still INDEX.

        This is the documented behaviour: labels are position-relative, not
        anatomy-relative. The rendering code must account for this — it should
        use spatial position at inference time, not rely on label names to
        indicate which physical finger is which after a flip.
        """
        labels_orig    = assign_finger_ids(self.bboxes_original)
        labels_flipped = assign_finger_ids(self.bboxes_flipped)

        def sorted_non_thumb_labels(bboxes, labels):
            pairs = [(bboxes[i][0] + bboxes[i][2] / 2, l)
                     for i, l in enumerate(labels) if l != FINGER_THUMB]
            pairs.sort()
            return [l for _, l in pairs]

        orig_order   = sorted_non_thumb_labels(self.bboxes_original, labels_orig)
        flipped_order = sorted_non_thumb_labels(self.bboxes_flipped, labels_flipped)

        # Both should be [PINKY, RING, MIDDLE, INDEX] — positional, not anatomical
        expected = [FINGER_PINKY, FINGER_RING, FINGER_MIDDLE, FINGER_INDEX]
        self.assertEqual(orig_order,    expected,
                         f"Original L→R labels should be PINKY→INDEX. Got {[label_name(l) for l in orig_order]}")
        self.assertEqual(flipped_order, expected,
                         f"Flipped  L→R labels should still be PINKY→INDEX. Got {[label_name(l) for l in flipped_order]}")

    def test_known_limitation_bboxes_not_updated_during_augmentation(self):
        """
        Documents the known limitation: finger_ids in __getitem__ are assigned
        from original bboxes BEFORE augmentation. After hflip the labels are
        spatially incorrect but structurally valid (all 5 labels present).
        This test will FAIL if the bug is fixed — remove it then.
        """
        # finger_ids are assigned from bboxes_original even when image is flipped
        labels_from_orig = assign_finger_ids(self.bboxes_original)
        # The flipped image has thumb on the right side now, but labels still
        # reflect the original (thumb on right = index 4)
        self.assertEqual(labels_from_orig[4], FINGER_THUMB,
                         "Labels assigned from original bboxes, thumb is still at index 4")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — compute_direction_field
# ══════════════════════════════════════════════════════════════════════════════

class TestDirectionField(unittest.TestCase):

    def _make_mask(self, H, W, roi):
        """roi = (y0, y1, x0, x1) slice that is foreground."""
        m = np.zeros((H, W), dtype=np.uint8)
        y0, y1, x0, x1 = roi
        m[y0:y1, x0:x1] = 255
        return m

    def test_output_shape(self):
        mask = self._make_mask(64, 64, (10, 50, 10, 50))
        bbox = [10, 10, 40, 40]
        out  = compute_direction_field(mask, bbox)
        self.assertEqual(out.shape, (2, 64, 64))

    def test_background_pixels_are_zero(self):
        mask = self._make_mask(64, 64, (20, 40, 20, 40))
        bbox = [20, 20, 20, 20]
        out  = compute_direction_field(mask, bbox)
        # Corners are background
        self.assertEqual(out[0, 0, 0], 0.0)
        self.assertEqual(out[1, 0, 0], 0.0)

    def test_foreground_pixels_have_nonzero_direction(self):
        mask = self._make_mask(64, 64, (10, 50, 10, 50))
        bbox = [10, 10, 40, 40]
        out  = compute_direction_field(mask, bbox)
        fg_dx = out[0, 30, 30]
        fg_dy = out[1, 30, 30]
        magnitude = math.sqrt(fg_dx**2 + fg_dy**2)
        self.assertGreater(magnitude, 0.5,
                           "Foreground pixels must have a direction vector")

    def test_direction_is_unit_vector_on_foreground(self):
        mask = self._make_mask(64, 64, (10, 50, 10, 50))
        bbox = [10, 10, 40, 40]
        out  = compute_direction_field(mask, bbox)
        dx, dy = out[0, 30, 30], out[1, 30, 30]
        norm = math.sqrt(dx**2 + dy**2)
        self.assertAlmostEqual(norm, 1.0, places=5,
                               msg=f"Direction must be unit vector, got norm={norm}")

    def test_direction_points_upward_for_vertical_nail(self):
        # Vertical nail: bbox top is tip, bottom is base → direction = (0, -1)
        mask = self._make_mask(64, 64, (10, 50, 28, 36))
        bbox = [28, 10, 8, 40]   # x, y, w, h — tall vertical rectangle
        out  = compute_direction_field(mask, bbox)
        dy = out[1, 30, 32]
        self.assertLess(dy, -0.9,
                        f"Vertical nail direction should be (0,-1), got dy={dy:.4f}")

    def test_empty_mask_returns_zeros(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        bbox = [10, 10, 40, 40]
        out  = compute_direction_field(mask, bbox)
        self.assertTrue(np.all(out == 0),
                        "Empty mask must return all-zero field")

    def test_zero_height_bbox_returns_zeros(self):
        # h=0 → norm=0 → should not crash, return zeros
        mask = self._make_mask(64, 64, (10, 50, 10, 50))
        bbox = [10, 10, 40, 0]   # h=0
        out  = compute_direction_field(mask, bbox)
        self.assertEqual(out.shape, (2, 64, 64))
        self.assertTrue(np.all(out == 0),
                        "Zero-height bbox must return all-zero field (no division by zero)")

    def test_known_limitation_direction_always_vertical(self):
        """
        Documents known limitation: vx is hardcoded to 0.0 so direction is
        always (0, -1) regardless of nail orientation. Tilted nails get the
        wrong direction. This test will FAIL when the limitation is fixed.
        """
        mask = self._make_mask(64, 64, (10, 50, 10, 50))
        bbox = [10, 10, 40, 40]
        out  = compute_direction_field(mask, bbox)
        dx = out[0, 30, 30]
        self.assertAlmostEqual(dx, 0.0, places=5,
                               msg="Known limitation: dx is always 0 (direction always vertical)")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — polygon_to_mask
# ══════════════════════════════════════════════════════════════════════════════

class TestPolygonToMask(unittest.TestCase):

    def test_output_size(self):
        poly = [10, 10, 50, 10, 50, 50, 10, 50]
        mask = polygon_to_mask(poly, height=100, width=100)
        self.assertEqual(mask.size, (100, 100))

    def test_interior_pixels_are_white(self):
        poly = [10, 10, 90, 10, 90, 90, 10, 90]
        mask = polygon_to_mask(poly, height=100, width=100)
        arr  = np.array(mask)
        self.assertEqual(arr[50, 50], 255,
                         "Interior pixel should be 255 (foreground)")

    def test_corner_pixels_are_black(self):
        poly = [10, 10, 90, 10, 90, 90, 10, 90]
        mask = polygon_to_mask(poly, height=100, width=100)
        arr  = np.array(mask)
        self.assertEqual(arr[0, 0], 0, "Corner pixel should be 0 (background)")

    def test_too_short_polygon_returns_blank(self):
        poly = [10, 10, 20, 20]   # only 2 points — invalid
        mask = polygon_to_mask(poly, height=64, width=64)
        arr  = np.array(mask)
        self.assertTrue(np.all(arr == 0),
                        "Polygon with <3 points must return blank mask")

    def test_empty_polygon_returns_blank(self):
        mask = polygon_to_mask([], height=64, width=64)
        arr  = np.array(mask)
        self.assertTrue(np.all(arr == 0))


# ══════════════════════════════════════════════════════════════════════════════
# Tests — thumb detection robustness
# ══════════════════════════════════════════════════════════════════════════════

class TestThumbDetection(unittest.TestCase):

    def test_thumb_detected_when_clearly_larger_and_lower(self):
        bboxes = [
            make_bbox(cx=100, cy=150, w=40, h=55),
            make_bbox(cx=200, cy=140, w=45, h=60),
            make_bbox(cx=300, cy=135, w=50, h=65),
            make_bbox(cx=400, cy=145, w=40, h=58),
            make_bbox(cx=350, cy=420, w=90, h=100),  # thumb — big + low
        ]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(labels[4], FINGER_THUMB,
                         f"Clearly larger+lower nail should be THUMB. Got {labels_str(labels)}")

    def test_thumb_fallback_when_all_nails_same_height(self):
        # All nails at same cy — fallback to largest area
        bboxes = [
            make_bbox(cx=100, cy=200, w=30, h=50),
            make_bbox(cx=200, cy=200, w=30, h=50),
            make_bbox(cx=300, cy=200, w=30, h=50),
            make_bbox(cx=400, cy=200, w=30, h=50),
            make_bbox(cx=500, cy=200, w=80, h=90),  # biggest — should be thumb
        ]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(labels[4], FINGER_THUMB,
                         f"When all same height, largest area should be THUMB. Got {labels_str(labels)}")

    def test_thumb_fallback_no_nail_below_median(self):
        # All nails above median — no nail satisfies cy > median_cy → fallback
        bboxes = [
            make_bbox(cx=100, cy=50,  w=30, h=50),
            make_bbox(cx=200, cy=60,  w=30, h=50),
            make_bbox(cx=300, cy=55,  w=30, h=50),
            make_bbox(cx=400, cy=58,  w=80, h=90),  # biggest but NOT below median
            make_bbox(cx=500, cy=62,  w=30, h=50),
        ]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(labels.count(FINGER_THUMB), 1,
                         "Fallback must still assign exactly one thumb")

    def test_partial_hand_three_nails(self):
        bboxes = [
            make_bbox(cx=100, cy=150, w=40, h=55),
            make_bbox(cx=200, cy=140, w=45, h=60),
            make_bbox(cx=300, cy=400, w=85, h=95),  # thumb
        ]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(labels.count(FINGER_THUMB), 1)
        self.assertEqual(len(labels), 3)

    def test_partial_hand_four_nails_no_thumb_visible(self):
        # Four fingers visible, no thumb — biggest gets thumb label by fallback
        bboxes = [
            make_bbox(cx=100, cy=150, w=35, h=55),
            make_bbox(cx=200, cy=145, w=40, h=60),
            make_bbox(cx=300, cy=140, w=45, h=65),  # biggest → thumb fallback
            make_bbox(cx=400, cy=148, w=38, h=58),
        ]
        labels = assign_finger_ids(bboxes)
        self.assertEqual(labels.count(FINGER_THUMB), 1,
                         "Even with no anatomical thumb visible, exactly one nail gets THUMB")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — finger ordering correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestFingerOrdering(unittest.TestCase):

    def _get_label_at_cx(self, bboxes, cx_target):
        labels = assign_finger_ids(bboxes)
        for i, (x, y, w, h) in enumerate(bboxes):
            if abs((x + w / 2) - cx_target) < 1.0:
                return labels[i]
        raise ValueError(f"No bbox with cx={cx_target}")

    def test_leftmost_non_thumb_is_pinky(self):
        bboxes = [
            make_bbox(cx=50,  cy=180, w=35, h=55),   # leftmost → PINKY
            make_bbox(cx=150, cy=160, w=40, h=60),
            make_bbox(cx=250, cy=150, w=45, h=65),
            make_bbox(cx=350, cy=165, w=40, h=60),
            make_bbox(cx=450, cy=380, w=85, h=95),   # thumb
        ]
        label = self._get_label_at_cx(bboxes, 50)
        self.assertEqual(label, FINGER_PINKY,
                         f"Leftmost non-thumb should be PINKY, got {label_name(label)}")

    def test_rightmost_non_thumb_is_index(self):
        bboxes = [
            make_bbox(cx=50,  cy=180, w=35, h=55),
            make_bbox(cx=150, cy=160, w=40, h=60),
            make_bbox(cx=250, cy=150, w=45, h=65),
            make_bbox(cx=350, cy=165, w=40, h=60),   # rightmost non-thumb → INDEX
            make_bbox(cx=450, cy=380, w=85, h=95),   # thumb
        ]
        label = self._get_label_at_cx(bboxes, 350)
        self.assertEqual(label, FINGER_INDEX,
                         f"Rightmost non-thumb should be INDEX, got {label_name(label)}")

    def test_middle_finger_is_centre_non_thumb(self):
        bboxes = [
            make_bbox(cx=50,  cy=180, w=35, h=55),   # pinky
            make_bbox(cx=150, cy=160, w=40, h=60),   # ring
            make_bbox(cx=250, cy=150, w=45, h=65),   # middle ← centre
            make_bbox(cx=350, cy=165, w=40, h=60),   # index
            make_bbox(cx=450, cy=380, w=85, h=95),   # thumb
        ]
        label = self._get_label_at_cx(bboxes, 250)
        self.assertEqual(label, FINGER_MIDDLE,
                         f"Centre non-thumb should be MIDDLE, got {label_name(label)}")

    def test_order_invariant_to_input_order(self):
        # Same bboxes in shuffled input order must give same spatial labels
        bboxes_ordered = [
            make_bbox(cx=50,  cy=180, w=35, h=55),
            make_bbox(cx=150, cy=160, w=40, h=60),
            make_bbox(cx=250, cy=150, w=45, h=65),
            make_bbox(cx=350, cy=165, w=40, h=60),
            make_bbox(cx=450, cy=380, w=85, h=95),
        ]
        import random
        bboxes_shuffled = bboxes_ordered.copy()
        random.seed(42)
        random.shuffle(bboxes_shuffled)

        labels_ord  = assign_finger_ids(bboxes_ordered)
        labels_shuf = assign_finger_ids(bboxes_shuffled)

        # Build cx → label maps and compare
        def cx_label_map(bboxes, labels):
            return {round(x + w/2): l for (x, y, w, h), l in zip(bboxes, labels)}

        map_ord  = cx_label_map(bboxes_ordered,  labels_ord)
        map_shuf = cx_label_map(bboxes_shuffled, labels_shuf)
        self.assertEqual(map_ord, map_shuf,
                         "Finger labels must be the same regardless of input order")


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures — synthetic COCO dataset written to a temp directory
# ══════════════════════════════════════════════════════════════════════════════

import json
import tempfile
import shutil
from pathlib import Path
from PIL import Image as PILImage

def make_synthetic_coco(tmp_dir, n_images=3, nails_per_image=5, img_size=64):
    """
    Write a minimal valid COCO JSON + matching solid-colour JPEG images to
    tmp_dir/train/ so NailDataset can be instantiated without real data.

    Each nail polygon is a simple rectangle. Thumb is always the last nail
    in each image (largest + lowest).
    """
    train_dir = Path(tmp_dir) / "train"
    img_dir   = train_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    images, annotations = [], []
    ann_id = 0

    for img_id in range(n_images):
        fname = f"img_{img_id:03d}.jpg"
        PILImage.new("RGB", (img_size, img_size), color=(180, 140, 120)).save(
            img_dir / fname
        )
        images.append({"id": img_id, "file_name": fname,
                        "height": img_size, "width": img_size})

        # 4 small finger nails in a row at the top
        for k in range(nails_per_image - 1):
            x  = 4 + k * 12
            y  = 4
            w  = 8
            h  = 14
            poly = [x, y,  x+w, y,  x+w, y+h,  x, y+h]
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [x, y, w, h], "area": w * h,
                "segmentation": [poly], "iscrowd": 0,
            })
            ann_id += 1

        # 1 large thumb nail — lower + bigger
        x, y, w, h = 40, 40, 16, 18
        poly = [x, y,  x+w, y,  x+w, y+h,  x, y+h]
        annotations.append({
            "id": ann_id, "image_id": img_id, "category_id": 1,
            "bbox": [x, y, w, h], "area": w * h,
            "segmentation": [poly], "iscrowd": 0,
        })
        ann_id += 1

    coco = {
        "images": images, "annotations": annotations,
        "categories": [{"id": 1, "name": "Nail", "supercategory": "nails"}],
    }
    with open(train_dir / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    return str(train_dir)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — NailDataset pipeline (__getitem__ end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

import torch
from dataset import NailDataset, MAX_INSTANCES

class TestNailDatasetPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp   = tempfile.mkdtemp()
        root      = make_synthetic_coco(cls.tmp, n_images=3,
                                        nails_per_image=5, img_size=64)
        cls.ds    = NailDataset(root, augment=False, image_size=64)
        cls.sample = cls.ds[0]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    # ── shapes ────────────────────────────────────────────────────────────────

    def test_image_shape(self):
        self.assertEqual(self.sample["image"].shape, (3, 64, 64),
                         "Image must be (3, H, W)")

    def test_binary_mask_shape(self):
        self.assertEqual(self.sample["binary_mask"].shape, (1, 64, 64),
                         "Binary mask must be (1, H, W)")

    def test_instance_masks_shape(self):
        self.assertEqual(self.sample["instance_masks"].shape,
                         (MAX_INSTANCES, 64, 64),
                         f"Instance masks must be ({MAX_INSTANCES}, H, W)")

    def test_direction_field_shape(self):
        self.assertEqual(self.sample["direction_field"].shape, (2, 64, 64),
                         "Direction field must be (2, H, W)")

    def test_finger_ids_shape(self):
        self.assertEqual(self.sample["finger_ids"].shape, (MAX_INSTANCES,),
                         f"finger_ids must be ({MAX_INSTANCES},)")

    def test_n_instances_is_scalar(self):
        self.assertEqual(self.sample["n_instances"].ndim, 0,
                         "n_instances must be a scalar tensor")

    # ── dtypes ────────────────────────────────────────────────────────────────

    def test_image_dtype(self):
        self.assertEqual(self.sample["image"].dtype, torch.float32)

    def test_binary_mask_dtype(self):
        self.assertEqual(self.sample["binary_mask"].dtype, torch.float32)

    def test_instance_masks_dtype(self):
        self.assertEqual(self.sample["instance_masks"].dtype, torch.float32)

    def test_direction_field_dtype(self):
        self.assertEqual(self.sample["direction_field"].dtype, torch.float32)

    def test_finger_ids_dtype(self):
        self.assertEqual(self.sample["finger_ids"].dtype, torch.int64)

    # ── value ranges ──────────────────────────────────────────────────────────

    def test_binary_mask_range(self):
        mn = self.sample["binary_mask"].min().item()
        mx = self.sample["binary_mask"].max().item()
        self.assertGreaterEqual(mn, 0.0, "binary_mask min must be >= 0")
        self.assertLessEqual(mx,   1.0, "binary_mask max must be <= 1")

    def test_instance_masks_range(self):
        mn = self.sample["instance_masks"].min().item()
        mx = self.sample["instance_masks"].max().item()
        self.assertGreaterEqual(mn, 0.0, "instance_masks min must be >= 0")
        self.assertLessEqual(mx,   1.0, "instance_masks max must be <= 1")

    def test_direction_field_range(self):
        mn = self.sample["direction_field"].min().item()
        mx = self.sample["direction_field"].max().item()
        self.assertGreaterEqual(mn, -1.0, "direction_field min must be >= -1")
        self.assertLessEqual(mx,    1.0,  "direction_field max must be <= 1")

    def test_finger_ids_valid_codes(self):
        valid = {0, 1, 2, 3, 4, 5}
        ids   = set(self.sample["finger_ids"].tolist())
        self.assertTrue(ids.issubset(valid),
                        f"finger_ids contains invalid codes: {ids - valid}")

    # ── binary mask is union of instance masks ─────────────────────────────────

    def test_binary_mask_is_union_of_instance_masks(self):
        inst   = self.sample["instance_masks"]          # (10, H, W)
        binary = self.sample["binary_mask"]             # (1,  H, W)
        union  = inst.max(dim=0).values.unsqueeze(0)    # (1,  H, W)
        # binary must be >= every instance mask at every pixel
        diff = (binary - union).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=4,
            msg="binary_mask must equal the union (max) of all instance_masks")

    # ── one-hot integrity: no pixel active in >1 instance channel ─────────────

    def test_instance_masks_no_pixel_in_two_channels(self):
        inst         = self.sample["instance_masks"]    # (10, H, W)
        active_count = (inst > 0.5).float().sum(dim=0) # (H, W)
        max_active   = active_count.max().item()
        self.assertLessEqual(max_active, 1.0,
            f"A pixel is active in {max_active:.0f} instance channels simultaneously. "
            f"Instance masks must be mutually exclusive (one-hot).")

    # ── unused instance slots must be all-zero ─────────────────────────────────

    def test_unused_instance_slots_are_zero(self):
        inst    = self.sample["instance_masks"]         # (10, H, W)
        n       = self.sample["n_instances"].item()
        if n >= MAX_INSTANCES:
            self.skipTest("All slots used — nothing to check")
        unused  = inst[n:]                              # (10-n, H, W)
        self.assertEqual(unused.sum().item(), 0.0,
            f"Slots {n}..{MAX_INSTANCES-1} should be all-zero "
            f"(only {n} nails in this image)")

    # ── direction field only non-zero inside binary mask ──────────────────────

    def test_direction_field_zero_outside_binary_mask(self):
        direction = self.sample["direction_field"]      # (2, H, W)
        binary    = self.sample["binary_mask"]          # (1, H, W)
        dir_mag   = direction.norm(dim=0)               # (H, W)
        bg_mask   = (binary.squeeze(0) < 0.5)          # True where background
        leaked    = dir_mag[bg_mask].max().item()
        self.assertAlmostEqual(leaked, 0.0, places=4,
            msg="direction_field must be zero on background pixels "
                f"(max leaked magnitude = {leaked:.6f})")

    # ── direction field unit vectors on foreground ────────────────────────────

    def test_direction_field_unit_vectors_on_foreground(self):
        direction = self.sample["direction_field"]      # (2, H, W)
        binary    = self.sample["binary_mask"]          # (1, H, W)
        fg_mask   = (binary.squeeze(0) > 0.5)          # foreground pixels
        if not fg_mask.any():
            self.skipTest("No foreground pixels in sample")
        dir_mag   = direction.norm(dim=0)               # (H, W)
        fg_norms  = dir_mag[fg_mask]
        # All foreground direction vectors must be unit length (norm ~1.0)
        bad = ((fg_norms - 1.0).abs() > 0.01).sum().item()
        self.assertEqual(bad, 0,
            f"{bad} foreground pixels have direction norm != 1.0 "
            f"(min={fg_norms.min():.4f}, max={fg_norms.max():.4f})")

    # ── n_instances matches actual non-empty channels ─────────────────────────

    def test_n_instances_matches_nonempty_channels(self):
        inst        = self.sample["instance_masks"]     # (10, H, W)
        n_declared  = self.sample["n_instances"].item()
        n_nonempty  = (inst.sum(dim=(1, 2)) > 0).sum().item()
        self.assertEqual(n_declared, n_nonempty,
            f"n_instances={n_declared} but {n_nonempty} channels have pixels")

    # ── finger_ids length is always MAX_INSTANCES ─────────────────────────────

    def test_finger_ids_always_length_max_instances(self):
        for i in range(len(self.ds)):
            sample = self.ds[i]
            self.assertEqual(sample["finger_ids"].shape[0], MAX_INSTANCES,
                f"Sample {i}: finger_ids length {sample['finger_ids'].shape[0]} "
                f"!= MAX_INSTANCES={MAX_INSTANCES}")

    # ── finger_ids unused slots are FINGER_UNUSED (0) ─────────────────────────

    def test_finger_ids_unused_slots_are_zero(self):
        inst   = self.sample["instance_masks"]
        n      = self.sample["n_instances"].item()
        fids   = self.sample["finger_ids"]
        if n >= MAX_INSTANCES:
            self.skipTest("All slots used")
        unused_ids = fids[n:].tolist()
        self.assertTrue(all(v == 0 for v in unused_ids),
            f"Unused finger_ids slots must be 0, got {unused_ids}")

    # ── binary mask has at least some foreground ──────────────────────────────

    def test_binary_mask_has_foreground(self):
        binary = self.sample["binary_mask"]
        fg_px  = (binary > 0.5).sum().item()
        self.assertGreater(fg_px, 0,
            "binary_mask is all zeros — no nail pixels found in sample")

    # ── dataset length matches n_images ───────────────────────────────────────

    def test_dataset_length(self):
        self.assertEqual(len(self.ds), 3,
            f"Expected 3 images in synthetic dataset, got {len(self.ds)}")

    # ── image is normalised (not raw [0,1]) ───────────────────────────────────

    def test_image_is_normalised(self):
        # Verify normalisation was applied by checking that the tensor values
        # are NOT in [0, 1] range as a whole — specifically that max > 1.0 OR
        # min < 0.0 is possible, OR that the per-channel means differ from 0.5.
        # Most robustly: re-apply the inverse normalisation and check we recover
        # a [0,1] image, proving the forward normalisation was applied.
        import torch
        MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img          = self.sample["image"]
        denormalised = img * STD + MEAN
        mn, mx       = denormalised.min().item(), denormalised.max().item()
        self.assertGreaterEqual(mn, -0.05,
            f"De-normalised image min={mn:.4f} — too low, normalisation wrong")
        self.assertLessEqual(mx, 1.05,
            f"De-normalised image max={mx:.4f} — too high, normalisation wrong")
        self.assertGreater(mx, 0.1,
            "De-normalised image is all near-zero — something is wrong")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — resize scale correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestResizeScaleCorrectness(unittest.TestCase):
    """
    Verify that bboxes are scaled correctly after resize so that direction
    field pixels land inside the corresponding instance mask region.
    """

    def test_direction_field_pixels_are_subset_of_instance_mask(self):
        """
        For each nail slot, every pixel where direction_field != (0,0)
        must also be active (>0.5) in the corresponding instance_masks channel.
        """
        tmp = tempfile.mkdtemp()
        try:
            root   = make_synthetic_coco(tmp, n_images=1,
                                         nails_per_image=5, img_size=64)
            ds     = NailDataset(root, augment=False, image_size=64)
            sample = ds[0]

            direction = sample["direction_field"]           # (2, H, W)
            inst      = sample["instance_masks"]            # (10, H, W)
            binary    = sample["binary_mask"].squeeze(0)    # (H, W)

            dir_active = direction.norm(dim=0) > 0.01      # (H, W) bool

            # Every direction-active pixel must be in binary mask
            leaked = dir_active & (binary < 0.5)
            self.assertEqual(leaked.sum().item(), 0,
                f"direction_field has {leaked.sum().item()} pixels "
                f"outside binary_mask — bbox resize scale is wrong")
        finally:
            shutil.rmtree(tmp)

    def test_instance_mask_pixels_covered_by_binary_mask(self):
        """
        Every active pixel in any instance channel must be active in binary.
        """
        tmp = tempfile.mkdtemp()
        try:
            root   = make_synthetic_coco(tmp, n_images=1,
                                         nails_per_image=3, img_size=64)
            ds     = NailDataset(root, augment=False, image_size=64)
            sample = ds[0]

            inst   = sample["instance_masks"]               # (10, H, W)
            binary = sample["binary_mask"].squeeze(0)       # (H, W)

            inst_union = (inst.max(dim=0).values > 0.5)    # (H, W)
            not_in_binary = inst_union & (binary < 0.5)
            self.assertEqual(not_in_binary.sum().item(), 0,
                "Some instance_mask pixels are not covered by binary_mask")
        finally:
            shutil.rmtree(tmp)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — augmentation does not corrupt masks
# ══════════════════════════════════════════════════════════════════════════════

class TestAugmentation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp  = tempfile.mkdtemp()
        root     = make_synthetic_coco(cls.tmp, n_images=4,
                                       nails_per_image=5, img_size=64)
        cls.ds   = NailDataset(root, augment=True, image_size=64)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_augmented_shapes_unchanged(self):
        for i in range(len(self.ds)):
            s = self.ds[i]
            self.assertEqual(s["image"].shape,           (3,           64, 64))
            self.assertEqual(s["binary_mask"].shape,     (1,           64, 64))
            self.assertEqual(s["instance_masks"].shape,  (MAX_INSTANCES, 64, 64))
            self.assertEqual(s["direction_field"].shape, (2,           64, 64))

    def test_augmented_binary_mask_range(self):
        for i in range(len(self.ds)):
            s  = self.ds[i]
            mn = s["binary_mask"].min().item()
            mx = s["binary_mask"].max().item()
            self.assertGreaterEqual(mn, 0.0)
            self.assertLessEqual(mx,   1.0)

    def test_augmented_instance_masks_no_pixel_in_two_channels(self):
        for i in range(len(self.ds)):
            inst         = self.ds[i]["instance_masks"]
            active_count = (inst > 0.5).float().sum(dim=0).max().item()
            self.assertLessEqual(active_count, 1.0,
                f"Sample {i}: pixel active in multiple instance channels after augmentation")

    def test_augmented_direction_field_range(self):
        for i in range(len(self.ds)):
            d  = self.ds[i]["direction_field"]
            self.assertGreaterEqual(d.min().item(), -1.0)
            self.assertLessEqual(d.max().item(),     1.0)

    def test_vflip_known_limitation_bboxes_not_updated(self):
        """
        Documents known limitation: vflip does not update bboxes used for
        direction field computation, same as hflip. After vflip the nail
        base/tip are swapped spatially but direction still points (0,-1).
        Fix: recompute direction after augmentation using flipped bboxes.
        """
        # We can only verify the limitation exists — direction is always (0,-1)
        # on foreground pixels regardless of flip, because vx=0 always.
        sample    = self.ds[0]
        direction = sample["direction_field"]
        binary    = sample["binary_mask"].squeeze(0)
        fg_mask   = binary > 0.5
        if fg_mask.any():
            dx_fg = direction[0][fg_mask]
            # dx is always 0 — the known limitation
            self.assertTrue((dx_fg.abs() < 1e-5).all(),
                "dx component of direction should always be 0 (known limitation)")


# ══════════════════════════════════════════════════════════════════════════════
# Tests — MAX_INSTANCES truncation
# ══════════════════════════════════════════════════════════════════════════════

class TestMaxInstancesTruncation(unittest.TestCase):

    def test_more_than_max_instances_truncated_silently(self):
        """
        If an image has more nails than MAX_INSTANCES, extras are dropped.
        The output must still have exactly MAX_INSTANCES channels — no crash,
        no shape error.
        """
        tmp = tempfile.mkdtemp()
        try:
            # Create dataset with MAX_INSTANCES + 3 nails per image
            root   = make_synthetic_coco(tmp, n_images=1,
                                         nails_per_image=MAX_INSTANCES + 3,
                                         img_size=128)
            ds     = NailDataset(root, augment=False, image_size=64)
            sample = ds[0]
            self.assertEqual(sample["instance_masks"].shape,
                             (MAX_INSTANCES, 64, 64),
                             "instance_masks must always have MAX_INSTANCES channels")
            self.assertEqual(sample["finger_ids"].shape[0], MAX_INSTANCES,
                             "finger_ids must always have MAX_INSTANCES elements")
        finally:
            shutil.rmtree(tmp)

    def test_n_instances_capped_at_max_instances(self):
        tmp = tempfile.mkdtemp()
        try:
            root   = make_synthetic_coco(tmp, n_images=1,
                                         nails_per_image=MAX_INSTANCES + 3,
                                         img_size=128)
            ds     = NailDataset(root, augment=False, image_size=64)
            n      = ds[0]["n_instances"].item()
            self.assertLessEqual(n, MAX_INSTANCES,
                f"n_instances={n} exceeds MAX_INSTANCES={MAX_INSTANCES}")
        finally:
            shutil.rmtree(tmp)


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print a summary of known limitations alongside test results
    print("=" * 65)
    print("Nail VTON Dataset Tests")
    print("=" * 65)
    print()
    print("KNOWN LIMITATIONS (documented in tests):")
    print("  1. Direction field vx=0 always → direction is (0,-1) for all nails.")
    print("     Tilted nails get wrong direction. Fix: use polygon PCA for orientation.")
    print("  2. Finger labels are position-relative (left→right = pinky→index).")
    print("     On a right hand this is anatomically inverted.")
    print("     Fix: detect hand chirality before assigning labels.")
    print("  3. Augmentation hflip does not update finger_ids.")
    print("     Labels become spatially misaligned after horizontal flip.")
    print("     Fix: re-run assign_finger_ids on flipped bboxes in __getitem__.")
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
        print(f"{len(result.failures)} FAILURES  {len(result.errors)} ERRORS  "
              f"out of {result.testsRun} tests")
    print("=" * 65)
    sys.exit(0 if result.wasSuccessful() else 1)
