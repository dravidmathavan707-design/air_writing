import cv2
import numpy as np

from recognize.detector import recognize_drawing


def _ink():
    return np.zeros((720, 1280, 3), dtype=np.uint8)


def _labels(canvas):
    return {(d.kind, d.label) for d in recognize_drawing(canvas)}


def test_circle_and_square_and_triangle():
    canvas = _ink()
    cv2.circle(canvas, (180, 220), 55, (235, 190, 56), 8)
    cv2.rectangle(canvas, (360, 170), (500, 310), (235, 190, 56), 8)
    pts = np.array([[700, 310], [780, 170], [860, 310]], dtype=np.int32)
    cv2.polylines(canvas, [pts], True, (235, 190, 56), 8)
    labels = {d.label for d in recognize_drawing(canvas) if d.kind == "shape"}
    assert "CIRCLE" in labels or "ELLIPSE" in labels
    assert "SQUARE" in labels or "RECTANGLE" in labels
    assert "TRIANGLE" in labels


def test_plus_symbol():
    canvas = _ink()
    cv2.rectangle(canvas, (200, 250), (360, 280), (235, 190, 56), -1)
    cv2.rectangle(canvas, (255, 195), (305, 335), (235, 190, 56), -1)
    labels = {d.label for d in recognize_drawing(canvas)}
    assert "PLUS" in labels


def test_letter_and_digit():
    canvas = _ink()
    cv2.putText(canvas, "A", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (235, 190, 56), 8, cv2.LINE_AA)
    cv2.putText(canvas, "5", (320, 280), cv2.FONT_HERSHEY_SIMPLEX, 5, (235, 190, 56), 8, cv2.LINE_AA)
    kinds = {(d.kind, d.label) for d in recognize_drawing(canvas)}
    assert ("letter", "A") in kinds
    assert ("digit", "5") in kinds


def test_word_hello_grouping():
    canvas = _ink()
    cv2.putText(canvas, "HI", (90, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (235, 190, 56), 8, cv2.LINE_AA)
    labels = {d.label for d in recognize_drawing(canvas) if d.kind == "word"}
    assert any("H" in lab and "I" in lab for lab in labels) or "HI" in labels


def test_house_object():
    canvas = _ink()
    roof = np.array([[200, 220], [320, 120], [440, 220]], dtype=np.int32)
    cv2.polylines(canvas, [roof], True, (235, 190, 56), 8)
    cv2.rectangle(canvas, (220, 220), (420, 380), (235, 190, 56), 8)
    labels = {d.label for d in recognize_drawing(canvas)}
    assert "HOUSE" in labels


def test_empty_canvas_is_quiet():
    assert recognize_drawing(_ink()) == []
