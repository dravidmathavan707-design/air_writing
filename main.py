from __future__ import annotations

import ctypes
import time

import cv2
import numpy as np

from animation.animation_manager import EFFECTS, AnimationManager
from camera.camera import Camera
from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_LAPTOP,
    CAMERA_WEBCAM,
    CAMERA_WIDTH,
    CYAN,
    DRAW_COLOR,
    DRAW_THICKNESS,
    GREEN,
    HINT,
    MODE_COLOR,
    MOTTO,
    MUTED,
    PANEL,
    RED,
    STATUS_COLOR,
    TEXT,
    TEXT_OUTLINE,
    TITLE,
    WELCOME,
    WINDOW_NAME,
)
from body.body_detector import FaceGestureCycle, any_hand_is_fist, any_hand_is_open
from face.face_detector import FaceDetector
from face.face_landmarks import bbox_from_points
from face.face_renderer import draw_face_portrait_from_points
from face.hair_detector import HairDetector
from hand.hand_detector import HandDetector
from capture.screen_capture import ScreenCapture
from hero.hero_manager import HeroManager
from hero.heroes import HERO_ORDER
from recognize.detector import DrawingRecognizer, draw_detections

SMOOTHING = 0.70
MIN_DRAW_DISTANCE = 1
MAX_INTERPOLATION_DISTANCE = 160
FINGER_TIP_RADIUS = 9
GESTURE_HOLD_FRAMES = 3

MODE_FINGER = "finger"
MODE_FACE = "face"
MODE_ANIMATION = "animation"
MODE_RECOGNIZE = "recognize"
MODE_HERO = "hero"

MODE_BUTTONS = [
    (MODE_FINGER, (20, 118, 128, 163), "FINGER"),
    (MODE_FACE, (136, 118, 230, 163), "FACE"),
    (MODE_ANIMATION, (238, 118, 378, 163), "ANIMATION"),
    (MODE_RECOGNIZE, (386, 118, 536, 163), "RECOGNIZE"),
    (MODE_HERO, (544, 118, 668, 163), "HERO"),
]
HERO_BUTTONS = [
    ("iron", (20, 173, 118, 210), "IRON"),
    ("web", (126, 173, 218, 210), "WEB"),
    ("tech", (226, 173, 318, 210), "TECH"),
    ("energy", (326, 173, 448, 210), "ENERGY"),
    ("cosmic", (456, 173, 578, 210), "COSMIC"),
    ("ninja", (586, 173, 700, 210), "NINJA"),
]
EFFECT_BUTTONS = [
    ("energy_sphere", (20, 173, 155, 210), "SPHERE"),
    ("wind_spiral", (165, 173, 310, 210), "SHURIKEN"),
    ("magic_portal", (320, 173, 455, 210), "PORTAL"),
    ("energy_blast", (465, 173, 585, 210), "BLAST"),
    ("chidori", (595, 173, 740, 210), "CHIDORI"),
]
FULL_BUTTON = (1100, 118, 1260, 163)
CAM_BUTTON = (920, 118, 1085, 163)

app_mode = MODE_FINGER
face_diagram_locked = False
face_scanning = False
pending_faces = []
pending_frame = None
gesture_cycle = FaceGestureCycle(hold_frames=GESTURE_HOLD_FRAMES)
animation_manager = AnimationManager()
hero_manager = HeroManager()
drawing_recognizer = DrawingRecognizer()
screen_capture = ScreenCapture()
fullscreen = False
camera_index = CAMERA_INDEX
pending_camera_switch = False


def distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def _blend_rect(frame, x1, y1, x2, y2, color=PANEL, alpha=0.72):
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    fill = np.full_like(roi, color)
    frame[y1:y2, x1:x2] = cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0)


def draw_text(frame, text, position, scale=0.65, color=TEXT, thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = position
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        TEXT_OUTLINE,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_continuous_line(canvas, previous, current):
    if previous is None:
        return
    movement = distance(previous, current)
    if movement <= MAX_INTERPOLATION_DISTANCE:
        cv2.line(canvas, previous, current, DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)
        return
    steps = max(2, int(movement / 8))
    for step in range(steps):
        start_ratio = step / steps
        end_ratio = (step + 1) / steps
        start = (
            int(previous[0] + (current[0] - previous[0]) * start_ratio),
            int(previous[1] + (current[1] - previous[1]) * start_ratio),
        )
        end = (
            int(previous[0] + (current[0] - previous[0]) * end_ratio),
            int(previous[1] + (current[1] - previous[1]) * end_ratio),
        )
        cv2.line(canvas, start, end, DRAW_COLOR, DRAW_THICKNESS, cv2.LINE_AA)


def _hit(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def reset_face_state():
    global face_diagram_locked, face_scanning, pending_faces, pending_frame
    face_diagram_locked = False
    face_scanning = False
    pending_faces = []
    pending_frame = None
    gesture_cycle.reset()


def _display_size():
    if fullscreen:
        try:
            screen_w = ctypes.windll.user32.GetSystemMetrics(0)
            screen_h = ctypes.windll.user32.GetSystemMetrics(1)
            if screen_w > 1 and screen_h > 1:
                return screen_w, screen_h
        except Exception:
            pass
    try:
        _wx, _wy, win_w, win_h = cv2.getWindowImageRect(WINDOW_NAME)
        if win_w > 1 and win_h > 1:
            return win_w, win_h
    except cv2.error:
        pass
    return CAMERA_WIDTH, CAMERA_HEIGHT


def _map_mouse(x, y):
    """Map window clicks onto the 1280x720 UI, including fullscreen stretch."""
    disp_w, disp_h = _display_size()
    if disp_w == CAMERA_WIDTH and disp_h == CAMERA_HEIGHT:
        return x, y
    return int(x * CAMERA_WIDTH / disp_w), int(y * CAMERA_HEIGHT / disp_h)


def toggle_fullscreen():
    global fullscreen
    fullscreen = not fullscreen
    try:
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
        )
        if not fullscreen:
            cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)
    except cv2.error:
        pass
    print("FULLSCREEN" if fullscreen else "WINDOW")


def mouse_callback(event, x, y, flags, param):
    global app_mode, pending_camera_switch
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x, y = _map_mouse(x, y)
    if _hit(x, y, CAM_BUTTON):
        pending_camera_switch = True
        return
    if _hit(x, y, FULL_BUTTON):
        toggle_fullscreen()
        return
    for mode, box, _label in MODE_BUTTONS:
        if _hit(x, y, box):
            app_mode = mode
            reset_face_state()
            animation_manager.reset()
            hero_manager.reset()
            print("MODE:", mode.upper())
            return
    if app_mode == MODE_ANIMATION:
        for name, box, _label in EFFECT_BUTTONS:
            if _hit(x, y, box):
                animation_manager.set_effect(name)
                print("EFFECT:", name)
                return
    if app_mode == MODE_HERO:
        for name, box, _label in HERO_BUTTONS:
            if _hit(x, y, box):
                hero_manager.set_hero(name)
                print("HERO:", name)
                return


def drawable_faces(faces):
    drawn = []
    for face in faces or []:
        points = face.get("points")
        if points is not None and len(points) >= 468:
            drawn.append(face)
    return drawn


def draw_faces_on_canvas(canvas, faces, hair_detector=None, source_frame=None, thickness=2):
    canvas.fill(0)
    for face in drawable_faces(faces):
        hair = None
        if hair_detector is not None and source_frame is not None:
            bbox = face.get("bbox") or bbox_from_points(face["points"])
            hair = hair_detector.extract_contour(source_frame, face_bbox=bbox)
        draw_face_portrait_from_points(
            canvas,
            face["points"],
            color=DRAW_COLOR,
            thickness=thickness,
            hair_contour=hair,
        )


def lock_scanned_faces(drawing, hair_detector):
    global face_diagram_locked, face_scanning, pending_faces
    faces = drawable_faces(pending_faces)
    if not faces:
        return False
    draw_faces_on_canvas(
        drawing,
        faces,
        hair_detector=hair_detector,
        source_frame=pending_frame,
        thickness=3,
    )
    face_diagram_locked = True
    face_scanning = False
    gesture_cycle.state = FaceGestureCycle.AWAIT_CLOSE
    gesture_cycle.close_hold = 0
    gesture_cycle.open_hold = 0
    return True


def _draw_button(frame, box, label, active, color=None):
    x1, y1, x2, y2 = box
    color = color or (GREEN if active else CYAN)
    _blend_rect(frame, x1, y1, x2, y2, PANEL, 0.78 if active else 0.68)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if active:
        cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), color, 1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    tx = x1 + max(4, (x2 - x1 - tw) // 2)
    ty = y1 + (y2 - y1 + th) // 2
    draw_text(frame, label, (tx, ty), 0.48, color, 2)


def main():
    global face_diagram_locked, face_scanning, pending_faces, pending_frame
    global app_mode, pending_camera_switch, camera_index

    camera = Camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    face_detector = FaceDetector()
    hair_detector = HairDetector()
    hand_detector = HandDetector(max_num_hands=2)
    drawing = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    previous_point = None
    smooth_previous = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    try:
        while True:
            if pending_camera_switch:
                pending_camera_switch = False
                nxt = CAMERA_WEBCAM if camera_index == CAMERA_LAPTOP else CAMERA_LAPTOP
                if camera.switch(nxt):
                    camera_index = nxt
                    hand_detector.reset_draw_state()
                    previous_point = None
                    smooth_previous = None
            success, frame = camera.read()
            if not success:
                print("ERROR: Camera frame unavailable")
                break

            height, width = frame.shape[:2]
            hand_results = hand_detector.detect(frame)
            hands = hand_detector.extract(hand_results, frame.shape)
            if app_mode not in (MODE_ANIMATION, MODE_HERO):
                hand_detector.draw_landmarks(frame, hand_results)
            hand_count = len(hands)

            faces = []
            face_count = 0
            face_detected = False
            if app_mode in (MODE_FACE, MODE_HERO):
                face_results = face_detector.detect(frame)
                faces = face_detector.extract_faces(face_results, frame.shape)
                face_count = len(faces)
                face_detected = face_count > 0

            status = "READY"

            if app_mode == MODE_FACE:
                hand_open = any_hand_is_open(hands)
                hand_closed = any_hand_is_fist(hands) and not hand_open
                event = gesture_cycle.update(hand_open, hand_closed)

                if event == "scan":
                    face_scanning = True
                    if faces:
                        pending_faces = faces
                        pending_frame = frame.copy()
                elif event == "draw":
                    face_scanning = False
                    if lock_scanned_faces(drawing, hair_detector):
                        count = len(drawable_faces(pending_faces))
                        status = (
                            f"DREW {count} FACE - CLOSE THEN OPEN YOUR HAND"
                            if count == 1
                            else f"DREW {count} FACES - CLOSE THEN OPEN YOUR HAND"
                        )
                    else:
                        status = "NO FACE TO DRAW - OPEN HAND TO SCAN"
                        gesture_cycle.reset()
                elif event == "need_open":
                    status = "HAND CLOSED - NOW OPEN YOUR HAND"
                elif event == "ready":
                    face_scanning = False
                    status = "READY - OPEN HAND TO SCAN AGAIN"

                cycle_state = gesture_cycle.state
                if event not in ("draw", "need_open", "ready"):
                    if cycle_state == FaceGestureCycle.AWAIT_CLOSE:
                        count = len(drawable_faces(pending_faces))
                        status = (
                            f"FACES DRAWN ({count}) - CLOSE YOUR HAND"
                            if count != 1
                            else "FACE DRAWN - CLOSE YOUR HAND"
                        )
                    elif cycle_state == FaceGestureCycle.AWAIT_OPEN:
                        status = "NOW OPEN YOUR HAND TO CONTINUE"
                    elif cycle_state == FaceGestureCycle.SCANNING:
                        face_scanning = True
                        if faces:
                            pending_faces = faces
                            pending_frame = frame.copy()
                        status = (
                            f"OPEN HAND: {face_count} FACE FOUND - CLOSE HAND TO DRAW"
                            if face_count == 1
                            else f"OPEN HAND: {face_count} FACES FOUND - CLOSE HAND TO DRAW"
                        )
                        if not face_diagram_locked:
                            draw_faces_on_canvas(
                                drawing,
                                faces or pending_faces,
                                thickness=1,
                            )
                    elif face_diagram_locked:
                        count = len(drawable_faces(pending_faces))
                        status = (
                            f"LOCKED {count} FACE"
                            if count == 1
                            else f"LOCKED {count} FACES"
                        )
                    elif not face_detected:
                        status = "NO FACE - OPEN HAND TO SCAN"
                    else:
                        status = (
                            f"{face_count} FACE ON SCREEN - OPEN HAND TO SCAN"
                            if face_count == 1
                            else f"{face_count} FACES ON SCREEN - OPEN HAND TO SCAN"
                        )
                        draw_faces_on_canvas(drawing, faces, thickness=1)

                previous_point = None
                smooth_previous = None
                hand_detector.reset_draw_state()

            elif app_mode == MODE_ANIMATION:
                drawing_hand = hand_detector.select_drawing_hand(hands) if hands else None
                pose = animation_manager.update(drawing_hand)
                frame = animation_manager.render(frame, pose)
                status = animation_manager.status_text()
                previous_point = None
                smooth_previous = None
                hand_detector.reset_draw_state()

            elif app_mode == MODE_HERO:
                frame = hero_manager.render(frame, faces)
                status = hero_manager.status_text(face_detected, face_count)
                previous_point = None
                smooth_previous = None
                hand_detector.reset_draw_state()

            elif app_mode in (MODE_FINGER, MODE_RECOGNIZE):
                drawing_hand = hand_detector.select_drawing_hand(hands)
                can_draw = hand_detector.update_draw_state(drawing_hand)
                if drawing_hand is None:
                    if can_draw:
                        status = "FINGER DRAWING"
                    else:
                        status = "NO HAND"
                        previous_point = None
                        smooth_previous = None
                        hand_detector.tip_tracker.reset()
                else:
                    current_point = hand_detector.tip_tracker.update(
                        drawing_hand.tip_xy or drawing_hand.index_tip
                    )
                    cursor_color = DRAW_COLOR if can_draw else CYAN
                    cv2.circle(frame, current_point, FINGER_TIP_RADIUS, cursor_color, -1)
                    if not can_draw:
                        cv2.circle(
                            frame,
                            current_point,
                            FINGER_TIP_RADIUS + 6,
                            cursor_color,
                            2,
                            cv2.LINE_AA,
                        )
                    if can_draw:
                        status = "FINGER DRAWING"
                        if (
                            smooth_previous is not None
                            and distance(smooth_previous, current_point)
                            >= MIN_DRAW_DISTANCE
                        ):
                            draw_continuous_line(
                                drawing,
                                smooth_previous,
                                current_point,
                            )
                        smooth_previous = current_point
                        previous_point = current_point
                    else:
                        status = "HAND DETECTED - POINT INDEX TO DRAW"
                        previous_point = None
                        smooth_previous = None

            if app_mode not in (MODE_ANIMATION, MODE_HERO):
                mask = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY) > 0
                frame[mask] = drawing[mask]
            if app_mode == MODE_RECOGNIZE:
                drawing_recognizer.update(drawing, time.perf_counter())
                draw_detections(frame, drawing_recognizer.detections)
                status = drawing_recognizer.summary()

            _blend_rect(frame, 0, 0, width, 112, PANEL, 0.74)
            _blend_rect(frame, 0, height - 128, width, height, PANEL, 0.78)
            extra_row = app_mode in (MODE_ANIMATION, MODE_HERO)
            _blend_rect(frame, 12, 108, 752, 220 if extra_row else 174, PANEL, 0.58)

            draw_text(frame, "AIR DRAWING AI", (25, 36), 0.78, TITLE, 2, cv2.FONT_HERSHEY_COMPLEX)
            draw_text(frame, "WELCOME TO CCE_NIT", (620, 36), 0.58, WELCOME, 2, cv2.FONT_HERSHEY_DUPLEX)
            draw_text(
                frame,
                "Start your journey at NIT, turn your ideas into innovation,",
                (25, 70),
                0.58,
                MOTTO,
                2,
                cv2.FONT_HERSHEY_DUPLEX,
            )
            draw_text(
                frame,
                "and create a future beyond imagination.",
                (25, 102),
                0.58,
                MOTTO,
                2,
                cv2.FONT_HERSHEY_DUPLEX,
            )

            for mode, box, label in MODE_BUTTONS:
                _draw_button(frame, box, label, app_mode == mode)
            cam_label = "WEB" if camera_index == CAMERA_WEBCAM else "LAPTOP"
            _draw_button(frame, CAM_BUTTON, cam_label, camera_index == CAMERA_WEBCAM)
            _draw_button(
                frame,
                FULL_BUTTON,
                "WINDOW" if fullscreen else "FULL",
                fullscreen,
            )
            if app_mode == MODE_ANIMATION:
                for name, box, label in EFFECT_BUTTONS:
                    _draw_button(frame, box, label, animation_manager.active_effect == name)
            if app_mode == MODE_HERO:
                for name, box, label in HERO_BUTTONS:
                    _draw_button(frame, box, label, hero_manager.style == name)

            if app_mode == MODE_FACE:
                instruction = "FACE: OPEN=SCAN  CLOSE=DRAW  THEN CLOSE + OPEN"
            elif app_mode == MODE_ANIMATION:
                if animation_manager.active_effect == "wind_spiral":
                    instruction = "RASENSHURIKEN: FIST=SPIN CHARGE   OPEN PALM=RELEASE"
                elif animation_manager.active_effect == "chidori":
                    instruction = "CHIDORI: FIST=LIGHTNING BLADE   OPEN=SENBON / SPEAR / STREAM"
                else:
                    instruction = "ANIMATION: FIST=CHARGE   OPEN PALM=RELEASE"
            elif app_mode == MODE_RECOGNIZE:
                instruction = "RECOGNIZE: DRAW LETTERS DIGITS SHAPES SYMBOLS WORDS"
            elif app_mode == MODE_HERO:
                instruction = "HERO: MASK FITS EVERY FACE ON SCREEN  1-6 SELECT"
            else:
                instruction = "FINGER: POINT INDEX, CURL OTHER FINGERS"
            draw_text(frame, instruction, (20, height - 88), 0.50, HINT, 1, cv2.FONT_HERSHEY_SIMPLEX)

            status_y = height - 58
            draw_text(
                frame,
                f"HANDS: {hand_count}/2",
                (25, status_y),
                0.60,
                GREEN if hand_count else RED,
                2,
            )
            if app_mode in (MODE_FACE, MODE_HERO):
                draw_text(
                    frame,
                    f"FACES: {face_count}",
                    (200, status_y),
                    0.60,
                    GREEN if face_detected else RED,
                    2,
                )
            mode_label = {
                MODE_FINGER: "FINGER MODE",
                MODE_FACE: "FACE MODE",
                MODE_ANIMATION: "ANIMATION MODE",
                MODE_RECOGNIZE: "RECOGNIZE MODE",
                MODE_HERO: "HERO MODE",
            }[app_mode]
            draw_text(frame, f"MODE: {mode_label}", (360, status_y), 0.58, MODE_COLOR, 2, cv2.FONT_HERSHEY_DUPLEX)
            draw_text(
                frame,
                f"STATUS: {status}",
                (620, status_y),
                0.50,
                STATUS_COLOR,
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )
            draw_text(
                frame,
                "C = CLEAR    K = CAMERA    F11 = FULLSCREEN    P = SHOT    V = RECORD    Q = QUIT",
                (25, height - 24),
                0.48,
                MUTED,
                1,
                cv2.FONT_HERSHEY_SIMPLEX,
            )

            screen_capture.write(frame)

            cv2.imshow(WINDOW_NAME, frame)
            raw = cv2.waitKeyEx(1)
            key = raw & 0xFF
            f11 = raw > 255 and ((raw & 0xFFFF) == 0x7A or (raw >> 16) == 0x7A)

            if key == 27 and fullscreen:
                toggle_fullscreen()
            elif key == ord("q"):
                break
            elif f11:
                toggle_fullscreen()
            elif key == ord("c"):
                drawing.fill(0)
                reset_face_state()
                animation_manager.reset()
                hero_manager.reset()
                drawing_recognizer.reset()
                previous_point = None
                smooth_previous = None
                hand_detector.reset_draw_state()
                print("Drawing cleared.")
            elif key == ord("k"):
                pending_camera_switch = True
            elif key == ord("p"):
                screen_capture.screenshot(frame, time.perf_counter())
            elif key == ord("v"):
                screen_capture.toggle_record(frame)
            elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
                idx = key - ord("1")
                if app_mode == MODE_HERO:
                    if idx < len(HERO_ORDER):
                        hero_manager.set_hero(HERO_ORDER[idx])
                elif key != ord("6"):
                    app_mode = MODE_ANIMATION
                    animation_manager.set_effect(EFFECTS[idx])
            elif key == ord("h"):
                app_mode = MODE_HERO
                hero_manager.reset()
                print("MODE: HERO")
            elif key == ord("r"):
                app_mode = MODE_RECOGNIZE
                drawing_recognizer.update(drawing, time.perf_counter(), force=True)
            elif key == ord("f"):
                app_mode = MODE_FACE
                pending_faces = faces
                pending_frame = frame.copy()
                if lock_scanned_faces(drawing, hair_detector):
                    print(f"Drew {len(drawable_faces(pending_faces))} face(s).")
                else:
                    print("No face to draw.")
            elif key == ord("s"):
                cv2.imwrite("air_drawing.png", drawing if app_mode != MODE_ANIMATION else frame)
                print("Saved: air_drawing.png")

    finally:
        screen_capture.close()
        camera.release()
        hand_detector.close()
        face_detector.close()
        hair_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
