"""
Simply display the contents of the webcam with optional mirroring using OpenCV
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2
import mediapipe as mp
import utils
import numpy as np
import tensorflow as tf
from collections import Counter
from collections import deque


def show_webcam(mirror=False):
    pose_model = tf.keras.models.load_model("keypoint_classifier.hdf5")
    gesture_model = tf.keras.models.load_model("point_history_classifier.hdf5")

    poses_class_names = ["Open", "Fist", "Point"]
    gesture_class_names = [
        "Stop",
        "Clockwise",
        "Counter Clockwise",
        "Move",
        "Pinch",
        "Unpinch",
    ]

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
    )

    history_length = 16
    point_history = deque(maxlen=history_length)

    # フィンガージェスチャー履歴 ################################################
    finger_gesture_history = deque(maxlen=history_length)

    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)

        x, y, _ = img.shape

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                landmark_list = utils.calc_landmark_list(img, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = utils.pre_process_landmark(
                    landmark_list)

                tmp = pose_model.predict([pre_processed_landmark_list])
                result = np.amax(np.squeeze(tmp))
                pose_class_name = poses_class_names[np.argmax(np.squeeze(tmp))]

                pre_processed_point_history_list = utils.pre_process_point_history(
                    img, point_history
                )

                if pose_class_name == "Point":
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = np.argmax(
                        np.squeeze(
                            gesture_model.predict(
                                np.array(
                                    [pre_processed_point_history_list], dtype=np.float32
                                )
                            )
                        )
                    )

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                gesture_class_name = gesture_class_names[most_common_fg_id[0][0]]

                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                if result < 0.7:
                    pose_class_name = "Other"

                # show the prediction on the frame
                cv2.putText(
                    img,
                    pose_class_name,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    gesture_class_name,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.imshow("my webcam", img)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == "__main__":
    main()
