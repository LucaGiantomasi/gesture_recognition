from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scatter import Scatter
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
from kivy.clock import Clock
import cv2
import mediapipe as mp
import json
import tensorflow as tf
from collections import Counter
from collections import deque
import utils
import numpy as np

cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode="store_true", max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.7
)
pose_model = tf.keras.models.load_model("keypoint_classifier.hdf5")
gesture_model = tf.keras.models.load_model("point_history_classifier.hdf5")

poses_class_names = ["Open", "Fist", "Point"]
gesture_class_names = [
    "Stop",
    "Clockwise",
    "Counter Clockwise",
    "Move",
    "Pinch",
    "Unpinch"
]
history_length = 16
point_history = deque(maxlen=history_length)
finger_gesture_history = deque(maxlen=history_length)

class MovingCircle(Widget):
    pass


class DetailRectangle(Widget):
    source = StringProperty(None)
    text = StringProperty(None)


class ImageWidget(Scatter):
    source = StringProperty(None)

    def __init__(self, **kwargs):
        super(ImageWidget, self).__init__(**kwargs)
        self.do_rotation = False
        self.do_scale = False
        self.do_translation = False


class MainWindow(Screen):
    user_hand = ObjectProperty(None)
    index = NumericProperty(0)
    length = NumericProperty(0)
    start_pos_x = NumericProperty(0)
    stop_frame = NumericProperty(0)

    def __init__(self, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        self.set_image(0)

    def update(self):
        # Get movement of user hand and move circle accordingly
        ret, img = cam.read()
        if not ret:
            return
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False
        results = hands.process(img)
        img.flags.writeable = True

        if self.stop_frame < 8:
            self.stop_frame += 1

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                center = [hand_landmarks.landmark[9].x * self.width * 1.7,
                          (1 - hand_landmarks.landmark[9].y) * self.height * 1.7]
                self.user_hand.center = center

                landmark_list = utils.calc_landmark_list(
                    img, hand_landmarks)

################################################################################################################################################

                # POSE RECOGNITION
                pre_processed_landmark_list = utils.pre_process_landmark(
                    landmark_list)

                tmp = pose_model.predict([pre_processed_landmark_list])
                result = np.amax(np.squeeze(tmp))
                pose_class_name = poses_class_names[np.argmax(
                    np.squeeze(tmp))]

                if result < 0.7:
                    pose_class_name = "Other"

                if pose_class_name == "Fist":
                    # Set start position of fist
                    if self.start_pos_x == 0 and self.stop_frame >= 8:
                        self.start_pos_x = center[0]

                    if self.start_pos_x != 0:
                        # If fist moved to the right get image before
                        if (center[0] - self.start_pos_x) >= 200:
                            self.index = (
                                self.index + self.length - 1) % self.length
                            self.set_image(self.index)
                            self.start_pos_x = 0
                            self.stop_frame = 0
                        # If fist moved to the left get image after
                        elif (center[0] - self.start_pos_x) <= -200:
                            self.index = (self.index + 1) % self.length
                            self.set_image(self.index)
                            self.start_pos_x = 0
                            self.stop_frame = 0
                else:
                    self.start_pos_x = 0

################################################################################################################################################

                # GESTURE RECOGNITION
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
                if gesture_class_name == "Pinch":
                    # Check if it is on a detail
                    for child in self.children:
                        if isinstance(child, DetailRectangle):
                            if self.user_hand.collide_widget(child):
                                app = App.get_running_app()
                                app.root.current = "second"
                                app.root.transition.direction = "left"
                                # Set image and text of second window
                                app.root.second_window.source = "images/" + child.source
                                app.root.second_window.text = child.text
                                # Clear all queues
                                point_history.clear()
                                finger_gesture_history.clear()


    def set_image(self, index):
        with open('config/image_configs.json') as f:
            data = json.load(f)
            if self.length == 0:
                self.length = len(data)
            data = data[index]

            for child in self.children[:]:
                if isinstance(child, DetailRectangle) or isinstance(child, ImageWidget):
                    self.remove_widget(child)

            for detail in data["details"]:
                self.add_widget(
                    DetailRectangle(size_hint=detail['dimension'], pos_hint={'x': detail['start_position'][0], 'y': detail['start_position'][1]}, source=detail['source'], text=detail['text']))
            # Add image always in background
            self.add_widget(ImageWidget(
                size=self.size, source="images/" + data['source']), 99)
            point_history.clear()
            finger_gesture_history.clear()


class SecondWindow(Screen):
    source = StringProperty(None)
    text = StringProperty("")

    def update(self):
        _, img = cam.read()
        if img is not None:
            img = cv2.flip(img, -1)
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img)

            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_list = utils.calc_landmark_list(
                        img, hand_landmarks)

                    # POSE RECOGNITION
                    pre_processed_landmark_list = utils.pre_process_landmark(
                        landmark_list)

                    tmp = pose_model.predict([pre_processed_landmark_list])
                    result = np.amax(np.squeeze(tmp))
                    pose_class_name = poses_class_names[np.argmax(
                        np.squeeze(tmp))]

                    if result < 0.7:
                        pose_class_name = "Other"

                    # GESTURE RECOGNITION
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
                    if gesture_class_name == "Unpinch":
                        app = App.get_running_app()
                        app.root.current = "main"
                        app.root.transition.direction = "right"
                        point_history.clear()
                        finger_gesture_history.clear()


class WindowManager(ScreenManager):
    def update(self, dt):
        if self.current == "main":
            return self.main_window.update()
        else:
            return self.second_window.update()


class Main(App):
    def build(self):
        app = WindowManager()
        Clock.schedule_interval(app.update, 1.0 / 30.0)
        return app


Main().run()
