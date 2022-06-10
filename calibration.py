from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.vector import Vector
import cv2
import mediapipe as mp


cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)


class MovingCircle(Widget):
    def move(self):
        self.pos = self.pos


class CalibrationApp(Widget):
    user_hand = ObjectProperty(None)

    def set_user_hand(self, center):
        self.user_hand.center = center

    def update(self, dt):
        # Get movement of user hand and move circle accordingly
        _, img = cam.read()
        img = cv2.flip(img, -1)
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                center = [hand_landmarks.landmark[9].x * self.width,
                          hand_landmarks.landmark[9].y * self.height]
                self.set_user_hand(center=center)

        # self.user_hand.move()


class Calibration(App):
    def build(self):
        app = CalibrationApp()
        Clock.schedule_interval(app.update, 1.0 / 30.0)
        return app


Calibration().run()
