# from kivy.graphics.transformation import Matrix
from kivy.app import App
from kivy.uix.scatter import Scatter
from kivy.properties import StringProperty
from kivy.properties import ObjectProperty
from kivy.uix.widget import Widget
from kivy.clock import Clock
import cv2
import mediapipe as mp


cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5
)


class MovingCircle(Widget):
    pass


class ImageWidget(Scatter):
    source = StringProperty(None)

    def __init__(self, **kwargs):
        super(ImageWidget, self).__init__(**kwargs)
        self.do_rotation = False
        self.do_scale = False
        self.do_translation = False

    # def on_touch_down(self, touch):
    #     # # Override Scatter's `on_touch_down` behavior for mouse scroll
    #     if touch.button == 'left':
    #         factor = 1.1
    #     elif touch.button == 'right':
    #         factor = 1 / 1.1
    #     if factor is not None:
    #         self.apply_transform(Matrix().scale(factor, factor, factor),
    #                              anchor=touch.pos)


class ImageApp(Widget):
    source = StringProperty(None)
    user_hand = ObjectProperty(None)

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


class Image(App):
    def build(self):
        app = ImageApp(source="wallpaper.jpg")
        Clock.schedule_interval(app.update, 1.0 / 30.0)
        return app


Image().run()
