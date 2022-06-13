from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.scatter import Scatter
from kivy.properties import StringProperty
from kivy.graphics.transformation import Matrix


class ImageWidget(Scatter):
    source = StringProperty(None)

    def __init__(self, **kwargs):
        super(ImageWidget, self).__init__(**kwargs)
        self.do_rotation = False
        # self.do_scale = False
        self.do_translation = False

    def on_touch_down(self, touch):
        # # Override Scatter's `on_touch_down` behavior for mouse scroll
        if touch.button == 'left':
            factor = 1.1
        elif touch.button == 'right':
            factor = 1 / 1.1
        if factor is not None:
            self.apply_transform(Matrix().scale(factor, factor, factor),
                                 anchor=touch.pos)


class Image(App):
    def build(self):
        app = ImageWidget(source="wallpaper.jpg")
        return app


Image().run()
