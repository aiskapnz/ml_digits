# SPDX-License-Identifier: GPL-3.0-or-later
# DigitDraw: GTK app to draw digits and predict them with an SVM.
# Copyright (C) 2026 aiskapnz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import cairo
import gi
from joblib import load
from PIL import Image
from sklearn import svm

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GdkPixbuf, GLib, Gtk


def get_clf() -> svm.SVC:
    import os

    model_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models",
        "sk_learn_digits.joblib",
    )
    return load(model_file)


@Gtk.Template(filename="main_window.ui")
class DrawingAreaWindow(Adw.ApplicationWindow):
    __gtype_name__ = "DrawingAreaWindow"
    root_box: Gtk.Box = Gtk.Template.Child()
    input_label: Gtk.Label = Gtk.Template.Child()
    drawing_area: Gtk.DrawingArea = Gtk.Template.Child()
    preview_image: Gtk.Image = Gtk.Template.Child()
    clear_button: Gtk.Button = Gtk.Template.Child()
    predicted_label: Gtk.Label = Gtk.Template.Child()

    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app)

        self._clf = get_clf()

        self._surface: cairo.ImageSurface | None = None
        self._last_point: tuple[float, float] | None = None

        self.drawing_area.set_draw_func(self._on_draw)
        self.drawing_area.connect("resize", self._on_resize)

        self.clear_button.connect("clicked", self._on_clear)

        drag = Gtk.GestureDrag()
        drag.set_button(1)
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self.drawing_area.add_controller(drag)

        self.set_predicted_digit(None)
        self._prediction_pending = False

    def clear(self):
        if self._surface is None:
            return

        context = cairo.Context(self._surface)
        context.set_source_rgb(1.0, 1.0, 1.0)
        context.paint()
        self._surface.flush()
        self.drawing_area.queue_draw()

        # to clear predicted label
        self.update_prediction()

    def get_image_8x8(self) -> Image.Image | None:
        if self._surface is None:
            return

        width = self._surface.get_width()
        height = self._surface.get_height()
        data = bytes(self._surface.get_data())
        return resize_to_8x8(data, width, height)

    def update_prediction(self):
        def _update_prediction():
            image_8x8 = self.get_image_8x8()
            if image_8x8 is None:
                return

            self.update_preview_image(image_8x8)

            # to grayscale
            grayscale_image_8x8 = image_8x8.convert("L")

            # convert image data for svm.SCV
            floats = [(~b & 0xFF) // 16.0 for b in grayscale_image_8x8.tobytes()]

            if not all(f == 0 for f in floats):
                predicted_digit = self._clf.predict([floats])
                self.set_predicted_digit(predicted_digit[0])
            else:
                self.set_predicted_digit(None)

            self._prediction_pending = False

        if not self._prediction_pending:
            self._prediction_pending = True
            GLib.idle_add(_update_prediction)

    def update_preview_image(self, image: Image.Image):
        self.preview_image.set_from_pixbuf(preview_pixbuf(image))

    def set_predicted_digit(self, predicted_digit: int | None):
        label = "..." if predicted_digit is None else f"{predicted_digit}"

        self.predicted_label.set_label(label)

    def _draw_line(self, x: float, y: float) -> None:
        if self._surface is None:
            return

        if self._last_point is None:
            self._last_point = (x, y)
            return

        context = cairo.Context(self._surface)
        context.set_source_rgb(0.0, 0.0, 0.0)
        context.set_line_width(20.0)
        context.set_line_cap(cairo.LineCap.ROUND)
        context.move_to(*self._last_point)
        context.line_to(x, y)
        context.stroke()
        self._last_point = (x, y)

    def _on_drag_begin(
        self, _gesture: Gtk.GestureDrag, start_x: float, start_y: float
    ) -> None:
        self._last_point = (start_x, start_y)
        self._draw_line(start_x, start_y)
        self.drawing_area.queue_draw()

    def _on_drag_update(
        self, gesture: Gtk.GestureDrag, _offset_x: float, _offset_y: float
    ) -> None:
        _has_start, start_x, start_y = gesture.get_start_point()
        _has_offset, offset_x, offset_y = gesture.get_offset()
        self._draw_line(start_x + offset_x, start_y + offset_y)
        self.drawing_area.queue_draw()
        self.update_prediction()

    def _on_drag_end(
        self, _gesture: Gtk.GestureDrag, _offset_x: float, _offset_y: float
    ) -> None:
        self._last_point = None
        self.update_prediction()

    def _on_clear(self, user_data):
        self.clear()

    def _on_resize(self, _area: Gtk.DrawingArea, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.clear()

    def _on_draw(
        self,
        _area: Gtk.DrawingArea,
        context: cairo.Context,
        _width: int,
        _height: int,
    ) -> None:
        if self._surface is None:
            return

        context.set_source_surface(self._surface, 0, 0)
        context.paint()


class DrawingApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="com.example.DrawingApp")

    def do_activate(self) -> None:
        window = self.props.active_window
        if window is None:
            window = DrawingAreaWindow(self)
        window.present()


def resize_to_8x8(data: bytes, width: int, height: int) -> Image.Image:
    # L = 8-bit grayscale (A8)
    pil = Image.frombytes("RGBA", (width, height), data)
    return pil.resize((8, 8), Image.Resampling.LANCZOS)


def preview_pixbuf(image: Image.Image) -> GdkPixbuf.Pixbuf | None:
    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(image.tobytes()),
        GdkPixbuf.Colorspace.RGB,
        True,
        8,
        image.width,
        image.height,
        image.width * 4,
    )
    # todo: scale width, height as args
    return pixbuf.scale_simple(100, 100, GdkPixbuf.InterpType.TILES)


def main() -> None:
    app = DrawingApp()
    app.run()


if __name__ == "__main__":
    main()
