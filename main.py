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

from __future__ import annotations

import os
import sys
from multiprocessing import Pipe, Process, connection
from threading import Thread

import cairo
import gi
import numpy as np
import openvino as ov
import tensorflow as tf
from joblib import load
from PIL import Image
from sklearn import svm

import digits_display

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import (  # type: ignore[import-not-found]  # noqa: E402
    Adw,  # pyright: ignore[reportMissingModuleSource]
    Gdk,  # pyright: ignore[reportMissingModuleSource]
    GLib,  # pyright: ignore[reportMissingModuleSource]
    Gtk,  # pyright: ignore[reportMissingModuleSource]
)


@Gtk.Template(filename="main_window.ui")
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = "MainWindow"
    root_box: Gtk.Box = Gtk.Template.Child()
    input_label: Gtk.Label = Gtk.Template.Child()
    drawing_area: Gtk.DrawingArea = Gtk.Template.Child()
    sklearn_preview_image: Gtk.Image = Gtk.Template.Child()
    clear_button: Gtk.Button = Gtk.Template.Child()
    sklearn_predicted_label: Gtk.Label = Gtk.Template.Child()
    tf_predicted_label: Gtk.Label = Gtk.Template.Child()
    tf_preview_image: Gtk.Image = Gtk.Template.Child()
    tf_digits_display: digits_display.DigitsDisplay = Gtk.Template.Child()

    def __init__(self, app: DrawingApp) -> None:
        super().__init__(application=app)
        self.app = app

        self._surface: cairo.ImageSurface | None = None
        self._last_point: tuple[float, float] | None = None

        self.drawing_area.set_draw_func(self._on_draw)
        self.drawing_area.connect("resize", self._on_resize)
        self.drawing_area.set_cursor(get_drawing_cursor())

        self.clear_button.connect("clicked", self._on_clear)

        drag = Gtk.GestureDrag()
        drag.set_button(1)
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self.drawing_area.add_controller(drag)

        self._on_prediction(None)
        self._prediction_pending = False

        self.is_waiting_result = True
        self._result_waiter_thread = Thread(target=self.result_waiter, daemon=True)
        self._result_waiter_thread.start()

    def do_close_request(self) -> bool:
        self.is_waiting_result = False
        return Adw.ApplicationWindow.do_close_request(self)

    def result_waiter(self):
        while self.is_waiting_result:
            result: tuple[SKLearnResult, TFResult] | None = self.app.worker_conn.recv()
            if result is None:
                break
            GLib.idle_add(self._on_prediction, result)

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

    def get_image_data(self) -> tuple[bytes, tuple[int, int]] | None:
        if self._surface is None:
            return

        width = self._surface.get_width()
        height = self._surface.get_height()
        data = bytes(self._surface.get_data())
        return (data, (width, height))

    def update_prediction(self):
        if not self._prediction_pending:
            self._prediction_pending = True
            image_data = self.get_image_data()

            if image_data is None:
                self._prediction_pending = False
                return

            data, size = image_data
            image = Image.frombytes("RGBA", size, data)

            # send image to process
            self.app.worker_conn.send(image)

    def _on_prediction(self, result: tuple[SKLearnResult, TFResult] | None):
        self._prediction_pending = False
        sklearn_result, tf_result = None, None

        if result is not None:
            sklearn_result, tf_result = result

        self._on_sklearn_prediction(sklearn_result)
        self._on_tf_prediction(tf_result)

    def _on_tf_prediction(self, result: TFResult | None):
        label = "..."

        if result is not None:
            preview_texture = new_preview_texture(result.preview_image)
            self.tf_preview_image.set_from_paintable(preview_texture)
            self.tf_digits_display.set_probs(result.predicted_digits)

            index, value = max(enumerate(result.predicted_digits), key=lambda iv: iv[1])
            # treshhold is 70%
            if value > 0.7:
                label = f"{index}"

        self.tf_predicted_label.set_label(label)

    def _on_sklearn_prediction(self, result: SKLearnResult | None):
        digit = None
        if result is not None:
            digit = result.predicted_digit
            self.sklearn_preview_image.set_from_paintable(
                new_preview_texture(result.preview_image)
            )

        label = "..." if digit is None else f"{digit}"
        self.sklearn_predicted_label.set_label(label)

    def _draw_line(self, x: float, y: float) -> None:
        if self._surface is None:
            return

        if self._last_point is None:
            self._last_point = (x, y)
            return

        context = cairo.Context(self._surface)
        context.set_source_rgb(0.0, 0.0, 0.0)
        context.set_line_width(16.0)
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
        self.connect("shutdown", self.on_shutdown)
        parent_conn, child_conn = Pipe()
        self.worker_conn = parent_conn
        self.tf_process = Process(target=run_tf_worker, args=(child_conn, "ov"))
        self.tf_process.start()
        self.main_window = None

    def do_activate(self) -> None:
        self.main_window = self.props.active_window
        if self.main_window is None:
            self.main_window = MainWindow(self)
        self.main_window.present()

    def on_shutdown(self, _data):
        self.worker_conn.send(None)
        self.tf_process.join()


class TFResult:
    def __init__(self, preview_image: Image.Image, predicted_digits: list[float]):
        self.preview_image = preview_image
        self.predicted_digits = predicted_digits


class SKLearnResult:
    def __init__(self, preview_image: Image.Image, predicted_digit: int | None):
        self.preview_image = preview_image
        self.predicted_digit = predicted_digit


def new_preview_image(image: Image.Image) -> Image.Image:
    if image.format != "RGBA":
        image = image.convert("RGBA")

    if image.size != (100, 100):
        image = image.resize((100, 100), Image.Resampling.BOX)
    return image


def new_preview_texture(image: Image.Image) -> Gdk.MemoryTexture | None:
    image = new_preview_image(image)

    texture = Gdk.MemoryTexture.new(
        image.width,
        image.height,
        Gdk.MemoryFormat.R8G8B8A8,
        GLib.Bytes.new(image.tobytes()),
        image.width * 4,
    )

    return texture


def run_tf_worker(conn: connection.Connection, model_engine: str = "ov"):
    if model_engine == "tf":
        model = get_tf_model()

        def predict(floats):  # pyright: ignore[reportRedeclaration]
            return model.predict(floats)[0]
    elif model_engine == "ov":
        model = get_ov_compiled_model()
        output_key = model.output(0)

        def predict(floats):
            return model(floats)[output_key][0]
    else:
        return

    clf = get_clf()

    def _sklearn_process(grayscale_image: Image.Image) -> SKLearnResult:
        grayscale_image_8x8 = grayscale_image.resize((8, 8), Image.Resampling.LANCZOS)
        # convert image data for svm.SCV
        sk_learn_data = [(~b & 0xFF) / 16.0 for b in grayscale_image_8x8.tobytes()]

        predicted_digit = None
        if any(f > 0 for f in sk_learn_data):
            predicted_digit = clf.predict([sk_learn_data])[0]

        return SKLearnResult(new_preview_image(grayscale_image_8x8), predicted_digit)

    def _tf_process(grayscale_image: Image.Image) -> TFResult:
        grayscale_image_28x28 = grayscale_image.resize(
            (28, 28), Image.Resampling.LANCZOS
        )
        # convert image data for tf model
        tf_data = [(~b & 0xFF) / 255.0 for b in grayscale_image_28x28.tobytes()]

        floats = np.array(tf_data).reshape(1, -1, 28)
        predicted_digits = predict(floats)

        return TFResult(new_preview_image(grayscale_image_28x28), predicted_digits)

    # work loop
    while True:
        task_image: Image.Image | None = conn.recv()
        if task_image is None:
            break

        grayscale_image = task_image.convert("L")

        # scikit-learn
        sklearn_result = _sklearn_process(grayscale_image)

        # tensor flow (openvino)
        tf_result = _tf_process(grayscale_image)

        conn.send((sklearn_result, tf_result))

    conn.send(None)
    conn.close()


def to_full_path(path: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        path,
    )


def get_clf() -> svm.SVC:
    model_file = to_full_path("models/sk_learn_digits.joblib")
    return load(model_file)


def get_tf_model():
    model_file = to_full_path("models/tf_learn_digits.keras")
    return tf.keras.models.load_model(model_file)


def get_ov_compiled_model():
    core = ov.Core()
    model = core.read_model(to_full_path("models/tf_learn_digits.xml"))
    return core.compile_model(model=model)


def get_drawing_cursor() -> Gdk.Cursor:
    texture = Gdk.Texture.new_from_filename(to_full_path("res/pencil-symbolic.png"))
    return Gdk.Cursor.new_from_texture(texture, 0, 31)


def main() -> None:
    app = DrawingApp()
    app.run(sys.argv)


if __name__ == "__main__":
    main()
