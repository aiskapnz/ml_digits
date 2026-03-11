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
import time
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing import Pipe, Process, connection
from threading import Thread

import cairo
import cv2 as cv
import gi
import joblib
import numpy as np
import openvino as ov
import tensorflow as tf
from sklearn import svm

import digits_display

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import (  # noqa: E402
    Adw,  # pyright: ignore[reportMissingModuleSource]
    Gdk,  # pyright: ignore[reportMissingModuleSource]
    GLib,  # pyright: ignore[reportMissingModuleSource]
    Gtk,  # pyright: ignore[reportMissingModuleSource]
)


def to_full_path(path: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        path,
    )


TF_MODEL_PATH = to_full_path("models/tf_digits.keras")
OV_MODEL_PATH = to_full_path("models/ov_digits.xml")
SKLEARN_MODEL_PATH = to_full_path("models/sklearn_digits.joblib")
DRAWING_CURSOR_PATH = to_full_path("res/pencil-symbolic.png")

DIGIT_DISPLAY_THRESHOLD = 0.7


class Model(StrEnum):
    SKLEARN_MODEL = "skl"
    TF_MODEL = "tf"
    OV_MODEL = "ov"


@Gtk.Template(filename="main_window.ui")
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = "MainWindow"

    input_label: Gtk.Label = Gtk.Template.Child()
    drawing_area: Gtk.DrawingArea = Gtk.Template.Child()
    clear_button: Gtk.Button = Gtk.Template.Child()
    sklearn_preview_image: Gtk.Image = Gtk.Template.Child()
    sklearn_prediction_label: Gtk.Label = Gtk.Template.Child()
    sklearn_time_label: Gtk.Label = Gtk.Template.Child()
    tf_ov_toggle_group: Adw.ToggleGroup = Gtk.Template.Child()
    tf_toggle: Adw.Toggle = Gtk.Template.Child()
    ov_toggle: Adw.Toggle = Gtk.Template.Child()
    tf_preview_image: Gtk.Image = Gtk.Template.Child()
    tf_digits_display: digits_display.DigitsDisplay = Gtk.Template.Child()
    tf_ov_prediction_label: Gtk.Label = Gtk.Template.Child()
    tf_ov_time_label: Gtk.Label = Gtk.Template.Child()

    _prediction_pending = False

    def __init__(self, app: DrawingApp) -> None:
        super().__init__(application=app)

        self.app = app

        self.models_toggles = {
            Model.SKLEARN_MODEL: True,
            Model.TF_MODEL: False,
            Model.OV_MODEL: True,
        }

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

        self.tf_ov_toggle_group.connect(
            "notify::active", self._on_tf_ov_toggle_activate
        )
        self.tf_ov_toggle_group.set_active(active=self.ov_toggle.get_index())

        self.tf_digits_display.set_display_threshold(DIGIT_DISPLAY_THRESHOLD)

        self.is_waiting_result = True
        self._result_waiter_thread = Thread(target=self.result_waiter, daemon=True)
        self._result_waiter_thread.start()

        self.update_prediction()

    def do_close_request(self) -> bool:
        self.is_waiting_result = False
        return Adw.ApplicationWindow.do_close_request(self)

    def _on_tf_ov_toggle_activate(self, *_args):
        index = self.tf_ov_toggle_group.get_active()

        if index == self.ov_toggle.get_index():
            self.models_toggles[Model.OV_MODEL] = True
            self.update_prediction([Model.OV_MODEL])
        elif index == self.tf_toggle.get_index():
            self.models_toggles[Model.TF_MODEL] = True
            self.update_prediction([Model.TF_MODEL])

    def result_waiter(self) -> None:
        while self.is_waiting_result:
            results: list[TFOVResult | SKLearnResult] | None = (
                self.app.worker_conn.recv()
            )

            if results is None:
                break

            GLib.idle_add(self._on_prediction, results)

    def clear(self) -> None:
        if self._surface is None:
            return

        context = cairo.Context(self._surface)
        context.set_source_rgb(1.0, 1.0, 1.0)
        context.paint()
        self._surface.flush()
        self.drawing_area.queue_draw()

        # to clear predicted label
        self.update_prediction()

    def get_image(self) -> np.ndarray | None:
        if self._surface is None:
            return None

        width = self._surface.get_width()
        height = self._surface.get_height()
        data = self._surface.get_data()
        image = np.frombuffer(data, dtype=np.uint8).reshape((width, height, 4))
        return image

    def update_prediction(self, models_toggles: list[Model] | None = None):
        if not self._prediction_pending:
            if models_toggles is None:
                models_toggles = [
                    model for model, enabled in self.models_toggles.items() if enabled
                ]

            self._prediction_pending = True
            image = self.get_image()

            if image is None:
                self._prediction_pending = False
                return

            # send image to process
            self.app.worker_conn.send(
                InferenceTask(
                    image,
                    models_toggles,
                )
            )

    def _on_prediction(self, results: list[TFOVResult | SKLearnResult]):
        self._prediction_pending = False

        for result in results:
            if isinstance(result, SKLearnResult):
                self._on_sklearn_prediction(result)
            elif isinstance(result, TFOVResult):
                self._on_tf_ov_prediction(result)

    def _on_tf_ov_prediction(self, result: TFOVResult | None):
        prediction = "-"
        inference_time = "-"

        if result is not None:
            preview_texture = new_preview_texture(result.preview_image)
            self.tf_preview_image.set_from_paintable(preview_texture)
            self.tf_digits_display.set_probabilities(result.predicted_digits)

            if result.predicted_digits is not None:
                index, value = max(
                    enumerate(result.predicted_digits), key=lambda iv: iv[1]
                )
                if value >= DIGIT_DISPLAY_THRESHOLD:
                    prediction = f"{index}"

            if result.inference_time is not None:
                inference_time = f"{result.inference_time:.3f} ms"

        self.tf_ov_prediction_label.set_label(prediction)
        self.tf_ov_time_label.set_label(inference_time)

    def _on_sklearn_prediction(self, result: SKLearnResult | None):
        prediction = "-"
        inference_time = "-"

        if result is not None:
            self.sklearn_preview_image.set_from_paintable(
                new_preview_texture(result.preview_image)
            )

            if result.predicted_digit is not None:
                prediction = f"{result.predicted_digit}"

            if result.inference_time is not None:
                inference_time = f"{result.inference_time:.3f} ms"

        self.sklearn_prediction_label.set_label(prediction)
        self.sklearn_time_label.set_label(inference_time)

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

    def _on_clear(self, user_data):
        self.clear()

    def _on_resize(self, _area: Gtk.DrawingArea, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.clear()


@dataclass
class InferenceTask:
    image: np.ndarray
    models: list[Model]


@dataclass
class TFOVResult:
    preview_image: np.ndarray
    predicted_digits: list[float] | None
    inference_time: float | None  # inference_time in ms


@dataclass
class SKLearnResult:
    preview_image: np.ndarray
    predicted_digit: int | None
    inference_time: float | None  # inference_time in ms


class DrawingApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id="com.example.DrawingApp")
        self.connect("shutdown", self.on_shutdown)

        parent_conn, child_conn = Pipe()
        self.worker_conn = parent_conn
        self.worker_process = Process(target=run_worker, args=(child_conn,))
        self.worker_process.start()

        self.main_window = None

    def do_activate(self) -> None:
        self.main_window = self.props.active_window
        if self.main_window is None:
            self.main_window = MainWindow(self)
        self.main_window.present()

    def on_shutdown(self, _data):
        self.worker_conn.send(None)
        self.worker_process.join()


def run_worker(conn: connection.Connection) -> None:
    tf_model = get_tf_model()
    ov_model = get_ov_compiled_model()
    sklearn_model = get_sklearn_model()

    def tf_predict(floats):  # pyright: ignore[reportRedeclaration]
        return tf_model.predict(floats)[0]

    def ov_predict(floats):
        return ov_model(floats)[ov_model.output(0)][0]

    def _sklearn_process(grayscale_image: np.ndarray) -> SKLearnResult:
        grayscale_image_8x8 = cv.resize(
            grayscale_image, (8, 8), interpolation=cv.INTER_AREA
        )

        # convert image data for svm.SCV
        sklearn_data = grayscale_image_8x8 * 0.062745098  # 16.0 / 255.0 = 0.062745098
        predicted_digit = None
        inference_time = None

        if sklearn_data.any():
            start = time.perf_counter()
            predicted_digit = sklearn_model.predict(sklearn_data.reshape(1, -1))[0]
            end = time.perf_counter()

            inference_time = (end - start) * 1000

        return SKLearnResult(
            new_preview_image(np.invert(grayscale_image_8x8)),
            predicted_digit,
            inference_time,
        )

    def _tf_ov_process(grayscale_image: np.ndarray, model: Model) -> TFOVResult:
        grayscale_image_28x28 = cv.resize(
            grayscale_image, (28, 28), interpolation=cv.INTER_AREA
        )

        # convert image data for tf model
        tf_data = grayscale_image_28x28 / 255.0
        predicted_digits = None
        inference_time = None

        if tf_data.any():
            floats = tf_data.reshape(1, -1, 28)
            predictors = {
                Model.TF_MODEL: tf_predict,
                Model.OV_MODEL: ov_predict,
            }

            predict_fn = predictors.get(model)
            if predict_fn is None:
                raise ValueError(f"Unknown model: {model}")

            start = time.perf_counter()
            predicted_digits = predict_fn(floats)
            end = time.perf_counter()

            inference_time = (end - start) * 1000

        return TFOVResult(
            new_preview_image(np.invert(grayscale_image_28x28)),
            predicted_digits,
            inference_time,
        )

    # work loop
    while True:
        task: InferenceTask | None = conn.recv()
        if task is None:
            break

        grayscale_image = cv.cvtColor(task.image, cv.COLOR_RGBA2GRAY)
        grayscale_image = np.invert(grayscale_image)
        grayscale_image = crop_to_content(grayscale_image)

        results = []
        for model in task.models:
            if model == Model.SKLEARN_MODEL:
                results.append(_sklearn_process(grayscale_image))
            elif model in (Model.TF_MODEL, Model.OV_MODEL):
                results.append(_tf_ov_process(grayscale_image, model))

        conn.send(results)

    conn.send(None)
    conn.close()


def crop_to_content(image: np.ndarray) -> np.ndarray:
    """Crop to non-zero content and pad to a square with margin."""

    if not image.any():
        return image

    cropped_image = np.trim_zeros(image)
    columns, rows = cropped_image.shape
    side = max(28, columns, rows)  # min side: 28
    pad = int(side * 0.2)

    delta_columns = side - columns + pad
    left = delta_columns // 2
    right = delta_columns - left

    delta_rows = side - rows + pad
    top = delta_rows // 2
    bottom = delta_rows - top

    return np.pad(cropped_image, ((left, right), (top, bottom)))


def new_preview_image(image: np.ndarray) -> np.ndarray:
    """Create a resized RGBA preview image from a grayscale input."""
    """Image should be in grayscale format."""

    image = cv.resize(image, (100, 100), interpolation=cv.INTER_AREA)
    image = cv.cvtColor(image, cv.COLOR_GRAY2RGBA)
    return image


def new_preview_texture(image: np.ndarray) -> Gdk.MemoryTexture | None:
    """Create a Gdk.MemoryTexture from an RGBA numpy image array."""

    width, height, _ = image.shape
    image_bytes = image.tobytes()
    glib_bytes = GLib.Bytes.new(image_bytes)
    texture = Gdk.MemoryTexture.new(
        width,
        height,
        Gdk.MemoryFormat.R8G8B8A8,
        glib_bytes,
        width * 4,
    )

    return texture


def get_drawing_cursor() -> Gdk.Cursor:
    texture = Gdk.Texture.new_from_filename(DRAWING_CURSOR_PATH)
    return Gdk.Cursor.new_from_texture(texture, 0, 31)


def get_sklearn_model() -> svm.SVC:
    return joblib.load(SKLEARN_MODEL_PATH)


def get_tf_model() -> tf.keras.Model:
    return tf.keras.models.load_model(TF_MODEL_PATH)


def get_ov_compiled_model() -> ov.CompiledModel:
    core = ov.Core()
    model = core.read_model(OV_MODEL_PATH)
    return core.compile_model(model=model)


def main() -> None:
    app = DrawingApp()
    app.run(sys.argv)


if __name__ == "__main__":
    main()
