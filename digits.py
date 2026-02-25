import cairo
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import (  # type: ignore[import-not-found]  # noqa: E402
    Adw,  # pyright: ignore[reportMissingModuleSource]
    Gdk,  # pyright: ignore[reportMissingModuleSource]
    GLib,  # pyright: ignore[reportMissingModuleSource]
    Gtk,  # pyright: ignore[reportMissingModuleSource]
)

PALETTE = [
    Gdk.RGBA(0.00, 0.46, 0.25),
    Gdk.RGBA(0.12, 0.55, 0.27),
    Gdk.RGBA(0.23, 0.64, 0.27),
    Gdk.RGBA(0.35, 0.73, 0.27),
    Gdk.RGBA(0.47, 0.82, 0.24),
    Gdk.RGBA(0.61, 0.91, 0.19),
    Gdk.RGBA(0.76, 1.00, 0.11),
    Gdk.RGBA(0.87, 1.00, 0.55),
]
PALETTE_COUNT = len(PALETTE)


class Digits(Gtk.Box):
    __gtype_name__ = "Digits"

    def __init__(self):
        self.set_orientation(Gtk.Orientation.HORIZONTAL)
        self.set_spacing(4)

        self.colors_indexes: list[int] = []
        self.drawing_areas: list[Gtk.DrawingArea] = []

        for digit in range(10):
            drawing_area = Gtk.DrawingArea(
                hexpand=True,
                width_request=30,
                height_request=30,
            )
            drawing_area.set_draw_func(self.on_draw, digit)

            frame = Gtk.Frame(child=drawing_area)

            box = Gtk.Box(css_classes=["card"])
            box.append(frame)
            self.append(box)

            self.colors_indexes.append(0)
            self.drawing_areas.append(drawing_area)

    def on_draw(
        self,
        area: Gtk.DrawingArea,
        cr: cairo.Context,
        width: int,
        height: int,
        digit: int,
    ):
        color_index = self.colors_indexes[digit]
        color = PALETTE[color_index]
        cr.set_source_rgb(color.red, color.green, color.blue)
        cr.paint()

        brightness = (color.red + color.green + color.blue) / 3
        if brightness <= 0.5:
            cr.set_source_rgb(1.0, 1.0, 1.0)
        else:
            cr.set_source_rgb(0.0, 0.0, 0.0)
        cr.set_font_size(height * 0.5)
        cr.move_to(width / 3, height * 0.66)
        cr.show_text(str(digit))

    def set_probs(self, probs: list[float]):
        if len(probs) != 10:
            return

        for i, prob in enumerate(probs):
            color_index = int((PALETTE_COUNT - 1) * prob)
            if self.colors_indexes[i] == color_index:
                continue

            self.colors_indexes[i] = color_index
            self.drawing_areas[i].queue_draw()
