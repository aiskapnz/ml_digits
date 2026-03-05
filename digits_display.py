import cairo
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import (  # type: ignore[import-not-found]  # noqa: E402
    Adw,  # pyright: ignore[reportMissingModuleSource]
    Gdk,  # pyright: ignore[reportMissingModuleSource]
    Gtk,  # pyright: ignore[reportMissingModuleSource]
)


class ScreenPalette:
    def __init__(
        self,
        bg_color: Gdk.RGBA | str,
        off_pixel_color: Gdk.RGBA | str,
        on_pixel_color: Gdk.RGBA | str,
    ):
        if isinstance(bg_color, str):
            parsed_color = Gdk.RGBA()
            parsed_color.parse(bg_color)
            bg_color = parsed_color

        self.bg_color = bg_color

        if isinstance(off_pixel_color, str):
            parsed_color = Gdk.RGBA()
            parsed_color.parse(off_pixel_color)
            off_pixel_color = parsed_color

        self.off_pixel_color = off_pixel_color

        if isinstance(on_pixel_color, str):
            parsed_color = Gdk.RGBA()
            parsed_color.parse(on_pixel_color)
            on_pixel_color = parsed_color

        self.on_pixel_color = on_pixel_color


DEFAULT_DARK_PALETTE = ScreenPalette("#222b00", "#3a4600", "#8d9e4c")
DEFAULT_LIGHT_PALETTE = ScreenPalette("#3a4600", "#546201", "#ecffaa")

DIGIT_BIT_MATRIX_WIDTH = 5

D0_BIT_MATRIX = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]

D1_BIT_MATRIX = [
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
]

D2_BIT_MATRIX = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [1, 1, 1, 1, 1],
]

D3_BIT_MATRIX = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]

D4_BIT_MATRIX = [
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
]

D5_BIT_MATRIX = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [1, 1, 1, 1, 0],
]

D6_BIT_MATRIX = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]

D7_BIT_MATRIX = [
    [1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
]

D8_BIT_MATRIX = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]

D9_BIT_MATRIX = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
]

DIGIT_BIT_MATRICES = [
    D0_BIT_MATRIX,
    D1_BIT_MATRIX,
    D2_BIT_MATRIX,
    D3_BIT_MATRIX,
    D4_BIT_MATRIX,
    D5_BIT_MATRIX,
    D6_BIT_MATRIX,
    D7_BIT_MATRIX,
    D8_BIT_MATRIX,
    D9_BIT_MATRIX,
]


class DigitsDisplay(Adw.Bin):
    __gtype_name__ = "DigitsDisplay"
    digits_probs: list[float]

    def __init__(self, display_threshold=0.7):
        self.display_threshold = display_threshold

        self.w_block = 35
        self.h_block = 55
        self.w_cell = self.w_block / 7
        self.h_cell = self.h_block / 11
        self.pixel_size = (self.w_cell * 0.8, self.h_cell * 0.8)
        self.reset_digit_probs()

        self.drawing_area = Gtk.DrawingArea(
            width_request=int(self.w_block * 10 + self.w_cell * 2),
            height_request=int(self.h_block + self.h_cell * 2),
        )
        self.drawing_area.set_draw_func(self.on_draw)

        self.set_child(Gtk.Frame(child=self.drawing_area))

        manager = Adw.StyleManager.get_default()
        manager.connect("notify::dark", self.on_dark)
        self.set_palette(manager.get_dark())

    def reset_digit_probs(self):
        self.digits_probs = [0.0] * 10

    def set_display_threshold(self, display_threshold: float):
        """Set the probability threshold for lighting a digit display."""

        self.display_threshold = display_threshold
        self.redraw()

    def on_dark(self, manager: Adw.StyleManager, _p):
        self.set_palette(manager.get_dark())

    def on_draw(
        self,
        area: Gtk.DrawingArea,
        cr: cairo.Context,
        width: int,
        height: int,
    ):
        plt = self.palette

        cr.set_source_rgb(plt.bg_color.red, plt.bg_color.green, plt.bg_color.blue)
        cr.paint()

        w_cell, h_cell = self.w_cell, self.h_cell

        for d in range(10):
            x_offset = d * self.w_block + w_cell
            color = plt.off_pixel_color

            d_prob = self.digits_probs[d]
            if d_prob >= self.display_threshold:
                color = plt.on_pixel_color

            # draw a digit
            left = x_offset + w_cell
            y = h_cell * 2
            for row in DIGIT_BIT_MATRICES[d]:
                x = left
                for pixel in row:
                    if pixel == 1:
                        draw_pixel(cr, x, y, self.pixel_size, color)
                    x += w_cell

                y += h_cell

            # draw probability bar
            x = left
            y += h_cell
            for i in range(DIGIT_BIT_MATRIX_WIDTH):
                color = plt.off_pixel_color
                # not display for values under 0.1
                if d_prob > 0.1 and (d_prob >= i / DIGIT_BIT_MATRIX_WIDTH):
                    color = plt.on_pixel_color

                draw_pixel(cr, x, y, self.pixel_size, color)
                x += w_cell

    def set_palette(self, dark: bool):
        self.palette = DEFAULT_DARK_PALETTE if dark else DEFAULT_LIGHT_PALETTE
        self.redraw()

    def set_probs(self, probs: list[float] | None):
        if probs is None or len(probs) != 10:
            self.reset_digit_probs()
        else:
            self.digits_probs = probs

        self.redraw()

    def redraw(self):
        self.drawing_area.queue_draw()


def draw_pixel(
    cr: cairo.Context, x: float, y: float, size: tuple[float, float], color: Gdk.RGBA
):
    cr.set_source_rgb(color.red, color.green, color.blue)
    cr.rectangle(x, y, size[0], size[1])
    cr.fill()
