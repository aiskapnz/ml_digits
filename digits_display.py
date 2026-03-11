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
        def _to_rgba(value: Gdk.RGBA | str) -> Gdk.RGBA:
            if isinstance(value, str):
                parsed_color = Gdk.RGBA()
                parsed_color.parse(value)
                return parsed_color

            return value

        self.bg_color = _to_rgba(bg_color)
        self.off_pixel_color = _to_rgba(off_pixel_color)
        self.on_pixel_color = _to_rgba(on_pixel_color)


DEFAULT_DARK_PALETTE = ScreenPalette("#222b00", "#3a4600", "#8d9e4c")
DEFAULT_LIGHT_PALETTE = ScreenPalette("#3a4600", "#546201", "#ecffaa")

# The block = digit + probability bar + spacing.
BLOCK_H_CELLS = 5
BLOCK_V_CELLS = 9
CELL_SIZE = (4, 4)
PIXEL_SIZE = (CELL_SIZE[0] * 0.8, CELL_SIZE[1] * 0.8)
BLOCK_SIZE = (BLOCK_H_CELLS * CELL_SIZE[0], BLOCK_V_CELLS * CELL_SIZE[1])
H_MARGIN = CELL_SIZE[0] * 2
V_MARGIN = CELL_SIZE[1] * 2
H_SPACING = CELL_SIZE[0] * 2
V_SPACING = CELL_SIZE[1]


D0_BIT_MATRIX = (
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 1, 1),
    (1, 0, 1, 0, 1),
    (1, 1, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
)

D1_BIT_MATRIX = (
    (0, 0, 0, 1, 0),
    (0, 0, 1, 1, 0),
    (0, 1, 0, 1, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 0, 1, 0),
)

D2_BIT_MATRIX = (
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 1, 0),
    (0, 0, 1, 0, 0),
    (0, 1, 0, 0, 0),
    (1, 1, 1, 1, 1),
)

D3_BIT_MATRIX = (
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (0, 0, 0, 0, 1),
    (0, 0, 1, 1, 0),
    (0, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
)

D4_BIT_MATRIX = (
    (0, 0, 0, 1, 1),
    (0, 0, 1, 0, 1),
    (0, 1, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (1, 1, 1, 1, 1),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1),
)

D5_BIT_MATRIX = (
    (1, 1, 1, 1, 1),
    (1, 0, 0, 0, 0),
    (1, 0, 0, 0, 0),
    (1, 1, 1, 1, 0),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 0, 1),
    (1, 1, 1, 1, 0),
)

D6_BIT_MATRIX = (
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 0),
    (1, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
)

D7_BIT_MATRIX = (
    (1, 1, 1, 1, 1),
    (0, 0, 0, 0, 1),
    (0, 0, 0, 1, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 1, 0, 0),
)

D8_BIT_MATRIX = (
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
)

D9_BIT_MATRIX = (
    (0, 1, 1, 1, 0),
    (1, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 1),
    (0, 0, 0, 0, 1),
    (1, 0, 0, 0, 1),
    (0, 1, 1, 1, 0),
)

DIGIT_BIT_MATRICES = (
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
)


class DigitsDisplay(Adw.Bin):
    __gtype_name__ = "DigitsDisplay"

    probabilities: list[float]
    digits_coordinates: list[list[tuple[float, float]]]
    bars_coordinates: list[list[tuple[float, float]]]

    def __init__(self, display_threshold=0.7):
        self.display_threshold = display_threshold

        self.w_block, self.h_block = BLOCK_SIZE
        self.w_cell, self.h_cell = CELL_SIZE
        self.pixel_size = PIXEL_SIZE

        self.update_coordinates()
        self.reset_probabilities()

        self.drawing_area = Gtk.DrawingArea(
            width_request=int((self.w_block + H_SPACING) * 10 + H_MARGIN),
            height_request=int(self.h_block + V_MARGIN * 2),
        )
        self.drawing_area.set_draw_func(self.on_draw)

        self.set_child(Gtk.Frame(child=self.drawing_area))

        manager = Adw.StyleManager.get_default()
        manager.connect("notify::dark", self.on_dark)
        self.set_palette(manager.get_dark())

    def on_dark(self, manager: Adw.StyleManager, _p):
        self.set_palette(manager.get_dark())

    def on_draw(
        self,
        _area: Gtk.DrawingArea,
        cr: cairo.Context,
        _width: int,
        _height: int,
    ):
        plt = self.palette

        # draw a background
        cr.set_source_rgb(plt.bg_color.red, plt.bg_color.green, plt.bg_color.blue)
        cr.paint()

        for digit in range(10):
            color = plt.off_pixel_color

            d_probability = self.probabilities[digit]
            if d_probability >= self.display_threshold:
                color = plt.on_pixel_color

            for x, y in self.digits_coordinates[digit]:
                draw_pixel(cr, x, y, self.pixel_size, color)

            for i in range(BLOCK_H_CELLS):
                color = plt.off_pixel_color
                # not display for values under 0.1
                if d_probability > 0.1 and (d_probability >= i / BLOCK_H_CELLS):
                    color = plt.on_pixel_color

                x, y = self.bars_coordinates[digit][i]
                draw_pixel(cr, x, y, self.pixel_size, color)

    def redraw(self):
        self.drawing_area.queue_draw()

    def set_display_threshold(self, display_threshold: float):
        """Set the probability threshold for lighting a digit display"""

        self.display_threshold = display_threshold
        self.redraw()

    def set_probabilities(self, probabilities: list[float] | None):
        """Set the probabilities for each digit display"""

        if probabilities is None or len(probabilities) != 10:
            self.reset_probabilities()
        else:
            self.probabilities = probabilities

        self.redraw()

    def reset_probabilities(self):
        """Reset the probabilities for each digit display to 0.0"""

        self.probabilities = [0.0] * 10

    def update_coordinates(self):
        digits_coordinates = []
        bars_coordinates = []

        for digit in range(10):
            d_coordinates = []
            bar_coordinates = []

            # coordinates of a digit
            left = H_MARGIN + digit * (self.w_block + H_SPACING)
            y = V_MARGIN
            for row in DIGIT_BIT_MATRICES[digit]:
                x = left
                for pixel in row:
                    if pixel == 1:
                        d_coordinates.append((x, y))
                    x += self.w_cell

                y += self.h_cell

            # coordinates of a probability bar
            x = left
            y += V_SPACING
            for _ in range(BLOCK_H_CELLS):
                bar_coordinates.append((x, y))
                x += self.w_cell

            digits_coordinates.append(d_coordinates)
            bars_coordinates.append(bar_coordinates)

        self.digits_coordinates = digits_coordinates
        self.bars_coordinates = bars_coordinates

    def set_palette(self, dark: bool):
        """Set the palette based on whether dark mode is enabled"""

        self.palette = DEFAULT_DARK_PALETTE if dark else DEFAULT_LIGHT_PALETTE
        self.redraw()


def draw_pixel(
    cr: cairo.Context, x: float, y: float, size: tuple[float, float], color: Gdk.RGBA
):
    cr.set_source_rgb(color.red, color.green, color.blue)
    cr.rectangle(x, y, size[0], size[1])
    cr.fill()
