import cairo
import gi
from PIL import Image

gi.require_version("Gtk", "4.0")
from gi.repository import GdkPixbuf, GLib, Gtk


class DrawingAreaWindow(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application) -> None:
        super().__init__(application=app)
        self.set_title("Drawing Area")
        self.set_default_size(800, 600)
        self.set_resizable(True)

        self._surface: cairo.ImageSurface | None = None
        self._last_point: tuple[float, float] | None = None

        self._root_box = Gtk.Box()
        self._root_box.set_orientation(Gtk.Orientation.VERTICAL)
        self._root_box.set_property("spacing", 12)

        self._input_label = Gtk.Label(label="Draw any digit (0-9):")
        self._root_box.append(self._input_label)

        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_size_request(200, 200)
        self._drawing_area.set_draw_func(self._on_draw)
        self._drawing_area.connect("resize", self._on_resize)
        self._drawing_area.set_halign(Gtk.Align.CENTER)
        self._root_box.append(self._drawing_area)

        self._resized_image = Gtk.Image()
        self._resized_image.set_size_request(100, 100)
        self._resized_image.set_pixel_size(100)
        self._root_box.append(self._resized_image)

        self._clear_button = Gtk.Button()
        self._clear_button.set_label("Clear")
        self._clear_button.set_halign(Gtk.Align.CENTER)
        self._clear_button.connect("clicked", self._on_clear)
        self._root_box.append(self._clear_button)

        self.set_child(self._root_box)

        drag = Gtk.GestureDrag()
        drag.set_button(1)
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self._drawing_area.add_controller(drag)

    def _on_clear(self, user_data):
        self.clear()

    def clear(self):
        if self._surface is None:
            return

        context = cairo.Context(self._surface)
        context.set_source_rgb(1.0, 1.0, 1.0)
        context.paint()
        self._surface.flush()
        self._drawing_area.queue_draw()

    def _on_resize(self, _area: Gtk.DrawingArea, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            return

        self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self.clear()
        # context = cairo.Context(new_surface)
        # context.set_source_rgb(1.0, 1.0, 1.0)
        # context.paint()
        # new_surface.flush()

        # self._surface = new_surface

    def _on_draw(
        self,
        _area: Gtk.DrawingArea,
        context: cairo.Context,
        width: int,
        height: int,
    ) -> None:
        if self._surface is None:
            return

        context.set_source_surface(self._surface, 0, 0)
        context.paint()

        # convert image to 8*8
        w = self._surface.get_width()
        h = self._surface.get_height()
        data = bytes(self._surface.get_data())
        image_8x8 = resize_to_8x8(w, h, data)

        # display image 8*8
        w, h = image_8x8.size
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes.new(image_8x8.tobytes()),
            GdkPixbuf.Colorspace.RGB,
            True,
            8,
            w,
            h,
            w * 4,
        )
        pixbuf = pixbuf.scale_simple(100, 100, GdkPixbuf.InterpType.TILES)
        self._resized_image.set_from_pixbuf(pixbuf)

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
        self._drawing_area.queue_draw()

    def _on_drag_update(
        self, gesture: Gtk.GestureDrag, _offset_x: float, _offset_y: float
    ) -> None:
        _has_start, start_x, start_y = gesture.get_start_point()
        _has_offset, offset_x, offset_y = gesture.get_offset()
        self._draw_line(start_x + offset_x, start_y + offset_y)
        self._drawing_area.queue_draw()

    def _on_drag_end(
        self, _gesture: Gtk.GestureDrag, _offset_x: float, _offset_y: float
    ) -> None:
        self._last_point = None


class DrawingApp(Gtk.Application):
    def __init__(self) -> None:
        super().__init__(application_id="com.example.DrawingApp")

    def do_activate(self) -> None:
        window = self.props.active_window
        if window is None:
            window = DrawingAreaWindow(self)
        window.present()


def resize_to_8x8(w, h, data):
    # L = 8-bit grayscale (A8)
    pil = Image.frombytes("RGBA", (w, h), data)
    return pil.resize((8, 8), Image.Resampling.LANCZOS)


def main() -> None:
    app = DrawingApp()
    app.run()


if __name__ == "__main__":
    main()
