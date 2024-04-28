import sys

import matplotlib
import matplotlib.backends.backend_agg as agg
from sympy.parsing.sympy_parser import parse_expr
from pygame.locals import DOUBLEBUF
import pygame as pg
import numpy as np
import sympy as sp
import pylab

matplotlib.use("Agg")


def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class ParameterInputBox:
    def __init__(self, name, font, x, y, w, h, value=1):
        self.incactive_error_color = (255, 100, 100)
        self.active_error_color = (255, 0, 0)
        self.incactive_color = (100, 100, 100)
        self.active_color = (0, 0, 0)
        self.font = font
        self.font_width = (
            self.font.render("1234567890", True, (0, 0, 0)).get_width() / 10
        )
        self.w = w
        self.rect = pg.Rect(x, y, w, h)
        self.color = self.incactive_color
        self.true_value = float(value)
        self.value = str(value)
        self.name = name
        self.name_surface = self.font.render(f"{self.name}: ", True, self.active_color)
        self.name_width = self.name_surface.get_width()
        self.txt_surface = self.font.render(self.value, True, self.color)
        self.active = False
        self.active_digit = 0

    def flt2str(self, val):
        str_val = f"{val:.12f}".strip("0")

        if str_val[0] == ".":
            str_val = "0" + str_val
        if "." not in str_val:
            str_val += ".0"

        left, right = str_val.split(".")
        if self.active_digit < 0:
            padding = -self.active_digit - len(right)
            if padding > 0:
                str_val += "0" * (-self.active_digit - len(right))
        else:
            padding = self.active_digit - len(left) + 1
            if padding > 0:
                str_val = "0" * (self.active_digit - len(left) + 1) + str_val

        return str_val

    def set_neighbour(self, below_box):
        self.below_box = below_box

    def set_activation(self, activation):
        self.active = activation

        if is_number(self.value):
            self.color = self.active_color if self.active else self.incactive_color
        else:
            self.color = (
                self.active_error_color if self.active else self.incactive_error_color
            )

        self.txt_surface = self.font.render(self.value, True, self.color)

    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            # Activate if user clicked on the input box
            self.set_activation(self.rect.collidepoint(event.pos))

        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    self.set_activation(False)
                    return
                elif event.key == pg.K_BACKSPACE:
                    self.value = self.value[:-1]
                    if is_number(self.value):
                        self.true_value = float(self.value)
                elif event.key in range(48, 58) or event.key == 46:
                    self.value += event.unicode
                    self.true_value = float(self.value)
                elif event.key == pg.K_RIGHT:
                    self.active_digit -= 1
                    self.value = self.flt2str(self.true_value)
                elif event.key == pg.K_LEFT:
                    self.active_digit += 1
                    self.value = self.flt2str(self.true_value)
                elif event.key == pg.K_UP:
                    self.true_value += 10**self.active_digit
                    self.value = self.flt2str(self.true_value)
                elif event.key == pg.K_DOWN:
                    self.true_value -= 10**self.active_digit
                    self.value = self.flt2str(self.true_value)
                elif event.key == pg.K_c:
                    self.true_value = 0
                    self.value = "0"
                elif event.key == pg.K_TAB:
                    event.key = 0
                    self.set_activation(False)
                    self.below_box.set_activation(True)
                    return

                if self.value != "" and not is_number(self.value):
                    self.color = self.active_error_color
                else:
                    self.color = self.active_color

                self.txt_surface = self.font.render(self.value, True, self.color)

    def update(self):
        # Resize the box if the text is too long.
        width = max(self.w, self.txt_surface.get_width() + 10)
        self.rect.w = width

    def draw(self, screen):
        # Blit the text.
        screen.blit(self.name_surface, (self.rect.x - self.name_width, self.rect.y))
        screen.blit(self.txt_surface, (self.rect.x + 5, self.rect.y))
        if self.active:
            offsett = len(self.value.split(".")[0]) - self.active_digit
            if self.active_digit >= 0:
                offsett -= 1

            w = int(self.font_width)
            y = self.rect.y + self.rect.height - 3
            x = w * offsett + self.rect.x + 5
            pg.draw.line(screen, (0, 0, 200), (x, y), (x + w, y), 3)
        # Blit the rect.
        pg.draw.rect(screen, self.color, self.rect, 2)


class Plot:
    def __init__(self, pos, x_ray, y_ray, f, params, fig):
        self.fig = fig
        self.pos = pos
        self.params = params
        self.x_ray = x_ray
        self.y_ray = y_ray
        self.f = f

    def set_params(self, new_params):
        self.params = new_params

    def handle_event(self, event):
        pass

    def update(self):
        pass

    def draw(self, screen):
        ax = self.fig.gca()
        ax.plot(self.x_ray, self.y_ray, "ko")
        ax.plot(self.x_ray, self.f(self.x_ray, *self.params))

        canvas = agg.FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()

        surf = pg.image.fromstring(raw_data, size, "RGB")
        screen.blit(surf, self.pos)
        self.fig.clf()


def find_initial_params():
    datafile = "datafile.dat" if len(sys.argv) < 2 else sys.argv[1]
    datapath = f"data/{datafile}"

    with open(datapath, "r", encoding="utf8") as datafile:
        func, expr = datafile.readline().split("=")
        variable, parameters = func[2:-2].split(";")
        parameters = parameters.replace(" ", "").split(",")
        initial_parameters = np.ones(len(parameters))

        second_line = datafile.readline()

        try:
            x, y = second_line.split(" ")
            x, y = float(x), float(y)
            skip = 1
        except ValueError:
            param_map = {param: i for i, param in enumerate(parameters)}
            for param, val, *_ in [pair.split("=") for pair in second_line.split(", ")]:
                try:
                    initial_parameters[param_map[param]] = float(val)
                except ValueError:
                    print(f"Malformed parameter value: {param}={val}")
            skip = 2

    x, *params = sp.symbols(f"{variable}, {' '.join(parameters)}")

    local_dict = {s: v for s, v in zip(parameters, params)}
    local_dict[variable] = x

    symbolic_f = parse_expr(expr, local_dict=local_dict)
    f = sp.lambdify([x, *params], symbolic_f)

    x_ray, y_ray = np.loadtxt(datapath, skiprows=skip).T

    with open("preprocessor/initial_params.dat", "r", encoding="utf8") as paramfile:
        text = paramfile.read()
        if text != "":
            for i, v in enumerate([float(s) for s in text.split(" ")]):
                if v != 1.0:
                    initial_parameters[i] = v

    pg.init()

    font = pg.font.SysFont("FreeMono, Monospace", 32)
    figsize = (7, 5)
    dpi = 100

    parameter_padding = 8
    parameter_space = (32 + parameter_padding) * len(parameters) + parameter_padding
    _tutorial_font_size = 32

    tutorial_text_1 = font.render("Change parameter values to", True, (0, 0, 0))
    tutorial_text_2 = font.render("approximate the corect parameters", True, (0, 0, 0))
    tutorial_height = tutorial_text_1.get_height() + tutorial_text_2.get_height()

    _window = pg.display.set_mode(
        (figsize[0] * dpi, figsize[1] * dpi + parameter_space + tutorial_height),
        DOUBLEBUF,
    )
    screen = pg.display.get_surface()
    clock = pg.time.Clock()

    pg.display.set_caption("Close window or press escape to use parameters")

    plot = Plot(
        (0, tutorial_height + parameter_space - 40),
        x_ray,
        y_ray,
        f,
        [1] * len(parameters),
        pylab.figure(figsize=figsize, dpi=dpi),
    )

    max_width = 0
    for parameter in parameters:
        width = font.render(f" {parameter}: ", True, (0, 0, 0)).get_width()
        max_width = max(width, max_width)

    inputs = [
        ParameterInputBox(
            parameter,
            font,
            max_width,
            10 + tutorial_height + parameter_padding * (i + 1) + 32 * i,
            100,
            32,
            f"{initial_parameters[i]}",
        )
        for i, parameter in enumerate(parameters)
    ]
    for i, box in enumerate(inputs):
        box.set_neighbour(inputs[(i + 1) % len(inputs)])
    inputs[0].set_activation(True)

    entities = [plot, *inputs]

    done = False
    while not done:
        for event in pg.event.get():
            for entity in entities:
                entity.handle_event(event)
            if event.type == pg.QUIT or (event.type == pg.KEYDOWN and event.key == 27):
                done = True

        for entity in entities:
            entity.update()

        function_parameters = plot.params
        for box in inputs:
            if is_number(box.value):
                param_index = parameters.index(box.name)
                function_parameters[param_index] = float(box.value)

        plot.set_params(function_parameters)

        screen.fill((255, 255, 255))
        screen.blit(tutorial_text_1, (5, 10))
        screen.blit(tutorial_text_2, (5, 10 + tutorial_text_1.get_height()))
        for entity in entities:
            entity.draw(screen)
        pg.display.flip()

        clock.tick(10)

    pg.quit()

    better_params = np.array(plot.params, dtype=float)
    with open("preprocessor/initial_params.dat", "w", encoding="utf8") as paramfile:
        paramfile.write(" ".join([str(v) for v in better_params]))


if __name__ == "__main__":
    find_initial_params()
