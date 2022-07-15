import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pylab
import pygame
from pygame.locals import *


class TradingScreen:

    def __init__(self) -> None:
        # Initializing PyGame and Display
        pygame.init()
        self.scr = pygame.display.set_mode((600, 400), DOUBLEBUF)

    def update(self, quotes, buys, sells, name, total_reward):
        self.scr.fill((255, 255, 255))  # Fill with WHITE
        fig = pylab.figure(figsize = [4, 4], dpi = 100, )
        ax = fig.gca()
        ax.title.set_text(f"{name}\nTotal: {total_reward}")
        ax.set_xticks([])

        # Displaying BUY actions
        ax.scatter(
            list(buys.keys()),
            list(buys.values()),
            marker = "o",
            color = "green"
        )

        # Displaying SELL actions
        ax.scatter(
            list(sells.keys()),
            list(sells.values()),
            marker = "o",
            color = "red"
        )

        ax.plot(quotes)

        # Rendering PLOT as Video Frame
        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()

        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        self.scr.blit(surf, (0, 0))
        pygame.display.update()
        plt.close('all')
