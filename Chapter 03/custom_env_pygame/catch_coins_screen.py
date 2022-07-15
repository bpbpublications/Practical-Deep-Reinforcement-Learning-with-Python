import os

#Set up the colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)


class CatchCoinsScreen:
    img_path = os.path.dirname(os.path.realpath(__file__)) + '/airplane.png'
    rct_size = 50

    def __init__(self, h = 5, w = 5) -> None:
        import pygame
        pygame.init()
        self.h = h
        self.w = w
        scr_size = ((w + 2) * self.rct_size, (h + 3) * self.rct_size)
        self.scr = pygame.display.set_mode(scr_size, 0, 32)
        self.img = pygame.image.load(CatchCoinsScreen.img_path)
        self.font = pygame.font.SysFont(None, 48)
        pygame.display.set_caption('Catch Coins Game')
        super().__init__()

    def plus(self):
        import pygame
        self.scr.fill(GREEN)
        pygame.display.update()

    def update(self, display, plane_pos, total_score):
        import pygame
        self.scr.fill(WHITE)
        for i in range(len(display)):
            line = display[len(display) - 1 - i]
            for j in range(len(line)):
                p = line[j]
                if p > 0:
                    coord = ((j + 1) * self.rct_size, (i + 1) * self.rct_size)
                    self.scr.blit(self.font.render(str(p), True, BLACK), coord)

        self.scr.blit(self.font.render(f'Total: {total_score}', True, BLACK), (10, 10))
        self.scr.blit(self.img, (self.rct_size * plane_pos + 30, (self.h + 1) * self.rct_size))
        pygame.display.update()

    @classmethod
    def render(cls, display, plane_pos, total_score):
        scr = CatchCoinsScreen()
        scr.update(display, plane_pos, total_score)
