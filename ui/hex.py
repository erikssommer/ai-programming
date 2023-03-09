from math import cos, sin, pi, radians

import numpy as np
import pygame
from pygame import gfxdraw
from pygame import time


class HexUI:
    def __init__(self, board_size: int, board=None):
        pygame.init()

        pygame.display.set_caption('Hex')
        self.board_size = board_size
        self.board = board

        assert 1 < self.board_size <= 26

        self.clock = time.Clock()
        self.hex_radius = 20
        self.x_offset, self.y_offset = 60, 60
        self.text_offset = 45
        self.screen = pygame.display.set_mode(
            (self.x_offset + (2 * self.hex_radius) * self.board_size + self.hex_radius * self.board_size,
             round(self.y_offset + (1.75 * self.hex_radius) * self.board_size)))

        # Colors
        self.red = (222, 29, 47)
        self.blue = (0, 121, 251)
        self.green = (0, 255, 0)
        self.white = (255, 255, 255)
        self.black = (40, 40, 40)
        self.gray = (70, 70, 70)

        # Players
        self.BLUE_PLAYER = 1
        self.RED_PLAYER = 2

        self.screen.fill(self.white)
        self.fonts = pygame.font.SysFont("Sans", 20)

        self.hex_lookup = {}
        self.rects, self.color, self.node = [], [self.black] * (self.board_size ** 2), None

    def draw_hexagon(self, surface: object, position: tuple, node: int, hex_color: tuple):
        # Vertex count and radius
        n = 6
        x, y = position
        offset = 3

        # Outline
        self.hex_lookup[node] = [(x + (self.hex_radius + offset) * cos(radians(90) + 2 * pi * _ / n),
                                  y + (self.hex_radius + offset) * sin(radians(90) + 2 * pi * _ / n))
                                 for _ in range(n)]
        gfxdraw.aapolygon(surface,
                          self.hex_lookup[node],
                          hex_color)

        # Shape
        gfxdraw.filled_polygon(surface,
                               [(x + self.hex_radius * cos(radians(90) + 2 * pi * _ / n),
                                 y + self.hex_radius * sin(radians(90) + 2 * pi * _ / n))
                                for _ in range(n)],
                               hex_color)

        # Antialiased shape outline
        gfxdraw.aapolygon(surface,
                          [(x + self.hex_radius * cos(radians(90) + 2 * pi * _ / n),
                            y + self.hex_radius * sin(radians(90) + 2 * pi * _ / n))
                           for _ in range(n)],
                          self.black)

        # Placeholder
        rect = pygame.draw.rect(surface,
                                hex_color,
                                pygame.Rect(x - self.hex_radius + offset, y - (self.hex_radius / 2),
                                            (self.hex_radius * 2) - (2 * offset), self.hex_radius))
        self.rects.append(rect)

        # Bounding box (colour-coded)
        bbox_offset = [0, 3]

        # Top side
        if 0 < node < self.board_size:
            points = ([self.hex_lookup[node - 1][3][_] - bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node - 1][4][_] - bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node][3][_] - bbox_offset[_] for _ in range(2)])
            gfxdraw.filled_polygon(surface,
                                   points,
                                   self.red)
            gfxdraw.aapolygon(surface,
                              points,
                              self.red)

        # Bottom side
        if self.board_size ** 2 - self.board_size < node < self.board_size ** 2:
            points = ([self.hex_lookup[node - 1][0][_] + bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node - 1][5][_] + bbox_offset[_] for _ in range(2)],
                      [self.hex_lookup[node][0][_] + bbox_offset[_] for _ in range(2)])
            gfxdraw.filled_polygon(surface,
                                   points,
                                   self.red)
            gfxdraw.aapolygon(surface,
                              points,
                              self.red)

        # Left side
        bbox_offset = [3, -3]

        if node % self.board_size == 0:
            if node >= self.board_size:
                points = ([self.hex_lookup[node - self.board_size][1][_] - bbox_offset[_] for _ in range(2)],
                          [self.hex_lookup[node - self.board_size][0][_] - bbox_offset[_] for _ in range(2)],
                          [self.hex_lookup[node][1][_] - bbox_offset[_] for _ in range(2)])
                gfxdraw.filled_polygon(surface,
                                       points,
                                       self.blue)
                gfxdraw.aapolygon(surface,
                                  points,
                                  self.blue)

        # Right side
        if (node + 1) % self.board_size == 0:
            if node > self.board_size:
                points = ([self.hex_lookup[node - self.board_size][4][_] + bbox_offset[_] for _ in
                           range(2)],
                          [self.hex_lookup[node - self.board_size][5][_] + bbox_offset[_] for _ in
                           range(2)],
                          [self.hex_lookup[node][4][_] + bbox_offset[_] for _ in range(2)])
                gfxdraw.filled_polygon(surface,
                                       points,
                                       self.blue)
                gfxdraw.aapolygon(surface,
                                  points,
                                  self.blue)

    def draw_text(self):
        alphabet = list(map(chr, range(97, 123)))

        for _ in range(self.board_size):
            # Columns
            text = self.fonts.render(alphabet[_].upper(), True, self.black, self.white)
            text_rect = text.get_rect()
            text_rect.center = (self.x_offset + (2 * self.hex_radius) * _, self.text_offset / 2)
            self.screen.blit(text, text_rect)

            # Rows
            text = self.fonts.render(str(_), True, self.black, self.white)
            text_rect = text.get_rect()
            text_rect.center = (
                (self.text_offset / 4 + self.hex_radius * _, self.y_offset + (1.75 * self.hex_radius) * _))
            self.screen.blit(text, text_rect)

    def draw_board(self):
        counter = 0
        for row in range(self.board_size):
            for column in range(self.board_size):
                if self.board[row][column] == 0:
                    self.draw_hexagon(self.screen, self.get_coordinates(row, column), counter, self.white)
                elif self.board[row][column] == 1:
                    self.draw_hexagon(self.screen, self.get_coordinates(row, column), counter, self.red)
                elif self.board[row][column] == 2:
                    self.draw_hexagon(self.screen, self.get_coordinates(row, column), counter, self.blue)
                counter += 1
        self.draw_text()

    def get_coordinates(self, row: int, column: int):
        x = self.x_offset + (2 * self.hex_radius) * column + self.hex_radius * row
        y = self.y_offset + (1.75 * self.hex_radius) * row

        return x, y

    def get_true_coordinates(self, node: int):
        return int(node / self.board_size), node % self.board_size

