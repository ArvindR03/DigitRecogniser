import pygame
import load_nn
import numpy as np
import tensorflow
import keras
import matplotlib.pyplot as plt

WIDTH = 28*15
ROWS = 28
WINDOW = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption('Digit Recognition')
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
GREEN = (40, 250, 40)

class Square:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.width = width
        self.x = row * width
        self.y = col * width
        self.total_rows = total_rows
        self.color = WHITE
        self.neighbours = []

    def get_pos(self):
        return self.row, self.col

    def isAlive(self):
        return self.color == BLACK

    def nnAlive(self):
        if self.color == BLACK:
            return 1
        else:
            return 0

    def makeAlive(self):
        self.color = BLACK

    def makeDead(self):
        self.color = WHITE

    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
def makeGrid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            square = Square(i, j, gap, rows)
            grid[i].append(square)
    return grid
    
def drawGridlines(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
    for j in range(rows):
        pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(WHITE)
    for row in grid:
        for square in row:
            square.draw(win)
    
    drawGridlines(win, rows, width)
    pygame.display.update()

def get_clicked_position(mouse, rows, width):
    gap = width // rows
    y, x = mouse
    row = y // gap
    col = x // gap
    return row, col

def nn_predict(grid, nn_running):
    nn_running = True
    inputX = []
    for row in grid:
        inputX.append([])
        for col in row:
            if col.nnAlive():
                inputX[-1].append(0.5)
            else:
                inputX[-1].append(0)
    model = load_nn.makeModel()
    print(f'{np.argmax(model.predict([inputX])) + 1}')
    nn_running = False

def main(win, width):
    grid = makeGrid(ROWS, width)
    run = True
    nn_running = False

    while run:
        draw(win, grid, ROWS, width)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif nn_running != True and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    nn_predict(grid, nn_running)

            if not nn_running:
                if pygame.mouse.get_pressed()[0]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_position(pos, ROWS, width)
                    if row < len(grid) and col < len(grid):
                        square = grid[row][col]
                        square.makeAlive()
                elif pygame.mouse.get_pressed()[2]:
                    pos = pygame.mouse.get_pos()
                    row, col = get_clicked_position(pos, ROWS, width)
                    if row < len(grid) and col < len(grid):
                        square = grid[row][col]
                        square.makeDead()
                

    pygame.quit()

if __name__ == '__main__':
    main(WINDOW, WIDTH)