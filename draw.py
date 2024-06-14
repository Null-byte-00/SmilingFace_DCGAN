import pygame
import os

pygame.init()
screen = pygame.display.set_mode((500,500))

def get_next_name(dir, filename):
    numbers = [0]
    for file in os.listdir(dir):
        numbers.append(int(file.replace(filename, '').replace('.png', '')))
    return os.path.join(dir ,filename + str(max(numbers) + 1) + ".png")


loop = True
drawing = False
while loop:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            loop = False
        if event.type == pygame.MOUSEMOTION:
            pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                pygame.image.save(screen, get_next_name('dataset/smilingface', 'smilingface'))
                screen.fill((0,0,0))
            if event.key == pygame.K_d:
                screen.fill((0,0,0))
    if drawing:
        pygame.draw.circle(screen, (255,255,255), pygame.mouse.get_pos(), 5)
    pygame.display.update()