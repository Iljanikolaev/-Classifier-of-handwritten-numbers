import pygame
import sys
from paint import Paint
from infoboard import InfoBoard
from settings import Settings

def run_programm():
	#Инициализация модулей pygame, создание экземпляра Settings и создание объекта экрана
	pygame.init()
	ai_settings = Settings()
	screen = pygame.display.set_mode((ai_settings.screen_width, ai_settings.screen_height))
	screen.fill(ai_settings.bg_color)
	pygame.display.set_caption('Классификатор рукописных цифр')
	
	#Создание экземпляров InfoBoard и Paint
	infob = InfoBoard(screen, ai_settings)
	paint = Paint(screen, ai_settings, infob)

	#Фиксируем fps
	clock = pygame.time.Clock()
	fps = 60
	
	#Запуск основного цикла классификатора
	while True:
		clock.tick(fps)
		#Отображения доски рисования
		paint.blit_bb()
		#Вывод основной информации классификатора
		infob.show_info()
		#Отслеживание событий мыши и клавиатуры
		paint.check_event()
		#Обновление экрана
		pygame.display.flip()
run_programm() 

