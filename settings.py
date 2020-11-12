import pygame
class Settings():
	'''Класс для хранения всех настроек классификатора рукописных чисел'''
	def __init__(self):
		'''Инициализирует настройки классификатора'''

		#Параметры экрана
		self.screen_width = 415
		self.screen_height = 250
		self.bg_color = (100, 150, 255)

		#Параметры доски для рисования
		self.blackboard_width = 250
		self.blackboard_height = 250
		self.blackboard_color = (0, 0, 0)

		#Параметры мелка(карандаша)
		self.crayon_color = (255, 255, 255)
		self.crayon_width = 4

		#Параметры надписей(цвет, шрифт)
		self.title_color = (30, 30, 30)
		self.title_font = pygame.font.SysFont(None, 25)

		#Параметры цифр(цвет, шрифт)
		self.number_color = (30, 30, 30)
		self.number_font = pygame.font.SysFont(None, 35)