import pygame
import sys
from nn_pred import net_predict, retrain

class Paint():
	"""Класс для рисования"""
	def __init__(self, screen, ai_settings, infob):
		'''Инициализирует атрибуты Paint'''
		self.screen = screen
		self.ai_settings = ai_settings
		self.infob = infob
		#Создание доски для рисования
		self.blackboard = pygame.Surface((ai_settings.blackboard_width, ai_settings.blackboard_height))
		self.blackboard.fill(ai_settings.blackboard_color)
		#Флаг рисования
		self.draw_on = False

	def check_event(self):
		'''Обрабатывает нажатия отпускания клавиш и события мыши'''
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				#Закрытие программы при нажатиии крестика
				sys.exit()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				#Рисование при нажатии клавиши мыши
				self.draw_on = True
				pygame.draw.circle(self.blackboard, self.ai_settings.crayon_color, pygame.mouse.get_pos(),self.ai_settings.crayon_width)
				if self.infob.wrong_predict == True:
					#В случае ошибки классификатора(нажатии клавиши TAB) - проверка нажатия кнопки с верным классом
					self.check_button(pygame.mouse.get_pos())
			elif event.type == pygame.MOUSEBUTTONUP:
				#Рисование прекращается при отпускании кнопки мыши
				self.draw_on = False
			elif event.type == pygame.MOUSEMOTION:
				#Рисование при перемещении курсора с нажатой кнопкой мыши
				if self.draw_on:
					self.draw()
				#Сохраняем координаты текущего положения курсора
				self.start_pos = pygame.mouse.get_pos()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_SPACE:
					self.clear_bb()
				elif event.key == pygame.K_RETURN:
					'''K_RETURN - клавиша Enter'''
					self.predict()
				elif event.key == pygame.K_TAB:
					'''Пользователь, в случае ошибки классификатора, может обучить сеть на собственном примере'''
					self.infob.wrong_predict =True

	def draw(self):
		'''Функция рисования'''
		#Сохраняем координаты последнего положения курсора и считаем разницу с начальным положением
		last_pos =  pygame.mouse.get_pos()
		dx = last_pos[0] - self.start_pos[0]
		dy= last_pos[1] - self.start_pos[1]
		#Находим по какой оси разница координта положений максимальна
		distance = max(abs(dx), abs(dy))
		#Разбиваем траекторию от нач. до кон. положения курсора мыши на части
		for i in range(distance):
			x = int(self.start_pos[0] + float(i)/distance*dx)
			y = int(self.start_pos[1] + float(i)/distance*dy)
			#Рисуем круг
			pygame.draw.circle(self.blackboard, self.ai_settings.crayon_color, [x, y],self.ai_settings.crayon_width)

	def predict(self):
		#Очистка информационного табла
		self.screen.fill(self.ai_settings.bg_color)
		self.infob.wrong_predict = False
		#сохранение рисунка
		pygame.image.save(self.blackboard, 'num_img.png')
		#получение предсказанного класса и вероятностей отнесения к классам
		prediction, probability = net_predict('./num_img.png', self.ai_settings.blackboard_color)
		#создание объектов надписей предсказания и вероятностей
		prediction = str(int(prediction))
		probability = [round(i, 3) for i in probability[0].cpu().tolist()]
		self.infob.prob_obj(prediction, probability)

	def clear_bb(self):
		#Очищаем доску для рисования, забываем предсказанное значение, убираем режим неверного предсказания
		self.infob.pred_num = None
		self.infob.wrong_predict = False
		self.blackboard.fill(self.ai_settings.blackboard_color)

	def check_button(self, pos):
		'''Функция проверяет была ли нажата кнопка с верным классом(цифрой) и обучает сеть на новом примере'''
		for i in range(10):
			#Проверка нажатия на цифру(верный класс)
			if self.infob.button[i][1].collidepoint(pos[0], pos[1]):
				#Обводка красным выбранной цифры и обновление экрана
				pygame.draw.rect(self.screen, (255, 0, 100), self.infob.button[i][1], 1)
				pygame.display.flip()
				#Изображение и правильный ответ подаются в сеть для обучения
				retrain('./num_img.png', i)
				#Выключаем режим "неверного предсказания"
				self.infob.wrong_predict = False

	def blit_bb(self):
		#Выводит доску для рисования на экран
		self.screen.blit(self.blackboard, (0, 0))
		
		
		
