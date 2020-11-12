import pygame 

class InfoBoard():
	
	'''Класс для вывода информационного табла'''
	def __init__(self, screen, ai_settings):
		'''Инициализирование атрибутов табла'''
		self.screen = screen
		self.ai_settings = ai_settings
		#Объект предсказанной цифры
		self.pred_num= None
		#Флаг неверного предсказания(В случае ошибки классификатора, пользователь нажимает "TAB", флаг присваивает значение True)
		self.wrong_predict = False
		#Создаем объекты информационного табла
		self.info_obj()
		#Создаем кнопки с цифрами для ввода верного ответа, в случае ошибки классификатора
		self.buttons_true_answer()


	def info_obj(self):
		'''Создание объектов информационного табла (статичные)'''
		#Создание поверхности с надписью "Класс" и объекта rect
		self.title_class = self.ai_settings.title_font.render('Класс', True, self.ai_settings.title_color, self.ai_settings.bg_color)
		self.title_class_rect = self.title_class.get_rect()
		#Надпись "Класс" раполагается сверху, справа от доски для рисования
		self.title_class_rect.top = 5
		self.title_class_rect.left = self.ai_settings.blackboard_width 
		#Создание поверхности с надписью "Вероятность" и объекта rect
		self.title_probability = self.ai_settings.title_font.render('Вероятность', True, self.ai_settings.title_color, self.ai_settings.bg_color)
		self.title_probability_rect = self.title_probability.get_rect()
		#Надпись "Вероятность" раполагается сверху, справа от надписи "Класс"
		self.title_probability_rect.top = 5
		self.title_probability_rect.left = self.title_class_rect.right + 5
		#Словарь для хранения объектов(поверхностей с надписями и rect) цифр(классов)
		self.num = {}
		for i in range(10):
			#Создание поверхностей с цифрами и rect
			self.num[i] = [self.ai_settings.number_font.render(str(i), True, self.ai_settings.number_color, self.ai_settings.bg_color)]
			self.num[i].append(self.num[i][0].get_rect())
			#Цифры расположены под надписью "Класс"
			self.num[i][-1].left = self.ai_settings.blackboard_width + 20
			self.num[i][-1].top = self.ai_settings.screen_height//11*(1 + i)


	def prob_obj(self, prediction, probability):
		'''Создание объектов надписи предсказанного числа и надписей вероятностей'''
		#Создает красную надпись предсказанного класса(цифры) в центре доски для рисования
		self.pred_num = [self.ai_settings.number_font.render(str(prediction), True, (255, 0, 0))]
		self.pred_num.append(self.pred_num[0].get_rect())
		self.pred_num[1].right = self.ai_settings.blackboard_width//2
		self.pred_num[1].top = 5 
		#Словарь для хранения объектов(поверхностей с надписями и rect) вероятностей отнесения к классу
		self.prob = {}
		for i in range(len(probability)):
			#Создание поверхностей с вероятностями и rect
		    self.prob[i] = [self.ai_settings.number_font.render(str(format(probability[i], '.3f')), True, self.ai_settings.number_color, self.ai_settings.bg_color)]
		    self.prob[i].append(self.prob[i][0].get_rect())
		   	#Вероятности расположены под надписью "Вероятность"
		    self.prob[i][-1].left = self.num[0][-1].right + 40
		    self.prob[i][-1].top = self.ai_settings.screen_height//11*(1 + i)


	def buttons_true_answer(self):
		'''Создание кнопок с цифрами для ввода верного класса в случае ошибки классификатора'''
		#Словарь для хранения объектов кнопок(поверхностей с надписями и rect)
		self.button = {}
		for i in range(10):
			#Создание поверхностей с цифрами(зеленого цвета) и rect
			self.button[i] = [self.ai_settings.number_font.render(str(i), True, (0, 255, 100), self.ai_settings.bg_color)]
			self.button[i].append(self.button[i][0].get_rect())
			#Кнопки располагаются на месте цифр информационного табла
			self.button[i][-1].left = self.ai_settings.blackboard_width + 20
			self.button[i][-1].top = self.ai_settings.screen_height//11*(1 + i)


	def show_info(self):
		'''Выводит информацинное табло на экран'''
		#Вывод цифры
		for i in self.num.values():
			self.screen.blit(i[0], i[1])
		#Выводит надписи "Класс" и "Вероятность"
		self.screen.blit(self.title_class, self.title_class_rect)
		self.screen.blit(self.title_probability, self.title_probability_rect)
		#Выводит предсказанный класс и вероятности отнесения к классу
		if self.pred_num:
			self.screen.blit(self.pred_num[0], self.pred_num[1])
			for i in self.prob.values():
				self.screen.blit(i[0], i[1])
		#Выводит кнопки с цифрами для ввода верного ответа
		if self.wrong_predict == True:
			for i in self.button.values():
				self.screen.blit(i[0], i[1])
