Классификатор рукописных цифр.

Приложение при помощи обученной нейронной сети распознает нарисованную пользователем цифру.

Интерфейс состоит из доски для рисования(черная область) и информационного табла.
Управление:
1)Чтобы начать рисовать необходимо зажать правую или левую кнопу мыши на доске для рисования.
2)Enter - на вход нейронной сети подается нарисованное пользоватлем цифра,
	на выходе сеть возвращает предсказанный класс(цифру) и вероятности отнесения к каждому из классов.
	В приложении предсказание  красным цветом выводится вверху доски для рисования.
	Вероятности выводятся на информационном табле.
3)Пробел - очистить доску для рисования
4)TAB - режим дообучения сети. В случае ошибки классификатора, сеть можно дообучить на собственных примерах, при нажатии TAB
цифры на информационном табле подсвечиваются зеленым цветом, необходимо нажать на верный класс(цифру); обучение происходит на
10 изображениях(дополнительные изображения получаются аугментацией исходного изображения).

Для запуска приложения используется файл class.py.
Для запуска обучения нейронной сети файл nn.py.


Нейронная сеть была обучена на датасете MNIST, точность(accuracy) на тестовом множестве составила 0.9935, 
далее дообучена конечными пользователями приложения.
MNIST - датасет изображений рукописных чисел(60 тыс. - для обучения, 10 тыс. - для теста), 
набор образцов взят из Бюро перепеси США и дополнен образцами из амереканских университетов.
В США некоторые цифры пишут не совсем так, как в России(об этом можно почитать в интернете), 
также в датасете MNIST достаточно много "неудачных" примеров, 
именно поэтому сеть была дообучена ещё на дополнительных примерах пользователей.

Пользователь может изменить архитектуру нейронной сети и поиграться с параметрами в файле nn.py и обучить собственную нейронную сеть.
