import torch
import torchvision.transforms
from nn import LeNet5
import os.path
import PIL 
import math

#Если нет файла c параметрами обученной сети - ошибка
if not os.path.isfile('./net.pth'):
    raise FileNotFoundError('Обучите нейронную сеть.')

'''Обработка нарисованного пользователем изображения перед подачей в сеть, 
rbg каналы конвертируем в один канал gray и представляем изображение в виде тензора'''
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.ToTensor(),
])

#Кo выходу сети применим SoftMax, чтобы получить интерпретируемые вероятности отнесения к классам
soft_max = torch.nn.Softmax(dim = 1)

def net_predict(img_path, bb_color):
    '''Функция для предсказания изображенной пользователем цифры'''
    #Загружаем обученную сеть
    net = load_trained_network()
    #Переводим сеть в режим тестирования
    net.eval()
    #Делаем изображение размером 28x28 c сохраннием пропорций
    #Загружаем изображение
    img = PIL.Image.open(img_path)
    #Получаем границу ненулевых областей на изображении
    img_box = img.getbbox()
    if img_box != None:
        #Обрезаем картинку по границе
        img = img.crop(img_box)
        #Максимальный параметр размера(ширина или высота) картинки делаем равным  20,
        #меньший параметр изменяем с сохранением пропорций
        w, h= img.size
        img = img.resize((int(20 * w/max(h,w)),  int(20*h/max(h,w))), PIL.Image.LANCZOS)
        #Считаем и добавляем рамку(центрируем изображение)
        box = (math.ceil((28 - img.size[0])/2), math.ceil((28 - img.size[1])/2))
        img = PIL.ImageOps.expand(img, box, bb_color)
        #Обрезаем 28x28
        img = img.crop((0, 0, 28, 28))
    else:
        img = img.resize((28, 28), PIL.Image.NEAREST)
    #Сохраняем изображение(можно не сохранять) и преобразуем его перед подачей в сеть
    img.save('./num_img.png')
    img = transforms(img)
    #Добавляем ещё канал, т.к в сеть подавали батчами
    img=img.unsqueeze(0).float()
    #Переходим в диапазон 0-255
    img = img*255
    #Подаем в сеть
    return net.forward(img).argmax(dim=1), soft_max(net.forward(img))


def retrain(im_path, num):
    '''Функция дообучения нейронной сети'''
    #Загружаем обученную сеть, оптимизатор и функцию потерь.
    net, optimizer, loss, epochs = load_trained_network(net_train = True)
    #Аугментация. Увеличиваем обучающий набор данных.
    aug_img = image_augmentation(im_path)
    #Создаем вектор из правильных ответов
    y = torch.full((len(aug_img), ), float(num)).long()
    '''Начинаем обучение сети'''
    #зануляем градиенты
    optimizer.zero_grad()
    #переводим модель в режим обучения
    net.train()
    #Вычисляем предсказания классов
    preds = net.forward(aug_img) 
    #считаем значения лосс-функции
    loss_value = loss(preds, y)
    #обратное распространение 
    loss_value.backward()
    #делаем градинтный шаг
    optimizer.step()
    #переводим модель в режим тестирования
    net.eval()
    #Вычисляем предсказания на тесте
    test_preds = net.forward(aug_img)
    #В качестве метрики качества используем точность предсказания(accuracy)
    accuracy = (test_preds.argmax(dim=1) == y).float().mean().data.cpu()
    print('Нейронная сеть обучилась, accuracy = ' + str(accuracy.item()))
    #Сохраняем параметры сети
    torch.save({
            'epochs' : epochs + 1,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss
                }, './net.pth')


def load_trained_network(net_train = False):
    '''Загружает обученную сеть'''
    #Cеть будет работать на ЦПУ
    device = torch.device('cpu')   
    #Создаем экземпляр LeNet5, загружаем веса обученной сети
    net = LeNet5(activation='relu', conv_size=3, pooling='max', use_batch_norm=True)
    #Загружаем словарь с парметрами обученной нейронной сети
    chekpoint = torch.load('./net.pth', map_location = device)
    #Загружаем парметры обученной сети
    net.load_state_dict(chekpoint['model_state_dict'])
    if net_train:
        #Создаем экземпляр оптимизатора
        optimizer = torch.optim.Adam(net.parameters(), lr = 1.0e-3)
        #Загружаем парметры оптимизатора, функцию потерь и количество эпох
        optimizer.load_state_dict(chekpoint['optimizer_state_dict'])
        loss = chekpoint['loss']
        epochs = chekpoint['epochs']
        return net, optimizer, loss, epochs
    return net

def image_augmentation(im_path):
    ''' 
    Аугментация изображения.
    Преобразует исходное изображение различными способами(сжатие, растяжение, поворот), сохраняет в список.
    Случайно возвращает 9 изменненых изображений + 1 исходное.
    '''
    #Загружаем изображение
    img = PIL.Image.open(im_path)
    #Обрезаем картинку по границе ненулевых областей
    img = img.crop(img.getbbox())
    #Создаем список, где будут храниться исходные и измененные значения ширины и высоты изоображения
    resized = [[img.size[0]], [img.size[1]]]
    #Максимальный параметр((max(ширина,высота)) размера уменьшаем в 1.3 раза и добавляем в список
    resized[img.size.index(max(img.size))].append(max(img.size) // 1.3)
    #Минимальный параметр размера(min(ширина, высота)) увеличиваем в 1.3 раза и добавляем в список
    resized[img.size.index(min(img.size))].append(min(min(img.size) * 1.3, 20))
    #Список, где будут хранится исходное и измененные изображения(сжатие, растяжение, поворот на определенный угол исходное изображение)
    aug_img= []
    #Преобразуем исходное изображение(сжимаем, растягиваем, поворачиваем)
    for i in range(len(resized[0])):
        for j in range(len(resized[1])):
            #Изменяем размеры исходного изображения
            new_img = img.resize((int(resized[0][i]), int(resized[1][j])), PIL.Image.LANCZOS)
            #Достраиваем новое изображение до размера 28x28
            box = (math.ceil((28 - new_img.size[0])/2), math.ceil((28 - new_img.size[1])/2))
            new_img = PIL.ImageOps.expand(new_img, box, (0,0,0))
            new_img = new_img.crop((0, 0, 28, 28))
            #Поворачиваем новое изображение от -20 до 20 грудусов с шагом 5 и сохраняем и переводим в тензор
            for k in range(-20, 21, 5):
                #Первоначальное изображение сохраним отдельно
                if (i == 0) and (j == 0) and (k == 0):
                    orig_image = transforms(new_img)*255
                    continue
                aug_img.append(transforms(new_img.rotate(k, PIL.Image.BICUBIC))*255) 
    #Объеденяем получившиеся тензоры в один
    aug_img= torch.stack(aug_img, dim = 0)
    #Случайно выбираем 10 изображений 
    aug_img = aug_img[torch.randperm(len(aug_img))[:9]] 
    #Добавляем первоначальное изображение
    aug_img = torch.cat((orig_image.unsqueeze(0), aug_img), dim = 0) 
    return aug_img