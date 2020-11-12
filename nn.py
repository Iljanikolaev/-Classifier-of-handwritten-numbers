import torch
import torchvision.datasets
import random
import numpy as np

#Фиксируем параметры для воспроизводимости результатов
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

class LeNet5(torch.nn.Module):
    '''Класс нейронной сети'''
    def __init__(self,
                 activation='tanh',
                 pooling='avg',
                 conv_size=5,
                 use_batch_norm=False):
        #Инициализация функции активации, сверточных слоев, ядра свертки, пулинга, батч нормализации
        super(LeNet5, self).__init__()
        
        self.conv_size = conv_size
        self.use_batch_norm = use_batch_norm
        
        if activation == 'tanh':
            activation_function = torch.nn.Tanh()
        elif activation == 'relu':
            activation_function  = torch.nn.ReLU()
        else:
            raise NotImplementedError
            
        if pooling == 'avg':
            pooling_layer = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling == 'max':
            pooling_layer  = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise NotImplementedError
        
        if conv_size == 5:
            self.conv1 = torch.nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, padding=2)
        elif conv_size == 3:
            self.conv1_1 = torch.nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=3, padding=1)
            self.conv1_2 = torch.nn.Conv2d(
                in_channels=6, out_channels=6, kernel_size=3, padding=1)
        else:
            raise NotImplementedError

        self.act1 = activation_function
        self.bn1 = torch.nn.BatchNorm2d(num_features=6)
        self.pool1 = pooling_layer
       
        if conv_size == 5:
            self.conv2 = self.conv2 = torch.nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, padding=0)
        elif conv_size == 3:
            self.conv2_1 = torch.nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=3, padding=0)
            self.conv2_2 = torch.nn.Conv2d(
                in_channels=16, out_channels=16, kernel_size=3, padding=0)
        else:
            raise NotImplementedError

        self.act2 = activation_function
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.pool2 = pooling_layer
        
        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = activation_function
    
        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = activation_function
        #Без softmax, т. к. в качестве лосс-функции используем кросс-энтропию
        self.fc3 = torch.nn.Linear(84, 10)
    
    def forward(self, x):
        #Прямое рапространение
        if self.conv_size == 5:
            x = self.conv1(x)
        elif self.conv_size == 3:
            x = self.conv1_2(self.conv1_1(x))
        x = self.act1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.pool1(x)
        if self.conv_size == 5:
            x = self.conv2(x)
        elif self.conv_size == 3:
            x = self.conv2_2(self.conv2_1(x))
        x = self.act2(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.pool2(x)
        #Преобразуем полученный после сверточных слоев тензор перед входами в полносвязную сеть
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        
        return x
if __name__ == '__main__':

    '''Загружаем датасет mnist(набор изображений рукописных цифр 28x28, 60тыс. - обучающая выборка, 10тыс. - тестовая)'''
    MNIST_train = torchvision.datasets.MNIST('./', download = True, train = True)
    MNIST_test = torchvision.datasets.MNIST('./', download = True, train = False)

    X_train = MNIST_train.data
    y_train = MNIST_train.targets
    X_test = MNIST_test.data
    y_test = MNIST_test.targets

    '''Каждая картинка из mnist размером 28x28,
    преобразуем каждую картинку в трехмерный тензор, добавляем канал(яркость серого пикселя)'''
    X_train = X_train.unsqueeze(1).float()
    X_test = X_test.unsqueeze(1).float()
    def train(net, X_train, y_train, X_test, y_test):
        '''Функция обучения  нейронной сети'''
        #Переносим сеть на ГПУ, если это возможно
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        #Лосс-функия - кросс-энтропия, оптимизатор - Adam
        loss = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3)
	    #размер батча - 100
        batch_size = 100
        #Будем накапливать accuracy(точность) и лосс на тесте
        test_accuracy_history = []
        test_loss_history = []
        #Тестируем на 3000 тыс. примеров из тестовой выборки
        X_test = X_test[:3000].to(device)
        y_test = y_test[:3000].to(device)

        for epoch in range(29):
            #Перемашиваем порядок подачи примеров в сеть
            order = np.random.permutation(len(X_train))
            for start_index in range(0, len(X_train), batch_size):
                #зануляем градиенты
                optimizer.zero_grad()
                #переводим модель в режим обучения
                net.train()
                #формируем индексы для батча
                batch_indexes = order[start_index:start_index+batch_size]
                #переносим батчи на девайс(ГПУ или ЦПУ)
                X_batch = X_train[batch_indexes].to(device)
                y_batch = y_train[batch_indexes].to(device)
                #прямое распространение
                preds = net.forward(X_batch) 
                #считаем значения лосс-функции
                loss_value = loss(preds, y_batch)
                #обратное распространение 
                loss_value.backward()
                #делаем градинтный шаг
                optimizer.step()
            #переводим модель в режим тестирования
            net.eval()
            test_preds = net.forward(X_test)
            test_loss_history.append(loss(test_preds, y_test).data.cpu())
            #В качестве метрики качества используем точность предсказания(accuracy)
            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().data.cpu()
            test_accuracy_history.append(accuracy)

            print(accuracy)
        print('---------------')
        #Сохраняем параметры обученной нейронной сети
        torch.save({
            'epochs' : epoch,
            'model_state_dict' : net.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss
                }, './net.pth')
        return test_accuracy_history, test_loss_history


    #Создаем экземпляр нейронной сети
    neural_network = LeNet5(activation='relu', conv_size=3, pooling='max', use_batch_norm=True)
    #Обучаем нейронную сеть
    train(neural_network, X_train, y_train, X_test, y_test)
    
    

