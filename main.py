import numpy as np
import matplotlib.pyplot as plt
import random

#Сигмоида в качестве активационной функции нейронов
def sigmoid(z):
    return 1/(1+np.exp(-z))
#Производная сигмоиды
def sigmoid_prime(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))
#Функция потерь, та самая 1/2*(y-y_k)^2, только для всей обучающей выборки
def cost_function(network, test_data):
    c = 0
    for example, y in test_data:
        yhat = network.feedforward(example)
        c+=np.sum((y-yhat)*(y-yhat))/2
    return c/len(test_data)


class Network():
    # инициализация сети: размеры, количество слоев, смещения biases, веса weights
    def __init__(self, sizes, output=True):

        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[
                                                      1:]]  # инициализация случайных смещений, векторов размерности (n,1), где n-количество нейронов в слое
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        # инициализация случайных весов, матриц размерности (m,n)
        # где n-кол-во нейронов в слое из которого они выходят, m-кол-во нейронов в слое, в который они входят
        self.output = output

    # Вычисление выходной активации при заданных входных данных(по сути прогон сети вперед)
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    # реализация стохастического градиентного спуска, отличие в том, что мы будем считать градиент функции потерь не по всей выборке
    # а по случайной ее части, размер этой случайной части задается параметром mini_batch_size
    # можно задать mini_batch_size, равный объему обучающей выборки, тогда получится просто градиентный спуск, но ухудшится сходимость к локальному минимуму
    # epochs-количество шагов градиентного спуска
    # eta - обучающий коэффицент(усваиваемость информации), чтобы мы могли корректировать длину шага градиентного спуска
    def SGD(self, training_data, epochs, mini_batch_size, eta):
        n = len(training_data)
        for j in range(epochs):
            random_data = np.copy(training_data)
            random_data.flags.writeable = True
            random.shuffle(random_data)  # 3 строки выше просто делают копию обучающих данных и перемешивают ее
            mini_batches = [
                random_data[k:k + mini_batch_size]
                for k in range(0, n,
                               mini_batch_size)]  # здесь мы случайно выбираем подвыборки размера mini_batch_size, на основе которых будем считать градиент
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  # вызываем метод для обновления весов и смещений

    # обновление весов и смещений
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # инициализация градиента смещений для каждого из слоев
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # инициализация градиента весов для каждого из слоев
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x,
                                                         y)  # вызов метода обратного распространения ошибки для подсчета градиента
            nabla_b = [nb + dnb for nb, dnb in
                       zip(nabla_b, delta_nabla_b)]  # вычисление градиента смещений для каждого из слоев
            nabla_w = [nw + dnw for nw, dnw in
                       zip(nabla_w, delta_nabla_w)]  # вычисление градиента весов для каждого из слоев

        eps = eta / len(
            mini_batch)  # eps назовем коэффициентом длины шага, если eps будет слишком большим, то будем проходить мимо минимумов
        # если же будет слишком малым, то мы до минимума и не дойдем,а зависит он от соотношения обучающего коэфф-та и объема подвыборки
        self.weights = [w - eps * nw for w, nw in
                        zip(self.weights, nabla_w)]  # изменение весов на основе градиента(делаем шаг к минимуму)
        self.biases = [b - eps * nb for b, nb in
                       zip(self.biases, nabla_b)]  # изменение смещений на основе градиента(делаем шаг к минимуму)

    # метод обратного распространения ошибки, с его помощью считаем градиенты весов и смещений
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # инициализация градиента смещений для каждого из слоев
        nabla_w = [np.zeros(w.shape) for w in self.weights]  # инициализация градиента весов для каждого из слоев
        # прямое распространение (forward pass)

        activation = x  # входные данные сети
        activations = [x]
        zs = []  # активации на выходном слое
        # в этом цикле считаем выходные активации
        for b, w in zip(self.biases, self.weights):
            # посчитать активации
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
            activations.append(activation)
            zs.append(z)
            pass

        # обратное распространение (backward pass)
        delta = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_prime(zs[-1]))
        nabla_b[-1] = delta  # градиент функции потерь по смещениям выходного слоя
        nabla_w[-1] = np.dot(delta, activations[-2].T)  # градиент функции потерь по весам выходного слоя
        # в этом цикле считаем градиент для смещений и активаций каждого слоя кроме последнего, ибо он у нас уже посчитан
        for l in range(2, self.num_layers):
            # дополнительные вычисления, чтобы легче записывалось
            #
            delta = np.multiply(np.dot(self.weights[-l + 1].T, delta), sigmoid_prime(zs[-l]))  # ошибка на слое L-l
            nabla_b[-l] = delta  # производная J по смещениям L-l-го слоя
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)  # производная J по весам L-l-го слоя
        return nabla_b, nabla_w

    def cost_derivative(self, output_activations, y):
        """
        Возвращает градиент выходного слоя
        """
        return (output_activations - y)


x_train = 99 * np.random.rand(100,
                              1) + 1  # это обучающая выборка, чуть позже мы объединяем x и у, чтобы отдать на вход сети (размер можно менять, в данном случае тут 100 элементов)
y_train = 1 / x_train

x_test = 99 * np.random.rand(200,
                             1) + 1  # это выборка для тестирования (размер можно менять, в данном случае тут 200 элементов)
y_test = 1 / x_test

epochs = int(input('Введите количество шагов алгоритма(пример 300): '))  # кол-во шагов
batch_size = int(input('Введите размер подвыборки(пример 5): '))  # размер подвыборки, на основе которой будет считаться градиент функции потерь
eta = int(input('Введите коэффицент обучаемости (пример 5): ')) # обучающий коэффициент, на самом деле коэффициент шага будет зависеть и от batch_size и будет выглядеть eta/batch_size, так что стоит рассматривать два последних параметра вместе


for i in range(1, 15):
    data_train = np.concatenate((x_train, y_train), 1)  # объединяем x c y  для подачи в класс сети


    nn = Network([1, i,
                  1])  # инициализация сети [кол-во нейронов во входном слое, кол-во нейронов в скрытом слое, кол-во нейронов в выходном слое]
    nn.SGD(data_train, epochs, batch_size, eta)  # запуск градиентного спуска

    x = np.arange(1.0, 100.0, 0.01)  # здесь мы создаем x и y  чисто для того, чтобы начертить график
    y = 1 / x

    y_pred = np.array([nn.feedforward(x) for x in x_test.flatten()])  # предсказания сети на выборке для тестирования

    error = 100 * np.sum(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) / len(
        y_test.flatten())  # считаем относительную ошибку аппроксимации по формуле sum(|(y-y_предсказанное)/y|)/кол-во

    print('Количество нейронов в скрытом слое:{0}'.format(i))
    print('Количество эпох:{0}'.format(epochs))
    print('Количество наблюдений в выборке для поиска градиента:{0}'.format(batch_size))
    print('Обучающий коэффициент:{0}'.format(eta))
    print('Ошибка:{:.2f}%'.format(error))

    fig, ax = plt.subplots()
    ax.plot(x, y)  # строим график функции

    ax.set(xlabel='x', ylabel='1/x', title='y=1/x слои:{0} шаги:{1} коэф.об-ти:{2} выборка:{3} ошибка:{4:.2f}%'.format(i,epochs,eta, batch_size,error))
    ax.grid()

    plt.scatter(x_test, y_pred)  # добавляем на график точки, полученные от нейросети
    plt.scatter(x_test, y_test)  # для контраста добавляем реальные значения функции в этих точках
    plt.show()  # отображаем график
    fig.savefig('plots/first_approx_{0}layers_{1}epochs_{2}lr_{3}batchsize.png'.format(i,epochs,eta, batch_size), dpi=300)


epochs = int(input('Введите количество шагов алгоритма(пример 300): '))  # кол-во шагов
batch_size = int(input('Введите размер подвыборки(пример 5): '))  # размер подвыборки, на основе которой будет считаться градиент функции потерь
eta = int(input('Введите коэффицент обучаемости (пример 5): ')) # обучающий коэффициент, на самом деле коэффициент шага будет зависеть и от batch_size и будет выглядеть eta/batch_size, так что стоит рассматривать два последних параметра вместе

x_train = 9 * np.random.rand(100, 1) + 1
y_train = np.exp((-1) * x_train)

x_test = 9 * np.random.rand(200, 1) + 1
y_test = np.exp((-1) * x_test)

for i in range(1, 15):
    data_train = np.concatenate((x_train, y_train), 1)

    nn = Network([1, i, 1])
    nn.SGD(data_train, epochs, batch_size, eta)

    x = np.arange(1.0, 10.0, 0.01)
    y = np.exp((-1) * x)

    y_pred = np.array([nn.feedforward(x) for x in x_test.flatten()])
    error = 100 * np.sum(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) / len(y_test.flatten())
    print('Количество нейронов в скрытом слое:{0}'.format(i))
    print('Количество эпох:{0}'.format(epochs))
    print('Количество наблюдений в выборке для поиска градиента:{0}'.format(batch_size))
    print('Обучающий коэффициент:{0}'.format(eta))
    print('Ошибка:{:.2f}%'.format(error))

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='x', ylabel='y', title='y=e^(-x) слои:{0} шаги:{1} коэф.об-ти:{2} выборка:{3} ошибка:{4:.2f}%'.format(i,epochs,eta, batch_size,error))
    ax.grid()

    plt.scatter(x_test, y_pred)
    plt.scatter(x_test, y_test)
    plt.show()
    fig.savefig('plots/second_approx_{0}layers_{1}epochs_{2}lr_{3}batchsize.png'.format(i,epochs,eta, batch_size), dpi=300)

x_train = (np.pi / 2 - 1) * np.random.rand(100, 1) + 1
y_train = np.cos(x_train)

x_test = (np.pi / 2 - 1) * np.random.rand(20, 1) + 1
y_test = np.cos(x_test)

epochs = int(input('Введите количество шагов алгоритма(пример 300): '))  # кол-во шагов
batch_size = int(input('Введите размер подвыборки(пример 5): '))  # размер подвыборки, на основе которой будет считаться градиент функции потерь
eta = int(input('Введите коэффицент обучаемости (пример 5): ')) # обучающий коэффициент, на самом деле коэффициент шага будет зависеть и от batch_size и будет выглядеть eta/batch_size, так что стоит рассматривать два последних параметра вместе


for i in range(3, 10):
    data_train = np.concatenate((x_train, y_train), 1)

    nn = Network([1, i, 1])
    nn.SGD(data_train, epochs, batch_size, eta)

    x = np.arange(1.0, 1.58, 0.01)
    y = np.cos(x)

    y_pred = np.array([nn.feedforward(x) for x in x_test.flatten()])
    error = 100 * np.sum(np.abs((y_pred.flatten() - y_test.flatten()) / y_test.flatten())) / len(y_test.flatten())
    print('Количество нейронов в скрытом слое:{0}'.format(i))
    print('Количество эпох:{0}'.format(epochs))
    print('Количество наблюдений в выборке для поиска градиента:{0}'.format(batch_size))
    print('Обучающий коэффициент:{0}'.format(eta))
    print('Ошибка:{:.2f}%'.format(error))

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel='x', ylabel='y', title='y=cos(x) слои:{0} шаги:{1} коэф.об-ти:{2} выборка:{3} ошибка:{4:.2f}%'.format(i,epochs,eta, batch_size,error))
    ax.grid()

    plt.scatter(x_test, y_pred)
    plt.scatter(x_test, y_test)
    plt.show()
    fig.savefig('plots/third_approx_{0}layers_{1}epochs_{2}lr_{3}batchsize.png'.format(i,epochs,eta, batch_size), dpi=300)


epochs = int(input('Введите количество шагов алгоритма(пример 300): '))  # кол-во шагов
batch_size = int(input('Введите размер подвыборки(пример 5): '))  # размер подвыборки, на основе которой будет считаться градиент функции потерь
eta = int(input('Введите коэффицент обучаемости (пример 5): ')) # обучающий коэффициент, на самом деле коэффициент шага будет зависеть и от batch_size и будет выглядеть eta/batch_size, так что стоит рассматривать два последних параметра вместе

# создаем обучающую и тестировочную выборку
x1_train = 2 * np.random.rand(100) - 1  # случайные значения для x1 в отрезке [-1;1]
x2_train = 2 * np.random.rand(100) - 1  # случайные значения для x2 в отрезке [-1;1]
f_train = 0.5 * np.sin(np.pi * x1_train * x2_train) * np.sin(2 * np.pi * x2_train * x2_train)

x1_test = 2 * np.random.rand(100) - 1
x2_test = 2 * np.random.rand(100) - 1
f_test = 0.5 * np.sin(np.pi * x1_test * x2_test) * np.sin(2 * np.pi * x2_test * x2_test)

data_train = [(np.array([x1, x2]).reshape(2, 1), f) for x1, x2, f in zip(x1_train, x2_train,
                                                                         f_train)]  # класс сети принимает на вход кортеж (x,y) вот здесь мы и создаем кортежи такого вида
data_test = [(np.array([x1, x2]).reshape(2, 1), f) for x1, x2, f in zip(x1_test, x2_test, f_test)]

for i in range(3, 10):

    nn = Network([2, i, 1])  # инициализация сети, в этот раз на входе две переменные
    nn.SGD(data_train, epochs, batch_size, eta)  # Запуск градиентного спуска

    f_pred = np.array([nn.feedforward(test[0]) for test in data_test])  # предсказания сети считаются также

    error = 100 * np.sum(np.abs((f_pred.flatten() - f_test.flatten()) / f_test.flatten())) / len(
        f_test.flatten())  # ошибка считается также

    print('Количество нейронов в скрытом слое:{0}'.format(i))
    print('Количество эпох:{0}'.format(epochs))
    print('Количество наблюдений в выборке для поиска градиента:{0}'.format(batch_size))
    print('Обучающий коэффициент:{0}'.format(eta))
    print('Ошибка:{:.2f}%'.format(error))

    x1 = np.arange(-1, 1, 0.01)  # множество значений x1 для построения поверхности
    x2 = np.arange(-1, 1, 0.01)  # множество значений x2 для построения поверхности
    x1, x2 = np.meshgrid(x1, x2)  # сетка

    f = 0.5 * np.sin(np.pi * x1 * x2) * np.sin(2 * np.pi * x2 * x2)  # значение функции

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x1, x2, f, vmin=f.min() * 2)  # строим поверхность
    ax.scatter(x1_test, x2_test, f_pred, c='red')  # предсказанные точки
    ax.scatter(x1_test, x2_test, f_test, c='green')  # реальные точки

    ax.set(xticklabels=[],
           yticklabels=[],
           zticklabels=[])

    plt.show()
    fig.savefig('plots/fourth_approx_{0}layers_{1}epochs_{2}lr_{3}batchsize.png'.format(i,epochs,eta, batch_size), dpi=300)