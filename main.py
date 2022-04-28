from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from numpy import asarray
from matplotlib import pyplot
import numpy as np
import sympy as sym
from sympy import Symbol, nsolve
import math
import mpmath

# определим набор данных

x = []
y = []
yk = []
k = 0
output = 0

z = []

# Найдем размер массивов x и y
# x = 0
y_max1 = 0.08*6000
y_max2 = 0.01*2000
y_max = min(y_max1, y_max2)

# y = 0
x_max1 = 0.08*3000
x_max2 = 0.01*3000
x_max = min(x_max1, x_max2)
size_x_y = max(y_max, x_max)

xn = 0
yn = 0

N = int(8*size_x_y)
h = size_x_y/N

for i in range(N+1):
    for j in range(N+1):
        ind1 = float(xn / 3000 + yn / 6000)
        ind2 = float(xn / 3000 + yn / 2000)
        if ind1 <= 0.08:
            if ind2 <= 0.01:
                x.append(xn)
                y.append(yn)
        yn = yn + h
    xn = xn + h
    yn = 0

r1 = []
r2 = []
for i in range(len(x)):
    r1.append(x[i]/3000 + y[i]/6000 - 0.08)
    r2.append(x[i]/3000 + y[i]/2000 - 0.01)

# y0 = 0
# x0 = 0
#for i in range(N):
#    for j in range(N):
#        ind1 = float(xn / 3000 + yn / 6000)
#        ind2 = float(xn / 3000 + yn / 2000)
#        if ind1 <= 0.08:
#            if ind2 <= 0.01:
#                x0 = xn
#                if yn > y0:
#                    y0 = yn
#        yn = yn + h
#    x.append(x0)
#    y.append(y0)
#    x0 = 0
#    y0 = 0
#    xn = xn + h
#    yn = 0
#print(x)
#print(y)

m = 0
for i in range(len(x)):
    if 15*x[i]+10*y[i] > output:
        output = 15*x[i] + 10*y[i]
        k = x[i]
        m = y[i]
    z.append(15*x[i] + 10*y[i])

print(output)
print(k)
print(m)

#print(z)

# преобразовываем массивы
input_array=[]
for i in range (len(x)):
    b=[]
    b.append(x[i])
    b.append(y[i])
    input_array.append(b)


z = np.array(z).reshape((len(z), 1))

#print(z)

input_array = np.array(input_array).reshape((len(input_array), 2))

#print(input_array)


# design the neural network model (разработка модели нейронной сети)
model = Sequential()
model.add(Dense(5, input_dim=2, activation='sigmoid', kernel_initializer='he_uniform'))
# model.add(Dense(10, input_shape=(3,), activation='sigmoid', kernel_initializer='he_uniform'))

#model.add(Dense(5, activation='sigmoid', kernel_initializer='he_uniform'))
model.add(Dense(1))

# define the loss function and optimization algorithm (определить функцию потерь и алгоритм оптимизации)
model.compile(loss='mse', optimizer='adam')

# ft the model on the training dataset (подгонка модели на обучающем датасете)
# model.fit(input_array, z, epochs=500, batch_size=10, verbose=0)
model.fit(input_array, z, epochs=250, batch_size=10, verbose=0)

# make predictions for the input data (сделать прогнозы для входных данных)
#zhat = model.predict(input_array)
zhat = model.predict(input_array).squeeze()

print(zhat)

# report model error (сообщить об ошибке модели)
print('MSE: %.3f' % mean_squared_error(z, zhat))

# вывод весовых тензоров
net_weight = model.get_weights() # возвращает список всех весовых тензоров в модели в виде массивов Numpy
print(net_weight)
# model.to_json()
print(net_weight[0][0][0]) # число до [0][0][4]
print(net_weight[0][1][0]) # число до [0][1][4]
print(net_weight[1][0]) # число до [1][4]
print(net_weight[2][0][0]) # число до [2][4][0]
print(net_weight[3][0]) # число [3][0]

weight_array01 = []
for i in range(5):
    weight_array01.append(net_weight[0][0][i])
print(weight_array01)

weight_array02 = []
for i in range(5):
    weight_array02.append(net_weight[0][1][i])
print(weight_array02)

weight_array1 = []
for i in range(5):
    weight_array1.append(net_weight[1][i])
print(weight_array1)

weight_array2 = []
for i in range(5):
    weight_array2.append(net_weight[2][i][0])
print(weight_array2)

weight_array3 = []
weight_array3.append(net_weight[3][0])
print(weight_array3)

fi = 0
F = 0

mpmath.mp.dps = 15
X = Symbol('X')
Y = Symbol('Y')
V1 = Symbol('V1')
V2 = Symbol('V2')
for j in range(5):
    fi = fi + weight_array01[j]*X
    fi = fi + weight_array02[j]*Y
    fi = fi + weight_array2[j]
    fi = 1/(1+pow(math.e, (-1)*fi))
    F = F + weight_array1[j]*fi
    fi = 0
F = F + weight_array3[0]
F = F + ((1/3000)*X + (1/6000)*Y - 0.08) * V1
F = F + ((1/3000)*X + (1/2000)*Y - 0.01) * V2
f1 = sym.diff(F, X)
f2 = sym.diff(F, Y)
f3 = sym.diff(F, V1)
f4 = sym.diff(F, V2)
print(f1)
print(f2)
print(f3)
print(f4)
eqs = [f1, f2, f3, f4]
answ = nsolve(eqs, [X, Y, V1, V2], [0, 0, 0, 1])
print(answ)

fi = 0
F = 0
for j in range(5):
    fi = fi + weight_array01[j]*answ[0]
    fi = fi + weight_array02[j]*answ[1]
    fi = fi + weight_array2[j]
    fi = 1/(1+pow(math.e, (-1)*fi))
    F = F + weight_array1[j]*fi
    fi = 0
F = F + weight_array3[0]
F = F + ((1/3000)*answ[0] + (1/6000)*answ[1] - 0.08) * answ[2]
F = F + ((1/3000)*answ[0] + (1/2000)*answ[1] - 0.01) * answ[3]
print(F)


# Примеры символьного дифференцирования и нахождения предела
#sym.diff(sym.sin(x), x) # результат cos(?)
#sym.diff(sym.sin(2 * x), x) # результат 2cos(2?)
#sym.diff(sym.tan(x), x)
#sym.limit((sym.tan(x + y) - sym.tan(x)) / y, y, 0)
#sym.diff(sym.sin(2 * x), x, 1) # результат 2cos(2?)
#sym.diff(sym.sin(2 * x), x, 2) # результат −4sin(2?)
#sym.diff(sym.sin(2 * x), x, 3) # результат −8cos(2?)

# pow(math.e, x)


# создание графика
# plot x vs y
#pyplot.scatter(x_plot, y_plot, label='Actual')

# plot x vs yhat
#pyplot.scatter(x_plot, yhat_plot, label='Predicted')
#pyplot.title('Input (x) versus Output (y)')
#pyplot.xlabel('Input Variable (x)')
#pyplot.ylabel('Output Variable (y)')
#pyplot.legend()
#pyplot.show()

# reshape arrays into into rows and cols (преобразование массивов в строки)
#x = x.reshape((len(x), 1))
#y = y.reshape((len(y), 1))

# separately scale the input and output variables (масштабируем данные)
#scale_x = MinMaxScaler()
#x = scale_x.fit_transform(x)
#scale_y = MinMaxScaler()
#y = scale_y.fit_transform(y)
#print(x.min(), x.max(), y.min(), y.max())

#scale_z = MinMaxScaler()
#z = scale_z.fit_transform(z)

#scale_input = MinMaxScaler()
#input_array = scale_z.fit_transform(input_array)

#input_array[i][0] = x[i]
#input_array[i][1] = y[i]
#x = np.array(x).reshape((len(x), 1))
#y = np.array(y).reshape((len(y), 1))

# inverse transforms (обратное масштабирование)
#x_plot = scale_x.inverse_transform(x)
#y_plot = scale_y.inverse_transform(y)
#yhat_plot = scale_y.inverse_transform(yhat)

#z_plot = scale_z.inverse_transform(z)
#yhat_plot = scale_y.inverse_transform(yhat)

# xi = list([i for i in range(0, 100)])
# yi = list([i for i in range(0, 100)])

#for i in range(len(xi)):
#    for j in range(len(yi)):
#        ind1 = float(xi[i] / 3000 + yi[j] / 6000)
#        ind2 = float(xi[i] / 3000 + yi[j] / 2000)
#        if ind1 <= 0.08:
#            if ind2 <= 0.01:
#                x.append(xi[i])
#                y.append(yi[j])
#print(x)
#print(y)