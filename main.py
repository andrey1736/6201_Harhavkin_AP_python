import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt

tree = ET.parse('config.xml')
root = tree.getroot()

# Читаем данные из файла в переменные
# Приводим их к типу числа с плавающей точкой из текста

for elem in root.findall('xmin'):
    xmin = float(elem.text)
for elem in root.findall('xmax'):
    xmax = float(elem.text)
for elem in root.findall('step'):
    step = float(elem.text)
for elem in root.findall('a'):
    a = float(elem.text)
for elem in root.findall('b'):
    b = float(elem.text)
for elem in root.findall('c'):
    c = float(elem.text) 

# Открываем файл на запись вычисленных значений
# В цикле одновременно расчитываем и записываем результаты

file = open('results.txt', 'w')
x = xmin
while x <= xmax:
    y = (((np.sin(a*x)*np.cos(b*x)))/(np.sin(x)+np.cos(x)))+c
    file.write(str(y) + '\n')
    x += step
file.close()

# Создаём линейное пространство с заданными параметрами
# Рисуем график и показываем его

x = np.linspace(int(xmin), int(xmax), int((xmax - xmin) / step + 1))
plt.plot(x, (((np.sin(a*x)*np.cos(b*x)))/(np.sin(x)+np.cos(x)))+c)
plt.show()
