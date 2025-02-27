import xml.etree.ElementTree as ET
import numpy as np
from matplotlib import pyplot as plt

tree = ET.parse('config.xml')
root = tree.getroot()

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

file = open('results.txt', 'w')
masX = np.arange(xmin, xmax, step)
masY = (((np.sin(a*masX)*np.cos(b*masX)))/(np.sin(masX)+np.cos(masX)))+c
for y in masY:
    file.write(str(y)+ "\n")
file.close()

plt.plot(masX,masY)
plt.show()
