# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
# setup plot details

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

thresholds = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], dtype=np.float64)
tps = np.array([17826, 17557, 16914, 16077, 15054, 13863, 12330, 10475, 8129, 5006, 2836], dtype=np.float64) # 正检
fps = np.array([65275, 39675, 22468, 12002, 6273, 3662, 2268, 1442, 829, 305, 111], dtype=np.float64) # 误检
fns = np.array([1005, 1274, 1917, 2754, 3777, 4968, 6501, 8356, 10702, 13825, 15995], dtype=np.float64) # 漏检

tps2 = np.array([17311, 17035, 16505, 15684, 14679, 13500, 12042, 10312, 8044, 4898, 2591], dtype=np.float64) # 正检
fps2 = np.array([155538, 71986, 28405, 13080, 6929, 4165, 2669, 1694, 971, 363, 119], dtype=np.float64) # 误检
fns2 = np.array([1520, 1796, 2326, 3147, 4158, 5331, 6789,  8519, 10793, 13933, 16240], dtype=np.float64) # 漏检

precision = np.divide(tps, np.add(tps, fps))
recall = np.divide(tps ,np.add(tps, fns))

precision2 = np.divide(tps2, np.add(tps2, fps2))
recall2 = np.divide(tps2 ,np.add(tps2, fns2))

print 'thresholds ',thresholds
print 'recall' , recall
print 'precision' , precision

print 'recall2' , recall2
print 'precision2' , precision2

# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, lw=lw, color='navy',
         label='Precision-Recall curve')
plt.plot(recall,precision,'ro')
plt.plot(recall2, precision2, lw=lw, color='Orange',
         label='Precision-Recall2 curve')
plt.plot(recall2,precision2,'ro')
#画对角线
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc="lower left")
plt.grid()
plt.savefig('PRcurve.png')
plt.show()
