import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from itertools import cycle
# setup plot details

colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2

thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99], dtype=np.float64)
tps = np.array([17557, 16914, 16077, 15054, 13863, 12042, 10312, ], dtype=np.float64)
fps = np.array([39675, 22468, 12002, 6273, 3662, 2669, 1694, ], dtype=np.float64)
fns = np.array([1274, 1917, 2754, 3777, 4968, 6789, 8519, ], dtype=np.float64)


precision = np.divide(tps, np.add(tps, fps))
recall = np.divide(tps ,np.add(tps, fns))
print(thresholds)
print(precision)
print(recall)



# Plot Precision-Recall curve
plt.clf()
plt.plot(recall, precision, lw=lw, color='navy',
         label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall')
plt.legend(loc="lower left")
plt.show()
