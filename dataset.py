import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('HandRecognition.csv', index_col = 0) # index_col = randmarks
fig = plt.figure(figsize = (200, 250))
ax = fig.add_subplot(111, projection = '3d')
x = df[' x_dist']
y = df[' y_dist']
z = df[' time']
C = np.random.randint(0, 50, 19400)
ax.scatter(x, y, z, c = C, marker = 'o', cmap = plt.cm.Blues) # scatter: 산점도로 표현, cmap: color map
ax.set_xlabel('Time')
ax.set_ylabel('X')
ax.set_zlabel('Y')

plt.show()