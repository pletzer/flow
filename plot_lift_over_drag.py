import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1])

angle = df['alpha']
lift = np.array(df['lift'])
error_lift = 1.96 * np.array(df['std_lift'])
drag = np.array(df['drag'])
error_drag = 1.96 * np.array(df['std_drag'])

fig, ax = plt.subplots()
ax.errorbar(angle, lift/drag, yerr=(lift/drag)*(error_lift/lift + error_drag/drag))
plt.xlabel('attack angle deg')
plt.ylabel('Lift/Drag')
plt.show()