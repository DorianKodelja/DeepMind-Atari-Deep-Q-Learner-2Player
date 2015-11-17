#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  ("Pong2PlayerVS.csv", "Competitive $\rho=1$"), 
  ("Pong2Player05p.csv", "Transition $\rho=0.5$"),
  ("Pong2Player0.csv", "Transition $\rho=0$"),
  ("Pong2Player025.csv", "Transition $\rho=-0.25$"), 
  ("Pong2Player05.csv", "Transition $\rho=-0.5$"), 
  ("Pong2Player075.csv", "Transition $\rho=-0.75$"),
  ("Pong2Player.csv", "Cooperative $\rho=-1$"),
]

labels = [
  r'$\rho = 1$' + '\n(competitive)',
  r'$0.5$',
  r'$0$',
  r'$-0.25$',
  r'$-0.5$',
  r'$-0.75$',
  r'$-1$' + '\n(cooperative)',
]

sideBouncePerPoint = []
wallBouncePerSideBounce = []
avgServingTime = []
for csv_file, label in csv_files:
  with open(csv_file, 'rb') as input:
    csv_reader = csv.reader(input, delimiter=";")
    csv_reader.next() # skip first row

    sideBounce = []
    wallBounce = []
    points = []
    servingTime = []

    for values in csv_reader:
      if not values[0].isdigit():
        continue
      if int(values[0]) != 49:
        continue

      wallBounce.append(float(values[2]))
      sideBounce.append(float(values[3]))
      points.append(float(values[4]))
      servingTime.append(float(values[5]))

    sideBouncePerPoint.append(map(lambda x, y: x / y, sideBounce, points))
    wallBouncePerSideBounce.append(map(lambda x, y: x / y, wallBounce, sideBounce))
    avgServingTime.append(map(lambda x, y: x / y / 4, servingTime, points))

sideBouncePerPoint = np.array(sideBouncePerPoint)
wallBouncePerSideBounce = np.array(wallBouncePerSideBounce)
avgServingTime = np.array(avgServingTime)
x = range(sideBouncePerPoint.shape[0]) * sideBouncePerPoint.shape[1]
#x += 0.02 * np.random.randn(len(x))

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['axes.labelsize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'

dpi = 300
plt.figure(figsize=(4,3))

plt.scatter(x, sideBouncePerPoint.T.ravel(), alpha = 0.5)
plt.ylabel("Paddle-bounces per point")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.locator_params(axis='y', nbins=7)
plt.xlim((-1, sideBouncePerPoint.shape[0]))
plt.ylim(ymin=-400)
plt.tight_layout()
plt.savefig('sidebounces_per_point_scatter.png', dpi=dpi)

plt.clf()
plt.scatter(x, wallBouncePerSideBounce.T.ravel(), alpha = 0.5)
plt.ylabel("Wall-bounces per paddle-bounce")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.locator_params(axis='y', nbins=7)
plt.xlim((-1, wallBouncePerSideBounce.shape[0]))
plt.ylim(ymin=-0.1)
plt.tight_layout()
plt.savefig('wallbounces_per_sidebounce_scatter.png', dpi=dpi)

plt.clf()
plt.scatter(x, avgServingTime.T.ravel(), alpha = 0.5)
plt.ylabel("Serving time per point")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.locator_params(axis='y', nbins=7)
plt.xlim((-1, avgServingTime.shape[0]))
plt.ylim(ymin=-100)
plt.tight_layout()
plt.savefig('serving_time_per_point_scatter.png', dpi=dpi)
