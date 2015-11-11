#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  ("Pong2PlayerVS.csv", "Competitive $\\rho=1$"), 
  ("Pong2Player05p.csv", "Transition $\\rho=0.5$"),
  ("Pong2Player0.csv", "Transition $\\rho=0$"),
  ("Pong2Player025.csv", "Transition $\\rho=-0.25$"), 
  ("Pong2Player05.csv", "Transition $\\rho=-0.5$"), 
  ("Pong2Player075.csv", "Transition $\\rho=-0.75$"),
  ("Pong2Player.csv", "Cooperative $\\rho=-1$"),
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

stats = []
with open("stats.tex","w") as output: 
    #output.write("\hline\n")
    #output.write("Agent & Average paddle-bounces per point & Average wall-bounces per paddle-bounce & Average serving time per point \\\\\n")
    #output.write("\hline\n")
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

          sideBouncePerPoint = map(lambda x, y: x / y, sideBounce, points)
          wallBouncePerSideBounce = map(lambda x, y: x / y, wallBounce, sideBounce)
          avgServingTime = map(lambda x, y: x / y / 4, servingTime, points)

          data = (label, np.mean(sideBouncePerPoint), np.std(sideBouncePerPoint),
              np.mean(wallBouncePerSideBounce), np.std(wallBouncePerSideBounce),
              np.mean(avgServingTime), np.std(avgServingTime))
          stats.append(data[1:])

          output.write("%s & $ %.3f \pm %.3f $ & $ %.3f \pm %.3f $ & $ %.3f \pm %.3f $ \\\\\n" % data)
          #output.write("\hline\n")

stats = np.array(stats)
x = range(1, stats.shape[0] + 1)

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['axes.labelsize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'

dpi = 300
plt.figure(figsize=(4,3))

plt.errorbar(x, stats[:,0], yerr=stats[:,1])
plt.ylabel("Paddle-bounces per point")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.locator_params(axis='y', nbins=5)
plt.xlim((0, len(x) + 1))
plt.tight_layout()
plt.savefig('sidebounces_per_point.png', dpi=dpi)

plt.clf()
plt.errorbar(x, stats[:,2], yerr=stats[:,3])
plt.ylabel("Wall-bounces per paddle-bounce")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.locator_params(axis='y', nbins=5)
plt.xlim((0, len(x) + 1))
plt.tight_layout()
plt.savefig('wallbounces_per_sidebounce.png', dpi=dpi)

plt.clf()
plt.errorbar(x, stats[:,4], yerr=stats[:,5])
plt.ylabel("Serving time per point")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.locator_params(axis='y', nbins=5)
plt.xlim((0, len(x) + 1))
plt.tight_layout()
plt.savefig('serving_time_per_point.png', dpi=dpi)
