#!/usr/bin/python

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  ("Pong2PlayerVS.csv", "Competitive (x=1)"), 
  ("Pong2Player0.csv", "Transition (x=0)"),
  ("Pong2Player025.csv", "Transition (x=-0.25)"), 
  ("Pong2Player05.csv", "Transition (x=-0.5)"), 
  ("Pong2Player075.csv", "Transition (x=-0.75)"),
  ("Pong2Player.csv", "Cooperative (x=-1)"),
]

stats = []
with open("stats.tex","w") as output: 
    output.write("\hline\n")
    output.write("Agent & Average paddletouches per point & Average wallbounces per paddletouch & Average serving time per point \\\\\n")
    output.write("\hline\n")
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
          avgServingTime = map(lambda x, y: x / y, servingTime, points)

          data = (label, np.mean(sideBouncePerPoint), np.std(sideBouncePerPoint),
              np.mean(wallBouncePerSideBounce), np.std(wallBouncePerSideBounce),
              np.mean(avgServingTime), np.std(avgServingTime))
          stats.append(data[1:])

          output.write("%s & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f \\\\\n" % data)
          output.write("\hline\n")

stats = np.array(stats)
x = range(1, stats.shape[0] + 1)
labels = [
  r'$\rho$' + ' = 1\n(competitive)',
  '0',
  '-0.25',
  '-0.5',
  '-0.75',
  '-1\n(cooperative)',
]

dpi = 300
plt.figure(figsize=(4,3))

plt.errorbar(x, stats[:,0], yerr=stats[:,1])
plt.ylabel("Paddletouches per point")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.xlim((0, len(x) + 1))
plt.tight_layout()
plt.savefig('sidebounces_per_point.png', dpi=dpi)

plt.clf()
plt.errorbar(x, stats[:,2], yerr=stats[:,3])
plt.ylabel("Wallbounces per paddletouch")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.xlim((0, len(x) + 1))
plt.tight_layout()
plt.savefig('wallbounces_per_sidebounce.png', dpi=dpi)

plt.clf()
plt.errorbar(x, stats[:,4], yerr=stats[:,5])
plt.ylabel("Serving time per point")
#ax = plt.gca()
#ax.set_yscale("log")
plt.xticks(x, labels)
plt.xlim((0, len(x) + 1))
plt.tight_layout()
plt.savefig('serving_time_per_point.png', dpi=dpi)
