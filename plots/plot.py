import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  "Pong2Player.csv",
  "Pong2PlayerVS.csv" 
]

labels = [
  "Cooperative",
  "Competitive"
]

data = []
for i, file_name in enumerate(csv_files):
  data.append(dict())
  with open(file_name, 'rb') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=";")
    csv_reader.next() # skip first row
    for row in csv_reader:
      if not row[0].isdigit():
        continue
      epoch = int(row[0])
      if epoch not in data[i].keys():
        data[i][epoch] = []
      wallbounces = float(row[2])
      sidebounces = float(row[3])
      points = float(row[4])
      servingtime = float(row[5])
      data[i][epoch].append((
        sidebounces / points,
        wallbounces / (wallbounces if sidebounces == 0 else sidebounces),
        servingtime / points / 4
      ))

epochs = []
means = []
stds = []
for i, file_name in enumerate(csv_files):
  epochs.append([])
  means.append([])
  stds.append([])
  for epoch in sorted(data[i].keys()):
    epochs[i].append(int(epoch))
    a = np.array(data[i][epoch])
    mean = np.mean(a, axis=0)
    means[i].append(mean)
    std = np.std(a, axis=0)
    stds[i].append(std)

epochs = [np.array(a) for a in epochs]
means = [np.array(a) for a in means]
stds = [np.array(a) for a in stds]

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['axes.labelsize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'

dpi = 300
plt.figure(figsize=(4,3))

for i, file_name in enumerate(csv_files):
  #plt.errorbar(epochs[i][10:], means[i][10:,1], yerr=stds[i][10:,1])
  plt.plot(epochs[i][10:], means[i][10:,1])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Wall-bounces per paddle-bounce")
plt.xlabel("Epoch")
lgd = plt.legend(labels, loc="lower center", bbox_to_anchor=(0.43, 1.),
           ncol=2, columnspacing=1, frameon=False)
plt.tight_layout()
plt.savefig('wallbounces_history.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.clf()
for i, file_name in enumerate(csv_files):
  #plt.errorbar(epochs[i], means[i][:,1], yerr=stds[i][:,1])
  plt.plot(epochs[i], means[i][:,0])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Paddle-bounces per point")
plt.xlabel("Epoch")
#ax = plt.gca()
#ax.set_yscale("log")
lgd = plt.legend(labels, loc="lower center", bbox_to_anchor=(0.43, 1.),
           ncol=2, columnspacing=1, frameon=False)
plt.tight_layout()
plt.savefig('sidebounces_history.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.clf()
for i, file_name in enumerate(csv_files):
  #plt.errorbar(epochs[i], means[i][:,2], yerr=stds[i][:,2])
  plt.plot(epochs[i], means[i][:,2])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Serving time per point")
plt.xlabel("Epoch")
lgd = plt.legend(labels, loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
plt.tight_layout()
plt.savefig('serving_time_history.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

plt.clf()
plt.plot(epochs[0][10:], means[0][10:,1])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Wall-bounces per paddle-bounce")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig('wallbounces_history_cooperative.png', dpi=dpi)

plt.clf()
plt.plot(epochs[1][10:], means[1][10:,1])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Wall-bounces per paddle-bounce")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig('wallbounces_history_competitive.png', dpi=dpi)

plt.clf()
plt.plot(epochs[0], means[0][:,0])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Paddle-bounces per point")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig('sidebounces_history_cooperative.png', dpi=dpi)

plt.clf()
plt.plot(epochs[1], means[1][:,0])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Paddle-bounces per point")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig('sidebounces_history_competitive.png', dpi=dpi)

plt.clf()
plt.plot(epochs[0], means[0][:,2])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Serving time per point")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig('serving_time_history_cooperative.png', dpi=dpi)

plt.clf()
plt.plot(epochs[1], means[1][:,2])
plt.locator_params(axis='y', nbins=5)
plt.ylabel("Serving time per point")
plt.xlabel("Epoch")
plt.tight_layout()
plt.savefig('serving_time_history_competitive.png', dpi=dpi)
