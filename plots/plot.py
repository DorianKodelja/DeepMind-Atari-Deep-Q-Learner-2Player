import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  "Pong2Player.csv",
  "Pong2PlayerVS.csv",
#  "Pong2Player05p.csv",
]

labels = [
  "Cooperative",
  "Competitive",
#  "rho = 0.5",
]

suffixes = [
  "cooperative",
  "competitive",
#  "05p",
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
#f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(4,3))
#ax = plt.subplot2grid((2,2), (0,0), rowspan=2)
#ax1 = plt.subplot2grid((2,2), (0,1))
#ax2 = plt.subplot2grid((2,2), (1,1))
f = plt.figure(figsize=(4,3))
ax = f.add_subplot(111)    # The big subplot
ax1 = f.add_subplot(211)
ax2 = f.add_subplot(212)
for i, file_name in enumerate(csv_files):
  #plt.errorbar(epochs[i], means[i][:,1], yerr=stds[i][:,1])
  ax1.plot(epochs[i], means[i][:,0])
  ax2.plot(epochs[i], means[i][:,0])

ax1.set_ylim(16, 400)
ax2.set_ylim(0, 11)
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop='off')  # don't put tick labels at the top
ax2.xaxis.tick_bottom()
ax1.locator_params(axis='y', nbins=3)
ax2.locator_params(axis='y', nbins=3)

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

ax.set_ylabel("Paddle-bounces per point", labelpad=10)
ax.set_xlabel("Epoch")
#ax = plt.gca()
#ax.set_yscale("log")
lgd = ax1.legend(labels, loc="lower center", bbox_to_anchor=(0.43, 1.),
           ncol=2, columnspacing=1, frameon=False)
plt.tight_layout()
f.subplots_adjust(hspace=0.1)
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

for i, file_name in enumerate(csv_files):
  plt.clf()
  plt.plot(epochs[i][10:], means[i][10:,1])
  plt.locator_params(axis='y', nbins=5)
  plt.ylabel("Wall-bounces per paddle-bounce")
  plt.xlabel("Epoch")
  plt.tight_layout()
  plt.savefig('wallbounces_history_%s.png' % suffixes[i], dpi=dpi)

  plt.clf()
  plt.plot(epochs[i], means[i][:,0])
  plt.locator_params(axis='y', nbins=5)
  plt.ylabel("Paddle-bounces per point")
  plt.xlabel("Epoch")
  #ax = plt.gca()
  #ax.set_yscale("log")
  #plt.ylim(ymin=-10)
  plt.tight_layout()
  plt.savefig('sidebounces_history_%s.png' % suffixes[i], dpi=dpi)

  plt.clf()
  plt.plot(epochs[i], means[i][:,2])
  plt.locator_params(axis='y', nbins=5)
  plt.ylabel("Serving time per point")
  plt.xlabel("Epoch")
  plt.tight_layout()
  plt.savefig('serving_time_history_%s.png' % suffixes[i], dpi=dpi)
