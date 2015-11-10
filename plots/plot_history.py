import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  "Pong2PlayerVS", 
  "Pong2Player05p",
  "Pong2Player0",
  "Pong2Player025",
  "Pong2Player05",
  "Pong2Player075",
  "Pong2Player",
]

mpl.rcParams['xtick.labelsize'] = 'smaller'
mpl.rcParams['ytick.labelsize'] = 'smaller'
mpl.rcParams['axes.labelsize'] = 'small'
mpl.rcParams['legend.fontsize'] = 'small'

dpi = 300
plt.figure(figsize=(4,3))

data = []
for i, file_name in enumerate(csv_files):
  print file_name
  data_a = np.loadtxt(file_name + "_history_A.csv", delimiter = ",", skiprows = 1)
  data_b = np.loadtxt(file_name + "_history_B.csv", delimiter = ",", skiprows = 1)

  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,1])
  plt.plot(data_b[:,0] / 2, data_b[:,1])
  plt.ylabel("Average reward")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Right player", "Left player"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  #plt.tight_layout()
  plt.savefig(file_name + '_average_reward.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,2])
  plt.plot(data_b[:,0] / 2, data_b[:,2])
  plt.ylabel("Reward count")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Right player", "Left player"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  #plt.tight_layout()
  plt.savefig(file_name + '_reward_counts.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,3])
  plt.plot(data_b[:,0] / 2, data_b[:,3])
  plt.ylabel("Episode count")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Right player", "Left player"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  #plt.tight_layout()
  plt.savefig(file_name + '_episode_counts.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,4])
  plt.plot(data_b[:,0] / 2, data_b[:,4])
  plt.ylabel("Mean Q-value")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Right player", "Left player"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  #plt.tight_layout()
  plt.savefig(file_name + '_meanq.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,5])
  plt.plot(data_b[:,0] / 2, data_b[:,5])
  plt.ylabel("TD error")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Right player", "Left player"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  #plt.tight_layout()
  plt.savefig(file_name + '_tderror.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')

  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,6])
  plt.plot(data_b[:,0] / 2, data_b[:,6])
  plt.ylabel("Time (seconds)")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Right player", "Left player"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  #plt.tight_layout()
  plt.savefig(file_name + '_time.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
