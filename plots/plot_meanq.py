import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

csv_files = [
  "Pong2PlayerVS", 
  "Pong2Player0",
  "Pong2Player025",
  "Pong2Player05",
  "Pong2Player075",
  "Pong2Player",
]

dpi = 300
plt.figure(figsize=(4,3))

data = []
for i, file_name in enumerate(csv_files):
  print file_name
  data_a = np.loadtxt(file_name + "_meanq_A.csv", delimiter = ",")
  data_b = np.loadtxt(file_name + "_meanq_B.csv", delimiter = ",")
  plt.clf()
  plt.plot(data_a[:,0] / 2, data_a[:,1])
  plt.plot(data_b[:,0] / 2, data_b[:,1])
  plt.ylabel("Mean Q-value")
  plt.xlabel("Epoch")
  lgd = plt.legend(["Player A", "Player B"], loc="lower center", bbox_to_anchor=(0.41, 1.),
           ncol=2, columnspacing=1, frameon=False)
  plt.tight_layout()
  plt.savefig(file_name + '_meanq.png', dpi=dpi, bbox_extra_artists=(lgd,), bbox_inches='tight')
