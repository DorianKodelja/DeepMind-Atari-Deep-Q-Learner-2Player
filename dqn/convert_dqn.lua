require 'nn'
require 'cutorch'
require 'image'
require 'msleep'
gd = require "gd"

if #arg < 3 then
  print('Usage: ', arg[0], ' <DQN file> <GIF file> <LOG file>')
  return
end

data = torch.load(arg[1])

print("Total reward: ", data.total_reward)
print("Reward count: ", data.reward_count)
print("Q-value count: ", #data.q_history)
print("Screen count: ", #data.screen_history)

width = data.screen_history[1]:size(4)
height = data.screen_history[1]:size(3)
giffile = arg[2]
logfile = arg[3]

im = gd.create(width, height)
previm = im
im:gifAnimBegin(giffile, false, -1)
print("Animating game...")
log = io.open(logfile, "w")
for i=1,(#data.screen_history) do
  --print("frame ", i)
  jpg = image.compressJPG(data.screen_history[i]:squeeze(), 100)
  im = gd.createFromJpegStr(jpg:storage():string())
  im:gifAnimAdd(giffile, true, 0, 0, 6, gd.DISPOSAL_NONE, previm)
  previm = im
  log:write(data.q_history[i], "\n")
end
gd.gifAnimEnd(giffile)
log:close()
print(string.format("Done animating, gif written to %s, log written to %s.", giffile, logfile))

