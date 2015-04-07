require 'nn'
require 'cutorch'
require 'image'
require 'msleep'

if #arg < 1 then
  print('Usage: ', arg[0], ' <DQN file>')
  return
end

data = torch.load(arg[1])

print("Total reward: ", data.total_reward)
print("Reward count: ", data.reward_count)
print("Q-value count: ", #data.q_history)
print("Screen count: ", #data.screen_history)

local win = nil

print("Animating game:")
for i=1,(#data.screen_history) do
  print("frame ", i)
  win = image.display({image=data.screen_history[i], win=win})
  msleep(100)
end
print("Done animating")

