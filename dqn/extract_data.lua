require 'nn'
require 'initenv'
require 'cutorch'

if #arg < 1 then
  print('Usage: ', arg[0], ' <DQN file>')
  return
end

data = torch.load(arg[1])
print("Epoch,Average reward,Reward count,Episode count,MeanQ,TD Error,Seconds")
for i=1,#data.v_history do
  print(table.concat({i, data.reward_history[i], data.reward_counts[i], 
	data.episode_counts[i], data.v_history[i], data.td_history[i], data.time_history[i]},','))
end

