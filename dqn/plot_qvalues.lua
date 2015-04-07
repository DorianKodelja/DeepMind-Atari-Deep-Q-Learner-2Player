require 'nn'
require 'cutorch'
require 'gnuplot'

if #arg < 1 then
  print('Usage: ', arg[0], ' <DQN file>')
  return
end

data = torch.load(arg[1])

print("Total reward: ", data.total_reward)
print("Reward count: ", data.reward_count)
print("Q-value count: ", #data.q_history)
print("Screen count: ", #data.screen_history)

gnuplot.figure()
gnuplot.title('Q-value history')
gnuplot.plot(torch.Tensor(data.q_history))

