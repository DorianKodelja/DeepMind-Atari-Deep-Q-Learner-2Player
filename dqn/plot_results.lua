require 'nn'
require 'initenv'
require 'cutorch'
require 'gnuplot'

if #arg < 1 then
  print('Usage: ', arg[0], ' <DQN file>')
  return
end

data = torch.load(arg[1])

--gnuplot.raw('set multiplot layout 2, 3')

gnuplot.figure()
gnuplot.title('Average reward per game during testing')
gnuplot.plot(torch.Tensor(data.reward_history))

gnuplot.figure()
gnuplot.title('Total count of rewards during testing')
gnuplot.plot(torch.Tensor(data.reward_counts))

gnuplot.figure()
gnuplot.title('Number of games played during testing')
gnuplot.plot(torch.Tensor(data.episode_counts))

gnuplot.figure()
gnuplot.title('Average Q-value of validation set')
gnuplot.plot(torch.Tensor(data.v_history))

gnuplot.figure()
gnuplot.title('TD error (old and new Q-value difference) of validation set')
gnuplot.plot(torch.Tensor(data.td_history))

gnuplot.figure()
gnuplot.title('Seconds elapsed after epoch')
gnuplot.plot(torch.Tensor(data.time_history))

--gnuplot.figure()
--gnuplot.title('Qmax history')
--gnuplot.plot(torch.Tensor(data.qmax_history))

