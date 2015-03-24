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
gnuplot.title('Reward history')
gnuplot.plot(torch.Tensor(data.reward_history))

gnuplot.figure()
gnuplot.title('Reward counts')
gnuplot.plot(torch.Tensor(data.reward_counts))

gnuplot.figure()
gnuplot.title('Episode counts')
gnuplot.plot(torch.Tensor(data.episode_counts))

--gnuplot.figure()
--gnuplot.title('Time history')
--gnuplot.plot(torch.Tensor(data.time_history))

gnuplot.figure()
gnuplot.title('V history')
gnuplot.plot(torch.Tensor(data.v_history))

gnuplot.figure()
gnuplot.title('TD history')
gnuplot.plot(torch.Tensor(data.td_history))

gnuplot.figure()
gnuplot.title('Qmax history')
gnuplot.plot(torch.Tensor(data.qmax_history))

