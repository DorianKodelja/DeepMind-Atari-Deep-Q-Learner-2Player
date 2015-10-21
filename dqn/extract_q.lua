require 'nn'
require 'initenv'
require 'cutorch'

if #arg < 1 then
  print('Usage: ', arg[0], ' <DQN file>')
  return
end

data = torch.load(arg[1])

for i,v in ipairs(data.v_history) do
  print(i .. ',' .. v)
end

