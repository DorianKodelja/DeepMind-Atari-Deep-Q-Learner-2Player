--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

gd = require "gd"
require "math"
if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('TrainAgent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of envirment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history for agent 1')
cmd:option('-nameB', '', 'filename used for saving network and training history for agent 2')
cmd:option('-network', '', 'reload pretrained network for agent 1')
cmd:option('-networkB', '', 'reload pretrained network for agent 2')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 3, 'fixed input seed for repeatable experiments')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-gif_file', '', 'GIF path to write session screens')
cmd:option('-csv_file', '', 'CSV path to write session data')
cmd:option('-version', '', 'epoch of training')
cmd:option('-datas_file', '', 'CSV path to write learning evaluation data')
cmd:text()

local opt = cmd:parse(arg)
local clock = os.clock
--- General setup.
local game_env, game_actions,game_actionsB, agent,agentB, opt,optB = setup2(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local version=opt.version
-- file names from command line
local gif_filename = opt.gif_file
local csv_filename = opt.csv_file
local datas_filename=opt.datas_file
print(gif_filename, csv_filename, datas_filename)

-- start a new game
local screen, rewardA,rewardB, terminal = game_env:newGame2()

-- compress screen to JPEG with 100% quality
local jpg = image.compressJPG(screen:squeeze(), 100)
-- create gd image from JPEG string
local im = gd.createFromJpegStr(jpg:storage():string())
-- convert truecolor to palette
im:trueColorToPalette(false, 256)

-- write GIF header, use global palette and infinite looping
im:gifAnimBegin(gif_filename, true, 0)
-- write first frame
im:gifAnimAdd(gif_filename, false, 0, 0, 7, gd.DISPOSAL_NONE)

-- remember the image and show it first
local previm = im
local win = image.display({image=screen})

-- open CSV file for writing and write header
local csv_file = assert(io.open(csv_filename, "w"))
csv_file:write('actionA;ActionB;max_qvalueA;max_qvalueB;rewardA;rewardB;terminal\n')
local datas_file = assert(io.open(datas_filename, "a+"))
if opt.seed==1 then datas_file:write('training Epoch;Seed;WallBounces;SideBounce;Points;ServingTime;RewardA;RewardB\n') end
print("Started playing...")
previousScore=0
totalSideBounce=0
previousWallBounce=false
totalWallBounce=0
previousSideBounce=0
servingTime=0
totalRewardA = 0
totalRewardB = 0
-- play one episode (game)
while not terminal and not crash do
    -- if action was chosen randomly, Q-value is 0
    agent.bestq = 0
    agentB.bestq = 0
    
    -- choose the best action
    local action_index = agent:perceive(rewardA, screen, terminal, true, 0.01)
    local action_indexB = agentB:perceive(rewardB, screen, terminal, true, 0.01)
    --if agent.bestq == 0 then
    --  print("A random action: " .. action_index)
    --else
    --  print("A agent action: " .. action_index)
    --end
    -- play game in test mode (episodes don't end when losing a life)
    screen, rewardA,rewardB, terminal, sideBouncing,wallBouncing,points,crash,serving = game_env:step2(game_actions[action_index],game_actionsB[action_indexB], false)
    if rewardA ~= 0 or rewardB ~= 0 then
       print(rewardA, rewardB, points)
    end
    totalRewardA = totalRewardA + rewardA
    totalRewardB = totalRewardB + rewardB
    --gather statisticts for one ball
    -- wallbouncing true when the ball is touching the wall, but we want to count only when it turn true
    if (wallBouncing==true and previousWallBounce==false) then
        totalWallBounce=totalWallBounce+1 
    end
    previousWallBounce=wallBouncing
    
    if (previousSideBounce<sideBouncing) then
        totalSideBounce=totalSideBounce+1
    end
    previousSideBounce=sideBouncing
    if(serving==true) then 
    	servingTime=servingTime+opt.actrep 
    end
   
    

    -- display screen
    image.display({image=screen, win=win})

    -- create gd image from tensor
    jpg = image.compressJPG(screen:squeeze(), 100)
    im = gd.createFromJpegStr(jpg:storage():string())
    
    -- use palette from previous (first) image
    im:trueColorToPalette(false, 256)
    im:paletteCopy(previm)

    -- write new GIF frame, no local palette, starting from left-top, 0.06s delay
    im:gifAnimAdd(gif_filename, false, 0, 0, 6, gd.DISPOSAL_NONE)
    -- remember previous screen for optimal compression
    previm = im

    -- write best Q-value for state to CSV file
    csv_file:write(action_index .. ';' ..action_indexB .. ';' .. agent.bestq .. ';' .. agentB.bestq .. ';' .. rewardA .. ';'.. rewardB .. ';' .. tostring(terminal) .. '\n')
    --print(previousScore.." / "..points.." bounce ",totalSideBounce,":"..totalWallBounce)
    
end
print("final "..previousScore.." / "..points.." bounce ",totalSideBounce,":"..totalWallBounce)
datas_file:write(""..version..";"..opt.seed..";"..totalWallBounce..";"..totalSideBounce..";"..points..";"..servingTime..";"..totalRewardA..";"..totalRewardB.."\n")

datas_file:close()

-- end GIF animation and close CSV file
gd.gifAnimEnd(gif_filename)
csv_file:close()

print("Finished playing, close window to exit!")
assert(false)

