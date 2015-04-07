--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:text()

local opt = cmd:parse(arg)

--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local q_history = {}
local screen_history = {}

local total_reward = 0
local nrewards = 0

local screen, reward, terminal = game_env:newGame()

local win = nil

-- play one episode (game)
while not terminal do
    --agent.bestq = 0

    local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

    q_history[#q_history + 1] = agent.bestq
    screen_history[#screen_history + 1] = screen:clone()

    -- display screen
    win = image.display({image=screen, win=win})

    -- Play game in test mode (episodes don't end when losing a life)
    screen, reward, terminal = game_env:step(game_actions[action_index])

    -- record every reward
    total_reward = total_reward + reward
    if reward ~= 0 then
       nrewards = nrewards + 1
    end
end

print("Finished playing, saving results...")

local filename = opt.name
torch.save(filename .. "_test.t7", {
                        total_reward = total_reward,
                        reward_count = nrewards,
                        q_history = q_history,
                        screen_history = screen_history,
                        })
print("Finished saving, flushing...")
io.flush()
print("Finished flushing, collecting garbage...")
collectgarbage()
print("Finished collecting garbage, all done!")
