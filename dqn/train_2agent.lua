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
cmd:option('-nameB', '', 'filename used for saving network and training history for the second player')
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
--local game_env, game_actions, agent, opt = setup2Player(opt,false)
local game_env, game_actions,game_actionsB, agent,agentB, opt,optB = setup2(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local learn_startB = agentB.learn_start
local start_time = sys.clock()
local reward_counts = {}
local reward_countsB = {}
local episode_counts = {}
local episode_countsB = {}
local time_history = {}
local v_history = {}
local v_historyB = {}
local qmax_history = {}
local qmax_historyB = {}
local td_history = {}
local td_historyB = {}
local reward_history = {}
local reward_historyB = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

local total_rewardB
local nrewardsB
local episode_rewardB

local screen, reward,rewardB, terminal = game_env:getState2()

print("Iteration ..", step)
local win = nil
while step < opt.steps do
    step = step + 1
    local action_index = agent:perceive(reward, screen, terminal)
    local action_indexB = agentB:perceive(rewardB, screen, terminal)
    -- game over? get next game!
    if not terminal then
        screen, reward,rewardB, terminal = game_env:step2(game_actions[action_index],game_actionsB[action_indexB], true)
    else
        if opt.random_starts > 0 then
            screen, reward,rewardB, terminal = game_env:nextRandomGame2()
        else
            screen, reward,rewardB, terminal = game_env:newGame2()
        end
    end

    -- display screen
    win = image.display({image=screen, win=win})

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        agentB:report()
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then

        screen, reward,rewardB, terminal = game_env:newGame2()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        total_rewardB = 0
        nrewardsB = 0
        nepisodesB = 0
        episode_rewardB = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
            local action_indexB = agentB:perceive(rewardB, screen, terminal, true, 0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward,rewardB, terminal = game_env:step2(game_actions[action_index],game_actionsB[action_indexB])

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            -- record every reward for player 2
            episode_rewardB = episode_rewardB + rewardB
            if rewardB ~= 0 then
               nrewardsB = nrewardsB + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen, reward, terminal = game_env:nextRandomGame()
            end
            if terminal then
                total_rewardB = total_rewardB + episode_rewardB
                episode_rewardB = 0
                nepisodesB = nepisodesB + 1
                screen, reward,rewardB, terminal = game_env:nextRandomGame2()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        agentB:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end

        local indB = #reward_historyB+1
        total_rewardB = total_rewardB/math.max(1, nepisodesB)

        if #reward_historyB == 0 or total_rewardB > torch.Tensor(reward_historyB):max() then
            agentB.best_network = agent.network:clone()
        end

        if agentB.v_avg then
            v_historyB[indB] = agentB.v_avg
            td_historyB[indB] = agentB.tderr_avg
            qmax_historyB[indB] = agentB.q_max
        end


        print("A: V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])
        print("B: V ", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])
        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes


        reward_historyB[indB] = total_rewardB
        reward_countsB[indB] = nrewardsB
        episode_countsB[indB] = nepisodesB




        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), rewardA: %.2f,rewardB: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards A: %d,num. rewards B: %d',
            step, step*opt.actrep, total_reward, total_rewardB,agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards,nrewardsB))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
	end
        filename = filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
	if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved A:', filename .. '.t7')
        ---save player B
        
        local sB, aB, rB, s2B, termB = agentB.valid_s, agentB.valid_a, agentB.valid_r,
            agentB.valid_s2, agentB.valid_term
        agentB.valid_s, agentB.valid_a, agentB.valid_r, agentB.valid_s2,
            agentB.valid_term = nil, nil, nil, nil, nil, nil, nil
        local wB, dwB, gB, g2B, deltaB, delta2B, deltasB, tmpB = agentB.w, agentB.dw,
            agentB.g, agentB.g2, agentB.delta, agentB.delta2, agentB.deltas, agentB.tmp
        agentB.w, agentB.dw, agentB.g, agentB.g2, agentB.delta, agentB.delta2,
            agentB.deltas, agentB.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filenameB = optB.name
        if optB.save_versions > 0 then
            filenameB = filenameB .. "_" .. math.floor(step / optB.save_versions)
        end
        filenameB = filenameB
        torch.save(filenameB .. ".t7", {agent = agentB,
                                model = agentB.network,
                                best_model = agentB.best_network,
                                reward_history = reward_historyB,
                                reward_counts = reward_countsB,
                                episode_counts = episode_countsB,
                                time_history = time_history,
                                v_history = v_historyB,
                                td_history = td_historyB,
                                qmax_history = qmax_historyB,
                                arguments=opt})
        if optB.saveNetworkParams then
            local netsB = {network=wB:clone():float()}
            torch.save(filenameB..'.params.t7', nets, 'ascii')
        end
        agentB.valid_s, agentB.valid_a, agentB.valid_rB, agentB.valid_s2,
            agentB.valid_term = sB, aB, rB, s2B, termB
        agentB.w, agentB.dw, agentB.g, agentB.g2, agentB.delta, agentB.delta2,
            agentB.deltas, agentB.tmp = wB, dwB, gB, g2B, deltaB, delta2B, deltasB, tmpB
        print('Saved B:', filenameB .. '.t7')
        io.flush()
        collectgarbage()
    end
end
