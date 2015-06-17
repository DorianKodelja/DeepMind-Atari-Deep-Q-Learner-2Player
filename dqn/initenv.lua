--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
dqn = {}

require 'torch'
require 'nn'
require 'nngraph'
require 'nnutils'
require 'image'
require 'Scale'
require 'NeuralQLearner'
require 'TransitionTable'
require 'Rectifier'


function torchSetup(_opt)
    _opt = _opt or {}
    local opt = table.copy(_opt)
    assert(opt)

    -- preprocess options:
    --- convert options strings to tables
    if opt.pool_frms then
        opt.pool_frms = str_to_table(opt.pool_frms)
    end
    if opt.env_params then
        opt.env_params = str_to_table(opt.env_params)
    end
    if opt.agent_params then
        opt.agent_params = str_to_table(opt.agent_params)
        opt.agent_params.gpu       = opt.gpu
        opt.agent_params.best      = opt.best
        opt.agent_params.verbose   = opt.verbose
        if opt.network ~= '' then
            opt.agent_params.network = opt.network
        end
    end

    --- general setup
    opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(opt.tensorType)
    if not opt.threads then
        opt.threads = 4
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end

    --- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    math.random = nil
    opt.seed = opt.seed or 1
    torch.manualSeed(opt.seed)
    if opt.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if opt.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if opt.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    return opt
end

function torchSetup2(_opt,_optB)
    _opt = _opt or {}

    local opt = table.copy(_opt)
    
    assert(opt)

    _optB = _optB or {}

    local optB = table.copy(_optB)
    
    assert(optB)

    

    -- preprocess options:
    --- convert options strings to tables
    if opt.pool_frms then
        opt.pool_frms = str_to_table(opt.pool_frms)
    end
    if opt.env_params then
        opt.env_params = str_to_table(opt.env_params)
    end
    if opt.agent_params then
        opt.agent_params = str_to_table(opt.agent_params)
        opt.agent_params.gpu       = opt.gpu
        opt.agent_params.best      = opt.best
        opt.agent_params.verbose   = opt.verbose
        if opt.network ~= '' then
            opt.agent_params.network = opt.network
        end
    end

   -- preprocess options for player 2:
    --- convert options strings to tables
    if optB.pool_frms then
        optB.pool_frms = str_to_table(optB.pool_frms)
    end
    if optB.env_params then
        optB.env_params = str_to_table(optB.env_params)
    end
    if optB.agent_params then
        optB.agent_params = str_to_table(optB.agent_params)
        optB.agent_params.gpu       = optB.gpu
        optB.agent_params.best      = optB.best
        optB.agent_params.verbose   = optB.verbose
        if optB.network ~= '' then
            optB.agent_params.network = optB.network
        end
    end

    --- general setup
    opt.tensorType =  opt.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(opt.tensorType)
    if not opt.threads then
        opt.threads = 4
    end
    torch.setnumthreads(opt.threads)
    if not opt.verbose then
        opt.verbose = 10
    end
    if opt.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end


-- general setup for player 2
    optB.tensorType =  optB.tensorType or 'torch.FloatTensor'
    torch.setdefaulttensortype(optB.tensorType)
    if not optB.threads then
        optB.threads = 4
    end
    torch.setnumthreads(optB.threads)
    if not optB.verbose then
        optB.verbose = 10
    end
    if optB.verbose >= 1 then
        print('Torch Threads:', torch.getnumthreads())
    end

    --- set gpu device
    if opt.gpu and opt.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if opt.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then opt.gpu = gpu_id+1 end
        end
        if opt.gpu > 0 then cutorch.setDevice(opt.gpu) end
        opt.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        opt.gpu = -1
        if opt.verbose >= 1 then
            print('Using CPU code only. GPU device id:', opt.gpu)
        end
    end


    --- set gpu device for player 2
    if optB.gpu and optB.gpu >= 0 then
        require 'cutorch'
        require 'cunn'
        if optB.gpu == 0 then
            local gpu_id = tonumber(os.getenv('GPU_ID'))
            if gpu_id then optB.gpu = gpu_id+1 end
        end
        if optB.gpu > 0 then cutorch.setDevice(optB.gpu) end
        optB.gpu = cutorch.getDevice()
        print('Using GPU device id:', opt.gpu-1)
    else
        optB.gpu = -1
        if optB.verbose >= 1 then
            print('Using CPU code only. GPU device id:', optB.gpu)
        end
    end

    --- set up random number generators
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    math.random = nil
    opt.seed = opt.seed or 1
    torch.manualSeed(opt.seed)
    if opt.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if opt.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if opt.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

--- set up random number generators for player 2
    -- removing lua RNG; seeding torch RNG with opt.seed and setting cutorch
    -- RNG seed to the first uniform random int32 from the previous RNG;
    -- this is preferred because using the same seed for both generators
    -- may introduce correlations; we assume that both torch RNGs ensure
    -- adequate dispersion for different seeds.
    math.random = nil
    optB.seed = optB.seed or 2
    torch.manualSeed(optB.seed)
    if optB.verbose >= 1 then
        print('Torch Seed:', torch.initialSeed())
    end
    local firstRandInt = torch.random()
    if optB.gpu >= 0 then
        cutorch.manualSeed(firstRandInt)
        if optB.verbose >= 1 then
            print('CUTorch Seed:', cutorch.initialSeed())
        end
    end

    return opt,optB
end


function setup(_opt)
    assert(_opt)

    --preprocess options:
    --- convert options strings to tables
    _opt.pool_frms = str_to_table(_opt.pool_frms)
    _opt.env_params = str_to_table(_opt.env_params)
    _opt.agent_params = str_to_table(_opt.agent_params)
    if _opt.agent_params.transition_params then
        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end

    --- first things first
    local opt = torchSetup(_opt)

    -- load training framework and environment
    local framework = require(opt.framework)
    assert(framework)

    local gameEnv = framework.GameEnvironment(opt)
    local gameActions = gameEnv:getActions()

    -- agent options
    _opt.agent_params.actions   = gameActions
    _opt.agent_params.gpu       = _opt.gpu
    _opt.agent_params.best      = _opt.best
    if _opt.network ~= '' then
        _opt.agent_params.network = _opt.network
    end
    _opt.agent_params.verbose = _opt.verbose
    if not _opt.agent_params.state_dim then
        _opt.agent_params.state_dim = gameEnv:nObsFeature()
    end

    local agent = dqn[_opt.agent](_opt.agent_params)

    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
    end

    return gameEnv, gameActions, agent, opt
end


-- two player setup
function setup2(_opt)
    assert(_opt)
    
    --preprocess options:
    --- convert options strings to tables
    _opt.pool_frms = str_to_table(_opt.pool_frms)
    _opt.env_params = str_to_table(_opt.env_params)
    _opt.agent_params = str_to_table(_opt.agent_params)
    if _opt.agent_params.transition_params then
        _opt.agent_params.transition_params =
            str_to_table(_opt.agent_params.transition_params)
    end
    local _optB = _opt
    _optB.name=_optB.nameB

    --preprocess options for player 2:
    --- convert options strings to tables
    _optB.pool_frms = str_to_table(_optB.pool_frms)
    _optB.env_params = str_to_table(_optB.env_params)
    _optB.agent_params = str_to_table(_optB.agent_params)
    if _optB.agent_params.transition_params then
        _optB.agent_params.transition_params =
            str_to_table(_optB.agent_params.transition_params)
    end

    --- first things first
    local opt = torchSetup2(_opt,_optB)


    -- load training framework and environment
    local framework = require(opt.framework)
    assert(framework)

    local gameEnv = framework.GameEnvironment(opt)
    local gameActions = gameEnv:getActions()
    local gameActionsB = gameEnv:getActionsB()

    -- agent options
    _opt.agent_params.actions   = gameActions
    _opt.agent_params.gpu       = _opt.gpu
    _opt.agent_params.best      = _opt.best
    if _opt.network ~= '' then
        _opt.agent_params.network = _opt.network
    end
    _opt.agent_params.verbose = _opt.verbose
    if not _opt.agent_params.state_dim then
        _opt.agent_params.state_dim = gameEnv:nObsFeature()
    end

    local agent = dqn[_opt.agent](_opt.agent_params)

    if opt.verbose >= 1 then
        print('Set up Torch using these options:')
        for k, v in pairs(opt) do
            print(k, v)
        end
    end

    -- agent 2 options
    _optB.agent_params.actions   = gameActionsB
    _optB.agent_params.gpu       = _optB.gpu
    _optB.agent_params.best      = _optB.best
    if _optB.network ~= '' then
        _optB.agent_params.network = _optB.network
    end
    _optB.agent_params.verbose = _optB.verbose
    if not _opt.agent_params.state_dim then
        _optB.agent_params.state_dim = gameEnv:nObsFeature()
    end

    local agent = dqn[_opt.agent](_opt.agent_params)
    local agentB = dqn[_optB.agent](_optB.agent_params)
    if opt.verbose >= 1 then
        print('Set up Torch using these options for agent 1:')
        for k, v in pairs(opt) do
            print(k, v)
        end
        print('Set up Torch using these options for agent 2:')
        for k, v in pairs(optB) do
            print(k, v)
        end
    end





    return gameEnv, gameActions,gameActionsB, agent,agentB, opt,optB
end



--- other functions

function str_to_table(str)
    if type(str) == 'table' then
        return str
    end
    if not str or type(str) ~= 'string' then
        if type(str) == 'table' then
            return str
        end
        return {}
    end
    local ttr
    if str ~= '' then
        local ttx=tt
        loadstring('tt = {' .. str .. '}')()
        ttr = tt
        tt = ttx
    else
        ttr = {}
    end
    return ttr
end

function table.copy(t)
    if t == nil then return nil end
    local nt = {}
    for k, v in pairs(t) do
        if type(v) == 'table' then
            nt[k] = table.copy(v)
        else
            nt[k] = v
        end
    end
    setmetatable(nt, table.copy(getmetatable(t)))
    return nt
end
