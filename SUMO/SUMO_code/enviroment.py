import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from .veri_cekme import (traci, pending, set_agent_action, update_pending_transitions,
                         get_agent_state, build_messages, get_tls_action_spaces,
                         get_tls_neighbors, load_net, NUM_SLOTS)

class TrafficEnv(ParallelEnv):
    metadata = {"name": "sumo_traffic_v0.0"}

    def __init__(self, cfg, decision_interval=20, yellow=3, max_time=3600,
                max_lanes=8, queue_cap=20.0, seed=0):
        self.cfg = cfg
        self.interval = decision_interval
        self.yellow = yellow
        self.max_time = max_time
        self.max_lanes = max_lanes
        self.queue_cap = queue_cap
        self.seed = seed

        #Ağ yapısını bir kez çek
        self.action_spaces_info = get_tls_action_spaces(cfg)
        self.possible_agents = list(self.action_spaces_info.keys())
        self.agents = []

        net = load_net(cfg)
        self.neigbors = get_tls_neighbors(self.possible_agents, net)

        self.max_actions = max(s["action_space_size"] for s in self.action_spaces_info.values())
        self.obs_dim = (self.max_lanes * 3 #şerit başına  3 tane
                        + self.max_actions #aktif faz one-hot
                        + NUM_SLOTS #Komşu mesajlar
                        + 1 #Sarıda mıyım?
                        )


    #PettingZoo arayüzü
    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, shape=(self.obs_dim, ), dtype=np.float32)

    def action_space(self, agent):
        return spaces.Discrete(self.max_actions)

    def reset(self, seed = None, options = None):
        if self.seed is not None:
            self.seed += 1 #Her episode da farklı trafik için
        traci.load(["-c", self.cfg, "--no-step-log", "true",
                    "--seed", str(self.seed)])

        pending.clear()
        self.agents = list(self.possible_agents)

        #lane listesini sabitle hep aynı kalsın
        self.lanes = {t: list(dict.fromkeys(traci.trafficlight.getControlledLanes(t)))
                      for t in self.agents}
        self.cur_action = {t: 0 for t in self.agents}


        #Başlangıç fazını ajan aksiyonlarından birisi olarak ayarla
        for t in self.agent:
            traci.trafficlight.setRedYellowGreenState(
                t, self.action_spaces_info[t]["available_actions"][0]["state"])

        traci.simulationStep()
        return self._observe(), {t: {} for t in self.agents}

    
    def step(self, actions):
        #Aksiyonları uygula
        for t, a in actions.items():
            n = self.action_spaces_info[t]["action_space_size"]
            a = int(a) if  int(a) < n else self.cur_action[t]
            set_agent_action(t,a,self.action_spaces_info)
            self.cur_action[t] = a

        #Karar aralığı kadar ilerlet, ödülü aralık boyunca topla
        reward_acc = {t: 0.0 for t in self.agents}

        for _ in range(self.interval):
            traci.simulationStep()
            update_pending_transitions()

            for t in self.agents:
                reward_acc[t] += self._step_reward

        obs = self._observe()
        rewards = {t: reward_acc[t] / self.interval for t in self.agents}

        done = traci.simulation.getTime() >= self.max_time
        terminations = {t: False for t in self.agents}
        truncations = {t: done for t in self.agents}
        infos = {t: {"action_mask": self._mask(t)} for t in self.agents}

        if done:
            self.agents = []
        return obs, rewards, terminations, truncations, infos
    
    
    #----Yardımcılar-----
    def _step_reward(self, t):
        halting = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.lanes[t])
        return -halting / self.queue_cap

    def _mask(self, t):
        mask = np.zeros(self.max_actions, dtype = np.int8)
        mask[: self.action_spaces_info[t]["action_space_size"]] = 1
        return mask

    def _observe(self):
        raw = {t: get_agent_state(t) for t in self.agents}
        msgs = build_messages(raw, self.neigbors)
        obs = {}

        for t in self.agents:
            lane_feats = []
            for l in self.lanes[t][:self.max_lanes]:
                d = raw[t]["lane_details"][l]
                speed = d["mean_speed"] if d["vehicle_count"] > 0 else 0.0
                lane_feats += [min(d["halting_count"] / self.queue_cap, 1.0),
                               min(d["vehicle_count"] / self.queue_cap, 1.0),
                               speed / 13.89]
            lane_feats += [0.0] * (self.max_lanes * 3 - len(lane_feats)) #padding

            phase = np.zeros(self.max_actions); phase[self.cur_action[t]] = 1.0
            nb = [min(m / self.queue_cap, 1.0) for m in msgs[t]]

            in_yellow = [1.0 if t in pending else 0.0]

            obs[t] = np.array(lane_feats + list(phase) + nb + in_yellow, dtype=np.float32)
        return obs




        return super().step(actions)

