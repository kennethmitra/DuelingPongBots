import torch

class Buffer:
    def __init__(self):

        # Stores one entry per time step
        self.tstep = []
        self.obs = []
        self.act = []
        self.logp = []
        self.val = []
        self.rew = []
        self.entropy = []
        self.disc_rtg_rews = []

        # One entry per episode
        self.per_episode_rews = []
        self.per_episode_disc_rews = []
        self.per_episode_length = []

    def record(self, timestep=None, obs=None, act=None, logp=None, val=None, rew=None, entropy=None):
        
        if timestep is not None: self.tstep.append(timestep)
        if obs is not None: self.obs.append(obs)
        if act is not None: self.act.append(act)
        if logp is not None: self.logp.append(logp)
        if val is not None: self.val.append(val)
        if rew is not None: self.rew.append(rew)
        if entropy is not None: self.entropy.append(entropy)

    def store_episode_stats(self, episode_rews, episode_disc_rtg_rews):

        self.per_episode_rews.append(torch.tensor(episode_rews, requires_grad=False).sum().item())
        self.disc_rtg_rews.extend(episode_disc_rtg_rews)
        self.per_episode_disc_rews.append(torch.tensor(episode_disc_rtg_rews, requires_grad=False).sum().item())
        self.per_episode_length.append(len(self.tstep))

        # When ending an episode, make sure all lists have same length
        assert len(self.tstep) == len(self.obs) == len(self.act) == len(self.logp) == len(self.val) == len(self.rew) == len(self.entropy) == len(self.disc_rtg_rews)
        assert len(self.per_episode_length) == len(self.per_episode_rews)

        
    def get(self):
        data = dict(tstep=self.tstep, obs=self.obs, act=self.act, logp=self.logp, val=self.val, rew=self.rew,
                    entropy=self.entropy, disc_rtg_rews=self.disc_rtg_rews, per_episode_rews=self.per_episode_rews,
                    per_episode_length=self.per_episode_length, per_episode_disc_rews=self.per_episode_disc_rews)
        return data

    def clear(self):
        del self.tstep[:]
        self.tstep.clear()
        del self.obs[:]
        self.obs.clear()
        del self.act[:]
        self.act.clear()
        del self.logp[:]
        self.logp.clear()
        del self.val[:]
        self.val.clear()
        del self.rew[:]
        self.rew.clear()
        del self.entropy[:]
        self.entropy.clear()
        del self.disc_rtg_rews[:]
        self.disc_rtg_rews.clear()
        del self.per_episode_rews[:]
        self.per_episode_rews.clear()
        del self.per_episode_length[:]
        self.per_episode_length.clear()
        del self.per_episode_disc_rews[:]
        self.per_episode_disc_rews.clear()


# Buffer Unit Tests
if __name__ == '__main__':
    buf = Buffer()
    buf.record(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
    buf.record(1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6)
    data = buf.get()
    print(data)
    print(data['obs'])
    buf.clear()
    data = buf.get()
    print(data)