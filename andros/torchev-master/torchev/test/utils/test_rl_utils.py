import numpy as np
from torchev.utils.rl_util import discount_reward, TimeBaselineFunction

def test_discount_reward() :
    rewards = [0.7, 0.3, -1.3, 2.3]
    gamma = 0.9
    discounted_rewards = discount_reward(rewards, gamma)
    manual_result = []
    for ii in range(len(rewards)) :
        tmp = rewards[ii]
        for jj in range(ii+1, len(rewards)) :
            tmp += gamma**(jj-ii) * rewards[jj]
        manual_result.append(tmp)

    assert np.allclose(discounted_rewards, manual_result)
    pass

def test_time_baseline_func() :
    rewards = np.random.rand(1000)
    timefun = TimeBaselineFunction(window=100)
    for rr in rewards :
        timefun.append(0, rr)
    assert np.allclose(np.mean(rewards[-100:]), timefun.mean_history(0)) 
    assert np.allclose(np.std(rewards[-100:]), timefun.std_history(0)) 
    pass
