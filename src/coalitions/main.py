import rooms
import agent as a
import matplotlib.pyplot as plot
import seaborn as sns
from tqdm import tqdm
import numpy as np
from copy import deepcopy

sns.set()


def episode(env, agents, nr_episode=0):
    state = env.reset()
    discounted_returns = [0] * len(agents)
    q_deltas = [0] * len(agents)
    discount_factor = 0.99
    done = False
    time_step = 0

    commits = [0] * len(agents)
    ensembles = 0

    while not done:
        state = deepcopy(state)

        # 1. Select action according to policy
        action = [agent.policy(state) for agent in agents]
        for i, a in enumerate(action):
            if a > 3:
                commits[i] += 1
        # 2. Execute selected action
        next_state, rewards, done, info = env.step(action)

        if env.ensemble:
            ensembles += 1

        # 3. Integrate new experience into agent
        for i, agent in enumerate(agents):
            q_deltas[i] += agent.update(state, action[i], rewards[i], next_state)

        state = next_state
        for i, agent in enumerate(agents):
            discounted_returns[i] += rewards[i] * \
                (discount_factor ** time_step)
        time_step += 1

    env.state_history.append(env.state(0))

    for agent in agents:
        agent.epsilon = max(agent.min_epsilon,
                            agent.epsilon - agent.epsilon_decay)

    commits = [c / time_step for c in commits]
    q_deltas = [q / time_step for q in q_deltas]

    return discounted_returns, env.ensemble, commits, q_deltas  # ensembles / time_step


training_episodes = 4000
discount_factor = 0.99
learning_rate = .2
init_epsilon = .1
epsilon_decay = 0
min_epsilon = .01

results = {}
for i, commit_mode in enumerate(rooms.COMMIT_MODES):
    n_agents = 2
    # instance = "./src/layouts/rooms_5_5_2.txt"
    instance = "./src/layouts/rooms_9_9_4.txt"
    env = rooms.load_env(n_agents, instance,
                         "videos/coalitions/rooms_ma_coalitions_" + str(commit_mode) + ".mp4", stochastic=False, commit_mode=commit_mode)
    nr_actions = env.action_space.n
    agents = []
    for _ in range(n_agents):
        agent = a.QLearningAgent(nr_actions, discount_factor,
                                 learning_rate=learning_rate, epsilon=init_epsilon, epsilon_decay=epsilon_decay, min_epsilon=min_epsilon)
        agents.append(agent)

    results = [episode(env, agents, i) for i in tqdm(range(training_episodes))]

    returns = []
    ensembles = []
    commits = []
    q_deltas = []
    for r, e, c, q in results:
        returns.append(r)
        ensembles.append(e)
        commits.append(c)
        q_deltas.append(q)

    plot.figure('results', figsize=(24, 18))

    plot.subplot(3, 4, i * 4 + 1)
    plot.title("reward distribution")
    for j in range(n_agents):
        plot.hist([r[j] for r in returns], alpha=.5, label='agent ' + str(j))
        plot.axvline(np.mean([r[j] for r in returns]), ls='--', label='mean ' + str(j))
    plot.xlabel("reward")
    plot.ylabel("count")
    plot.legend()

    N = 100

    plot.subplot(3, 4, i * 4 + 2)
    for j in range(n_agents):
        plot.plot(np.convolve([r[j] for r in returns],
                              np.ones((N,))/N, mode='valid'), label='agent ' + str(j))
    plot.title("progress")
    plot.xlabel("episode")
    plot.ylabel("discounted return")
    plot.ylim(0, 2)
    plot.legend()

    plot.subplot(3, 4, i * 4 + 3)
    for j in range(n_agents):
        plot.plot(np.convolve([c[j] for c in commits],
                              np.ones((N,))/N, mode='valid'), label='agent ' + str(j))
    plot.plot(np.convolve(ensembles, np.ones((N,))/N, mode='valid'), label='joint')
    plot.title("ensembles")
    plot.xlabel("episode")
    plot.ylabel("ensemble probability")
    plot.legend()

    plot.subplot(3, 4, i * 4 + 4)
    for j in range(n_agents):
        plot.plot(np.convolve([q[j] for q in q_deltas],
                              np.ones((N,))/N, mode='valid'), label='agent ' + str(j))
    plot.title("q deltas")
    plot.xlabel("episode")
    plot.ylabel("average q delta")
    plot.legend()

    plot.figure()
    env.save_video()

plot.figure('results')
plot.savefig('./img/coalitions/share_risk.png')
