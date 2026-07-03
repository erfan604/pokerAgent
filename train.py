from rlcard.agents import DQNAgent
from rlcard.utils import tournament, Logger, reorganize, plot_curve

from match import make_env, make_opponent


def train(episodes=1000, opponent='random'):
    env = make_env()
    poker_agent = DQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], mlp_layers=[64, 64])
    env.set_agents([poker_agent, make_opponent(env, opponent)])

    with Logger("experiments/nlhe_dqn") as logger:
        for episode in range(episodes):
            trajectories = [[] for _ in range(env.num_players)]
            state, player = env.reset()
            trajectories[player].append(state)

            while not env.is_over():
                action = env.agents[player].step(state)
                trajectories[player].append(action)
                state, player = env.step(action, env.agents[player].use_raw)
                if not env.is_over():
                    trajectories[player].append(state)

            for pid in range(env.num_players):
                trajectories[pid].append(env.get_state(pid))

            payoffs = env.get_payoffs()
            trajectories = reorganize(trajectories, payoffs)

            for transition in trajectories[0]:
                poker_agent.feed(transition)

            if episode % 50 == 0:
                logger.log_performance(env.timestep, tournament(env, 1000)[0])

        csv_path, fig_path = logger.csv_path, logger.fig_path
        plot_curve(csv_path, fig_path, "poker_agent")
    return poker_agent
