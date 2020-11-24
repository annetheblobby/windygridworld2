from random import randint
from random import random
import seaborn
import matplotlib.pyplot as plt
from tqdm import tqdm


def gen_grid_states_with_wind(total_grid_size, max_number_of_moves, column_wind, total_number_of_rows,
                              total_number_of_columns, destination_row, destination_column, is_stochastic):
    """
    sets the environment for the WindyGridWorld problem
	:param total_grid_size: total number of squares in the grid
	:param max_number_of_moves: movement type (4 == Peasant Moves, 8 == King's)
	:param column_wind: the wind level for each column
	:param total_number_of_rows: # of rows in the grid
	:param total_number_of_columns: # of columns in the grid
	:param destination_row: destination row
	:param destination_column: destination column
	:param is_stochastic: nature of wind (boolean - True means stochastic)
	:return: Transition states from each cell and Rewards
	"""

    # set up the rewards and transition states grid
    sPrime = 1
    if (is_stochastic): sPrime = 3
    transition_states = [[[-1 for _ in range(sPrime)] for _ in range(max_number_of_moves)] for i in
                         range(total_grid_size)]
    Reward = [-1 for _ in range(total_grid_size)]
    Reward[destination_row * destination_column] = 0

    # possible actions
    for i in range(total_number_of_rows):
        for j in range(total_number_of_columns):
            for k in range(max_number_of_moves):
                if k == 0:  # Up
                    transition_states[i * total_number_of_columns + j][k][0] = (
                                                                                       i - 1) * total_number_of_columns + j if i - 1 >= 0 else i * total_number_of_columns + j
                if k == 1:  # Right
                    transition_states[i * total_number_of_columns + j][k][
                        0] = i * total_number_of_columns + j + 1 if j + 1 < total_number_of_columns else i * total_number_of_columns + j
                if k == 2:  # Down
                    transition_states[i * total_number_of_columns + j][k][0] = (
                                                                                       i + 1) * total_number_of_columns + j if (
                                                                                                                                       i + 1) < total_number_of_rows else i * total_number_of_columns + j
                if k == 3:  # Left
                    transition_states[i * total_number_of_columns + j][k][
                        0] = i * total_number_of_columns + j - 1 if j - 1 >= 0 else i * total_number_of_columns + j
                if k == 4:  # UpRight
                    transition_states[i * total_number_of_columns + j][k][0] = (
                                                                                       i - 1) * total_number_of_columns + j + 1 if i - 1 >= 0 and j + 1 < total_number_of_columns else i * total_number_of_columns + j
                if k == 5:  # RightDown
                    transition_states[i * total_number_of_columns + j][k][0] = (
                                                                                       i + 1) * total_number_of_columns + j + 1 if (
                                                                                                                                           i + 1) < total_number_of_rows and j + 1 < total_number_of_columns else i * total_number_of_columns + j
                if k == 6:  # DownLeft
                    transition_states[i * total_number_of_columns + j][k][0] = (
                                                                                       i + 1) * total_number_of_columns + j - 1 if (
                                                                                                                                           i + 1) < total_number_of_rows and j - 1 >= 0 else i * total_number_of_columns + j
                if k == 7:  # LeftUp
                    transition_states[i * total_number_of_columns + j][k][0] = (
                                                                                       i - 1) * total_number_of_columns + j - 1 if i - 1 >= 0 and j - 1 >= 0 else i * total_number_of_columns + j

                # actions taken in the case of stochastic wind
                if is_stochastic:
                    stoc_wind = randint(column_wind[j] - 1, column_wind[j] + 1)
                    if column_wind[j] != 0:
                        transition_states[i * total_number_of_columns + j][k][0] = \
                            transition_states[i * total_number_of_columns + j][k][
                                0] - stoc_wind * total_number_of_columns if \
                                transition_states[i * total_number_of_columns + j][k][
                                    0] - stoc_wind * total_number_of_columns >= 0 else \
                                transition_states[i * total_number_of_columns + j][k][0]
                else:
                    transition_states[i * total_number_of_columns + j][k][0] = \
                        transition_states[i * total_number_of_columns + j][k][0] - column_wind[
                            j] * total_number_of_columns if transition_states[i * total_number_of_columns + j][k][0] - \
                                                            column_wind[
                                                                j] * total_number_of_columns >= 0 else \
                            transition_states[i * total_number_of_columns + j][k][0]
    # transition table that says where to go if you take a certain action.
    return transition_states, Reward


# egreedy descision making
def epsilon_greedy(q_value, epsilon, transition, move_type):
    """
    the epsilon greedy policy
    :param q_value: Q state values to be passed in
    :param epsilon: epsilon to compare with for policy
    :param transition: transition states to use
    :param move_type: 4 for Peasants Move, 8 for Kings Move
    :return: where to move
    """
    return q_value.index(max(q_value)) if random() > epsilon / transition else randint(0, move_type - 1)


def sarsa(transition_states, rewards, where_to_start, destination, move_type, total_size_of_grid, alpha, gamma, epsilon, total_number_of_episodes, total_number_of_rows, total_number_of_columns):
    """
    Applies the SARSA algorithm
    :param transition_states: Transition states
    :param rewards: rewards
    :param where_to_start: Starting point
    :param destination: Goal
    :param move_type: Actions
    :param total_size_of_grid: State
    :param alpha: learning rate (set at 0.5)
    :param gamma: set at 1
    :param epsilon: set at 0.1
    :param total_number_of_episodes: the amount of episode loops
    :param total_number_of_rows: row
    :param total_number_of_columns: column
    :return: the amount of steps required in one episode (to reach the goal)
    """
    Q = [[0 for _ in range(move_type)] for _ in range(total_size_of_grid)]
    J, K = 0, 1
    # print("total_number_of_episodes", "Time-Steps", "Steps-In-One-Episode")
    # print(J, K - 1, 0)
    steps_in_one_episode = []
    states_visited = [[0 for _ in range(total_number_of_columns)] for _ in range(total_number_of_rows)]
    for _ in tqdm(range(total_number_of_episodes)):
        I = 0
        s = where_to_start
        a = epsilon_greedy(Q[s], epsilon, K, move_type)
        states_visited[3][0] += 1
        while s != destination:
            s1 = transition_states[s][a][0]
            r = rewards[s1]
            a1 = epsilon_greedy(Q[s1], epsilon, K, move_type)
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s1][a1] - Q[s][a])
            s, a = s1, a1
            J += 1
            I += 1
            states_visited[s // 10][s % 10] += 1
        K += 1
        steps_in_one_episode.append(I)
        # print(J, K - 1, I)
    graph(states_visited)
    optimal_policy(Q, where_to_start, destination)
    return steps_in_one_episode


def q_learning_algo(transition_states, reward, starting_point, destination, move_type, total_state_size, alpha, gamma, epsilon, total_number_of_episodes, total_rows, total_columns):
    """
    Applies the q_value learning algorithm
    :param transition_states: Transition states
    :param reward: rewards
    :param starting_point: Starting point
    :param destination: Goal
    :param move_type: Actions
    :param total_state_size: State
    :param alpha: learning rate (set at 0.5)
    :param gamma: set at 1
    :param epsilon: set at 0.1
    :param total_number_of_episodes: the amount of episode loops
    :param total_rows: row
    :param total_columns: column
    :return: the amount of steps required in one episode (to reach the goal)
    """
    Q = [[0 for _ in range(move_type)] for _ in range(total_state_size)]
    J, K = 0, 1

    # print("total_number_of_episodes", "Time-Steps", "Steps-In-One-Episode")
    # print(J, K - 1, 0)
    states_visited = [[0 for _ in range(total_columns)] for _ in range(total_rows)]
    steps_in_one_episode = []
    for _ in tqdm(range(total_number_of_episodes)):
        I = 0
        s = starting_point
        states_visited[s // 10][s % 10] += 1
        while s != destination:
            a = epsilon_greedy(Q[s], epsilon, K, move_type)
            s1 = transition_states[s][a][0]
            r = reward[s1]
            Q[s][a] = Q[s][a] + alpha * (r + gamma * max(Q[s1]) - Q[s][a])
            s = s1
            J += 1
            I += 1
            states_visited[s // 10][s % 10] += 1
        K += 1
        steps_in_one_episode.append(I)
        # print(J, K - 1, I)
    graph(states_visited)
    optimal_policy(Q, starting_point, destination)
    return steps_in_one_episode


def optimal_policy(transition, src, dest):
    """
    generate the optimal policy
    :param transition: rewards graph
    :param src: the source/ start
    :param dest: destination/ goal
    """
    policy = [["" for _ in range(10)] for _ in range(7)]
    for i in range(len(transition)):
        if i == src:
            max_action = "total_size_of_grid"
        elif i == dest:
            max_action = "G"
        else:
            max_val = -99999
            for j in range(4):
                if transition[i][j] >= max_val:
                    max_val = transition[i][j]
                    if j == 0:
                        max_action = "^"
                    elif j == 1:
                        max_action = ">"
                    elif j == 2:
                        max_action = "⌄"
                    else:
                        max_action = "<"
        policy[i // 10][i % 10] = max_action

    for i in range(len(policy)):
        print(([''.join(['{:8}'.format(str(item)) for item in policy[i]])]))
        print("\n")


# graphs the states
def graph(states):
    seaborn.heatmap(states)
    plt.show()


def plot_convergence(rewards, rewards1):
    """
    convergence graphs to compare the two algorithms
    :param rewards: SARSA rewards
    :param rewards1: Qlearning rewards
    """
    plt.plot(rewards, label="SARSA", color="#afeeee")
    plt.plot(rewards1, label="q_value-Learning", color="orange")
    plt.ylabel("Training total_number_of_episodes")
    plt.xlabel("Steps/Episode")
    plt.legend(loc="upper left")
    plt.show()


def main():
    rows, column = [7, 10]
    start_point_row, start_point_column, dest_point_row, dest_point_column = [3, 0, 3, 7]
    wind = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)  # column_wind
    move_type = 4  # Type of moves
    stochastic = False  # is_stochastic nature of column_wind
    alpha = 0.5
    gamma = 1
    epsilon = 0.1
    total_episodes = 100

    total_states = rows * column
    flattened_start_point = start_point_row * column + start_point_column
    destination_point = dest_point_row * column + dest_point_column

    transition, reward = gen_grid_states_with_wind(total_states, move_type, wind, rows, column,  dest_point_row, dest_point_column, stochastic)

    steps_per_episode_sarsa = sarsa(transition, reward, flattened_start_point, destination_point, move_type, total_states, alpha, gamma, epsilon, total_episodes, rows, column)
    print("--------------------------------------")
    steps_per_episode_q_learning = q_learning_algo(transition, reward, flattened_start_point, destination_point, move_type, total_states, alpha, gamma, epsilon, 100, rows, column)
    plot_convergence(steps_per_episode_sarsa, steps_per_episode_q_learning)

# todo  modify to make it more us, make policy graph, report
if __name__ == '__main__':
    main()
