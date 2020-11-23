from random import randint
from random import random
import seaborn
import matplotlib.pyplot as plt
from tqdm import tqdm


def genEnvironment(S, A, Wind, R, C, SR, SC, DR, DC, Stochastic):
    """

	:param S: square number
	:param A: movement type
	:param Wind:
	:param R: row in grid
	:param C: column in grid
	:param SR: starting point row
	:param SC: starting point column
	:param DR:
	:param DC:
	:param Stochastic:
	:return:
	"""
    sPrime = 1
    if (Stochastic): sPrime = 3
    T = [[[-1 for _ in range(sPrime)] for _ in range(A)] for i in range(S)]
    Reward = [-1 for _ in range(S)]
    Reward[DR * DC] = 0

    for i in range(R):
        for j in range(C):
            for k in range(A):
                if k == 0:  # Up
                    T[i * C + j][k][0] = (i - 1) * C + j if i - 1 >= 0 else i * C + j
                if k == 1:  # Right
                    T[i * C + j][k][0] = i * C + j + 1 if j + 1 < C else i * C + j
                if k == 2:  # Down
                    T[i * C + j][k][0] = (i + 1) * C + j if (i + 1) < R else i * C + j
                if k == 3:  # Left
                    T[i * C + j][k][0] = i * C + j - 1 if j - 1 >= 0 else i * C + j
                if k == 4:  # UpRight
                    T[i * C + j][k][0] = (i - 1) * C + j + 1 if i - 1 >= 0 and j + 1 < C else i * C + j
                if k == 5:  # RightDown
                    T[i * C + j][k][0] = (i + 1) * C + j + 1 if (i + 1) < R and j + 1 < C else i * C + j
                if k == 6:  # DownLeft
                    T[i * C + j][k][0] = (i + 1) * C + j - 1 if (i + 1) < R and j - 1 >= 0 else i * C + j
                if k == 7:  # LeftUp
                    T[i * C + j][k][0] = (i - 1) * C + j - 1 if i - 1 >= 0 and j - 1 >= 0 else i * C + j

                if Stochastic:
                    stoc_wind = randint(Wind[j] - 1, Wind[j] + 1)
                    if Wind[j] != 0:
                        T[i * C + j][k][0] = T[i * C + j][k][0] - stoc_wind * C if T[i * C + j][k][
                                                                                       0] - stoc_wind * C >= 0 else \
                            T[i * C + j][k][0]
                else:
                    T[i * C + j][k][0] = T[i * C + j][k][0] - Wind[j] * C if T[i * C + j][k][0] - Wind[j] * C >= 0 else\
                        T[i * C + j][k][0]
    # transition table that says where to go if you take a certain action.
    # for i in range(len(T)):
    #     print(([''.join(['{:8}'.format(str(item)) for item in T[i]])]))
    # print("\n")
    return T, Reward


def eGreedy(Q, epsilon, t, A): return Q.index(max(Q)) if random() > epsilon/ t else randint(0, A - 1)


def SARSA(T, R, Source, Destination, A, S, alpha, gamma, epsilon, Stochastics, Episodes, Row, C):
    """

    :param T:
    :param R:
    :param Source:
    :param Destination:
    :param A:
    :param S:
    :param alpha:
    :param gamma:
    :param epsilon:
    :param Stochastics:
    :param Episodes:
    :param Row:
    :param C:
    :return:
    """
    Q = [[0 for _ in range(A)] for _ in range(S)]
    J, K = 0, 1
    # print("Episodes", "Time-Steps", "Steps-In-One-Episode")
    # print(J, K - 1, 0)
    steps_in_one_episode = []
    states_visited = [[0 for _ in range(C)] for _ in range(Row)]
    for _ in tqdm(range(Episodes)):
        I = 0
        s = Source
        a = eGreedy(Q[s], epsilon, K, A)
        states_visited[3][0] += 1
        while s != Destination:
            s1 = T[s][a][0]  # todo make this greedy
            r = R[s1]
            a1 = eGreedy(Q[s1], epsilon, K, A)
            Q[s][a] = Q[s][a] + alpha * (r + gamma * Q[s1][a1] - Q[s][a])
            s, a = s1, a1
            J += 1
            I += 1
            states_visited[s // 10][s % 10] += 1
        K += 1
        steps_in_one_episode.append(I)
        # print(J, K - 1, I)
    graph(states_visited)
    optimal_policy(Q, Source, Destination)
    return steps_in_one_episode


def Q_Learning(T, R, Source, Destination, A, S, alpha, gamma, epsilon, Stochastics, Episodes, Row, C):
    Q = [[0 for _ in range(A)] for _ in range(S)]
    J, K = 0, 1

    # print("Episodes", "Time-Steps", "Steps-In-One-Episode")
    # print(J, K - 1, 0)
    states_visited = [[0 for _ in range(C)] for _ in range(Row)]
    steps_in_one_episode = []
    for _ in tqdm(range(Episodes)):
        I = 0
        s = Source
        states_visited[s // 10][s % 10] += 1
        while s != Destination:
            a = eGreedy(Q[s], epsilon, K, A)
            s1 = T[s][a][0]
            r = R[s1]
            Q[s][a] = Q[s][a] + alpha * (r + gamma * max(Q[s1]) - Q[s][a])
            s = s1
            J += 1
            I += 1
            states_visited[s // 10][s % 10] += 1
        K += 1
        steps_in_one_episode.append(I)
        # print(J, K - 1, I)
    graph(states_visited)
    optimal_policy(Q, Source, Destination)
    return steps_in_one_episode


def optimal_policy(transition, src, dest):
    policy = [["" for _ in range(C)] for _ in range(R)]
    for i in range(len(transition)):
        if i == src:
            max_action = "S"
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


def graph(states):
    seaborn.heatmap(states)
    plt.show()


def plot_convergence(rewards, rewards1):
    plt.plot(rewards, label="SARSA", color="#afeeee")
    plt.plot(rewards1, label="Q-Learning", color="orange")
    plt.ylabel("Training Episodes")
    plt.xlabel("Steps/Episode")
    plt.legend(loc="upper left")

    plt.show()


R, C = [7, 10]
SR, SC, DR, DC = [3, 0, 3, 7]  # Source and Destination
Wind = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)  # Wind
A = 4  # Type of moves
Stochastic = False  # Stochastic nature of Wind
alpha = 0.5
gamma = 1
epsilon = 0.1
Episodes = 100

S = R * C
Source = SR * C + SC
Destination = DR * C + DC

T, Reward = genEnvironment(S, A, Wind, R, C, SR, SC, DR, DC, Stochastic)

Q = SARSA(T, Reward, Source, Destination, A, S, alpha, gamma, epsilon, Stochastic, Episodes, R, C)
print("--------------------------------------")
Q2 = Q_Learning(T, Reward, Source, Destination, A, S, alpha, gamma, epsilon, Stochastic, 100, R, C)
plot_convergence(Q, Q2)

# todo, comment, remove excess variables, change variable nomnoms
#  modify to make it more us, make policy grraph, report
