from time_handler import deadline, TimedOutExc
import numpy as np
import os
import goSim as goSim


class SingleMatch():
    def __init__(self, board_size, komi, match_folder):
        self.player_color = 'black'
        self.board_size = board_size
        self.komi = komi
        self.seed = np.random.rand()
        self.match_folder = match_folder
        self.opponent_action = -1

        self.env = goSim.GoEnv(player_color=self.player_color, observation_type='image3c', illegal_move_mode="raise", board_size=self.board_size, komi=self.komi)

        # init board
        self.obs_t = self.env.reset()

        # init players
        # Todo Select which one is the first player
        self.p1 = Player_1.AlphaGoPlayer(self.obs_t.copy(), self.seed, 1)
        self.p2 = Player_2.AlphaGoPlayer(self.obs_t.copy(), self.seed, 2)
        if not os.path.exists(self.match_folder):
            os.makedirs(self.match_folder)

    @deadline(5)
    def get_action(self, p):
        return p.get_action(self.obs_t.copy(), self.opponent_action)

    def run_match(self):
        done = False

        i = 0
        history = []
        while True:
            # Get player
            player_color = i % 2 + 1
            if player_color == 1:
                player = self.p1
            else:
                player = self.p2
            # print("!!!!!!!!!!!!")
            # print(self.env.state.color)
            # print(self.env.player_color)
            self.env.set_player_color(player_color)
            # print(self.env.state.color)
            # print(self.env.player_color)
            # print("!!!!!!!!!!!!")

            # Get Player action
            # Todo Check for out of memory and other errors
            try:
                a_t = self.get_action(player)
                # print(self.env.state.color)

            except TimedOutExc as e:
                print("took too long")
                a_t = goSim._pass_action(self.board_size)

            # Take action
            # print(self.env.state.color)
            # print(self.env.player_color)
            self.obs_t, a_t, r_t, done, info, cur_score = self.env.step(a_t)
            # print(self.env.state.color)
            # print("yyyyyyyyyyyyyy$$$$$$$$$$$$$$")
            self.env.render()
            self.opponent_action = a_t
            history.append(str(player_color) + ': ' + str(a_t))
            if done:
                if cur_score > 0:
                    # White i.e. player 2 wins
                    winner = "P2"
                elif cur_score < 0:
                    # Black i.e. player 1 wins
                    winner = "P1"
                else:
                    # Draw
                    winner = "DRAW"

                with open(self.match_folder + '/actions.csv', 'w') as fw:
                    for entry in history:
                        fw.write(entry + '\n')
                fw.close()


                return winner, cur_score

            # Book Keeping
            i += 1
        self.env.close()
