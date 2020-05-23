import os
import numpy as np
import importlib
import sys
import time

class Tournament():
    def __init__(self, student_list, num_matches, board_size, komi):
        self.student_list = student_list
        self.num_matches = num_matches
        self.board_size = board_size
        self.komi = komi
        self.folder_name = 'Tournament'
        self.module_folder = 'modules'

        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        if not os.path.exists(self.module_folder):
            os.makedirs(self.module_folder)
    
    def run_tournament(self):
        n = len(self.student_list)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.student_list[i]
                p2 = self.student_list[j]

                root_folder = self.folder_name + '/' + str(p1) + '_' + str(p2)
                head_to_head = RunMatches(p1, p2, self.num_matches, root_folder, self.board_size, self.komi)
                head_to_head.run_matches()



class RunMatches():
    def __init__(self, p1, p2, num_matches, root_folder, board_size, komi):
        self.player1 = p1
        self.player2 = p2
        self.num_matches = num_matches
        self.root_folder = root_folder
        self.board_size = board_size
        self.komi = komi

        if not os.path.exists(self.root_folder):
            os.makedirs(self.root_folder)
    
    def run_matches(self):
        for match_num in range(self.num_matches):
            first_player = None
            second_player = None
            if match_num % 2 == 0:
                first_player = self.player1
                second_player = self.player2
            else:
                first_player = self.player2
                second_player = self.player1
            match_folder = self.root_folder + '/match' + str(match_num + 1)
            with open('modules/tmp_match_' + str(self.player1) + '_' + str(self.player2) + '_' + str(match_num) + '.py', 'w') as fw:
                fw.write('import AlphaGoPlayer_' + str(first_player) + ' as Player_1\n')
                fw.write('import AlphaGoPlayer_' + str(second_player) + ' as Player_2\n')
                lines = None
                with open('single_match.py', 'r') as fr:
                    lines = fr.readlines()
                fr.close()
                for line in lines:
                    fw.write(line)
            fw.close()
            time.sleep(1)
            tmp_match = importlib.import_module('modules.tmp_match_' + str(self.player1) + '_' + str(self.player2) + '_' + str(match_num))
            match = tmp_match.SingleMatch(self.board_size, self.komi, match_folder)
            winner, final_score = match.run_match()
            print("WINNNER = ", winner)


t = Tournament([6,1], 4, 13, 7.5)

t.run_tournament()