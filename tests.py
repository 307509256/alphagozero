# -*- coding: utf-8 -*-
from conf import conf
conf['SIZE'] = 9  # Override settings for tests
conf['KOMI'] = 5.5  # Override settings for tests

import unittest
import numpy as np
import random
import os
from play import (
        color_board, _get_points, capture_group, make_play,legal_moves,
        index2coord, game_init
)

from play import (
    show_board
)

from self_play import (
        play_game
)
from engine import simulate, ModelEngine
from symmetry import (
        _id,
        left_diagonal, reverse_left_diagonal,
        right_diagonal, reverse_right_diagonal,
        vertical_axis, reverse_vertical_axis,
        horizontal_axis, reverse_horizontal_axis,
        rotation_90, reverse_rotation_90,
        rotation_180, reverse_rotation_180,
        rotation_270, reverse_rotation_270,
)
import itertools
from sgfsave import save_game_sgf
from gtp import Engine


class TestBoardMethods(unittest.TestCase):
    def test_self_sucide2(self):
        print ("test_self_sucide2\n")
        board, player = game_init()
        make_play(0, 0, 1, board) # black
        make_play(1, 0, 8, board) # white
        print (show_board(board))
        self.assertEqual(board[0][1][0][0][0], 1) # white stone
        self.assertEqual(board[0][0][1][1][1], 0) # was not taken
        self.assertEqual(isplane(1,2,8), 1)

if __name__ == '__main__':
    unittest.main()

