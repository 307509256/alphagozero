# -*- coding: utf-8 -*-
import numpy as np
from conf import conf

SIZE = conf['SIZE']
SWAP_INDEX = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]

def isplane(x,y,z):
    if x==0 or x==SIZE-1:
        return True
    if y==0 or y==SIZE-1:
        return True
    if z==0 or z==SIZE-1:
        return True
    return False
# x+y*9 = index    
def index2coord(index):
    z = index // (SIZE*SIZE)
    y = (index - z*SIZE*SIZE) // SIZE
    x = index - z*SIZE*SIZE - y*SIZE
    return x, y, z


def coord2index(x, y, z):
    return  z*SIZE*SIZE + y * SIZE + x


def legal_moves(board):
    # Occupied places
    mask1 = board[0,:,:,:,0].reshape(-1) != 0
    mask2 = board[0,:,:,:,1].reshape(-1) != 0
    mask = mask1 + mask2

    # Ko situations
    ko_mask = ((board[0,:,:,:,2] - board[0,:,:,:,0]))
    if (ko_mask == 1).sum() == 1:
        mask += (ko_mask == 1).reshape(-1)
    # Pass is always legal
    mask = np.append(mask, 0)
    return mask


def get_real_board(board):
    player = board[0,0,0,0,-1]
    if player == 1:
        real_board = board[0,:,:,:,0] - board[0,:,:,:,1]
    else:
        real_board = board[0,:,:,:,1] - board[0,:,:,:,0]
    return real_board


def _show_board(board, policy):
    real_board = get_real_board(board)
    if policy is not None:
        index = policy.argmax()
        x, y, z = index2coord(index)

    color = "B" if board[0][0][0][0][-1] == 1 else "W"
    string = "To play: %s\n" % color
    for k, row in enumerate(real_board):
        for j, col in enumerate(row):
            for i, c in enumerate(col):
                if c == 1:
                    string += u"○ "
                elif c == -1:
                    string += u"● "
                elif policy is not None and i == x and j == y and k == z:
                    string += u"X "
                else:
                    string += u". "
        string += "\n"
    if policy is not None and y == SIZE:
        string += "Pass policy"
    return string


def show_board(board, policy=None, history=1):
    results = []
    for i in reversed(range(history)):
        tmp_board = np.copy(board)
        tmp_board = tmp_board[:,:,:,:,i:]
        if i % 2 == 1:
            tmp_board[:,:,:,:,-1] *= -1
        results.append(_show_board(tmp_board, policy))
    return "\n".join(results)

dxdys = [(1, 0), (-1, 0), (0, 1), (0, -1)]
dxdydzs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

def capture_group(x, y, z, real_board, group=None):
    if group is None:
        group = [(x, y, z)]

    c = real_board[z][y][x]
    for dx, dy, dz in dxdydzs:
        nx = x + dx
        ny = y + dy
        nz = z + dz
        if (nx, ny, nz) in group:
            continue
        if not(0 <= nx < SIZE and 0 <= ny < SIZE and 0 <= nz < SIZE):
            continue
        dc = real_board[nz][ny][nx]
        if dc == 0:
            return None
        elif dc == c:
            group.append( (nx, ny, nz) )
            group = capture_group(nx, ny, nz, real_board, group=group)
            if group == None:
                return None
    return group


def take_stones(x, y, z, board):
    real_board = get_real_board(board)
    _player = 1 if board[0,0,0,0,-1] == 1 else -1
    for dx, dy, dz in dxdydzs:  # We need to check capture
        nx = x + dx
        ny = y + dy
        nz = z + dz
        if not(0 <= nx < SIZE and 0 <= ny < SIZE and 0 <= nz < SIZE):
            continue
        if real_board[nz][ny][nx] == 0:
            continue
        if real_board[nz][ny][nx] == _player:
            continue
        group = capture_group(nx, ny, nz, real_board)
        if group:
            for _x, _y, _z in group:
                assert board[0,_z,_y,_x,1] == 1
                board[0,_z,_y,_x,1] = 0
                real_board[_z][_y][_x] = 0
    for dx, dy, dz in dxdydzs + [(0, 0, 0)]:  # We need to check self sucide.
        nx = x + dx
        ny = y + dy
        nz = z + dz
        if not(0 <= nx < SIZE and 0 <= ny < SIZE and 0 <= nz < SIZE):
            continue
        if real_board[nz][ny][nx] == 0:
            continue
        if real_board[nz][ny][nx] != _player:
            continue
        group = capture_group(nx, ny, nz, real_board)
        if group:
            for _x, _y, _z in group:
                # Sucide
                assert board[0,_z,_y,_x,0] == 1
                board[0,_z,_y,_x,0] = 0
                real_board[_z][_y][_x] = 0

    return board


def make_play(x, y, z, board):
    player = board[0,0,0,0,-1]
    board[:,:,:,:,2:16] = board[:,:,:,:,0:14]
        
    if y != SIZE and isplane(x, y, z):
        # assert board[0,z,y,x,1] == 0
        # assert board[0,z,y,x,0] == 0
        board[0,z,y,x,0] = 1  # Careful here about indices
        board = take_stones(x, y, z, board)
    else:
        # "Skipping", player
        pass
    # swap_players
    board[:,:,:,:,range(16)] = board[:,:,:,:,SWAP_INDEX]
    player = -1 if player == 1 else 1
    board[:,:,:,:,-1] = player
    return board, player

# 遍历棋盘，递归找到黑白各占的数目
def _color_adjoint(i, j, k, color, board):
    # TOP
    SIZE1 = len(board)
    SIZE2 = len(board[0])
    SIZE3 = len(board[0][0])
    if i > 0 and board[i-1][j][k] == 0:
        board[i-1][j][k] = color
        _color_adjoint(i - 1, j, k, color, board)
    # BOTTOM
    if i < SIZE1 - 1 and board[i+1][j][k] == 0:
        board[i+1][j][k] = color
        _color_adjoint(i + 1, j, k,color, board)
    # LEFT
    if j > 0 and board[i][j - 1][k] == 0:
        board[i][j - 1][k] = color
        _color_adjoint(i, j - 1, k, color, board)
    # RIGHT
    if j < SIZE2 - 1 and board[i][j + 1][k] == 0:
        board[i][j + 1][k] = color
        _color_adjoint(i, j + 1,k, color, board)
    # out
    if k > 0 and board[i][j][k-1] == 0:
        board[i][j][k-1] = color
        _color_adjoint(i, j, k-1, color, board)
    # in
    if k < SIZE3 - 1 and board[i][j][k+1] == 0:
        board[i][j][k+1] = color
        _color_adjoint(i, j, k+1, color, board)    
    return board

def color_board(real_board, color):
    board = np.copy(real_board)
    for i, row in enumerate(board):
        for j, v in enumerate(row):
            for j, k in enumerate(v):
                if k == color:
                    _color_adjoint(i, j, k, color, board)
    return board

# 盘面的盈率  黑棋为正（1,2）， 白旗为负
def get_winner(board):
    real_board = get_real_board(board)
    points =  _get_points(real_board)
    black = points.get(1, 0) + points.get(2, 0)  # 取出 1和2出现的次数
    white = points.get(-1, 0) + points.get(-2, 0) + conf['KOMI']
    if black > white:
        return 1, black, white
    elif black == white:
        return 0, black, white
    else:
        return -1, black, white

# 计算在总数中出现的次数    
#  >>> a = numpy.array([0, 3, 0, 1, 0, 1, 2, 1, 0, 0, 0, 0, 1, 3, 4])
# >>> unique, counts = numpy.unique(a, return_counts=True)
# >>> dict(zip(unique, counts))
# {0: 7, 1: 4, 2: 1, 3: 2, 4: 1}
def _get_points(real_board):
    colored1 = color_board(real_board,  1)
    colored2 = color_board(real_board, -1)
    total = colored1 + colored2 # 一个数组
    unique, counts = np.unique(total, return_counts=True)
    points = dict(zip(unique, counts))
    return points


# [[[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]
#   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]] ...
def game_init():
    board = np.zeros((1, SIZE, SIZE, SIZE, 17), dtype=np.int32)
    player = 1
    board[:,:,:,:,-1] = player
    return board, player