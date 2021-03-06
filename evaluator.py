from self_play import play_game, self_play, save_game_data
from conf import conf
import os
from tqdm import tqdm
import datetime
from random import random

MCTS_SIMULATIONS = conf['MCTS_SIMULATIONS']
EVALUATE_N_GAMES = conf['EVALUATE_N_GAMES']
EVALUATE_MARGIN = conf['EVALUATE_MARGIN']

def elect_model_as_best_model(model):
    self_play(model, n_games=conf['N_GAMES'], mcts_simulations=conf['MCTS_SIMULATIONS'])
    full_filename = os.path.join(conf['MODEL_DIR'], conf['BEST_MODEL'])
    model.save(full_filename)

# PK的方式寻找最适合的模型
def evaluate(best_model, tested_model):
    total = 0
    wins = 0
    desc = "Evaluation %s vs %s" % (tested_model.name, best_model.name)
    tq = tqdm(range(EVALUATE_N_GAMES), desc=desc)
    for game in tq:
        start = datetime.datetime.now()

        if random() < .5:
            model1, model2 = best_model, tested_model
        else:
            model2, model1 = best_model, tested_model

        game_data = play_game(model1, model2, MCTS_SIMULATIONS, stop_exploration=0)
        stop = datetime.datetime.now()

        winner_model = game_data['winner_model']
        if winner_model == tested_model.name:
            wins += 1
        total += 1
        moves = len(game_data['moves'])
        print (wins, moves)
        if wins > 0  and moves > 0 :
            new_desc = desc + " (winrate:%s%% %.2fs/move)" % (int(wins/total*100), (stop - start).seconds / moves)
            tq.set_description(new_desc)
        
        
        save_game_data(best_model.name, game_data)


    if wins/total > EVALUATE_MARGIN:
        print("We found a new best model : %s!" % tested_model.name)
        elect_model_as_best_model(tested_model)
        return True
    return False
