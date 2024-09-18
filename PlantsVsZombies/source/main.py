__author__ = 'marble_xu'

from PvZAI.ai_interface import PVZAI
from . import tool
from . import constants as c
from .state import mainmenu, screen, level

def main(mode, speed, randomized=False):
    game = tool.Control(mode)
    state_dict = {c.MAIN_MENU: mainmenu.Menu(),
                  c.GAME_VICTORY: screen.GameVictoryScreen(),
                  c.GAME_LOSE: screen.GameLoseScreen(),
                  c.LEVEL: level.Level(mode)}
    if mode == c.MANUAL:
        define_time(speed)
        game.setup_states(state_dict, c.MAIN_MENU, c.LEVEL, randomized)
        game.main(randomized)
    elif mode == c.TRAIN:
        define_time(speed)
        ai = PVZAI(game, state_dict, c.LEVEL, randomized )
        ai.train()
    elif mode in [c.RL_AGENT, c.GA_AGENT, c.HEU_AGENT, c.RLGA_AGENT]:
        define_time(speed)
        ai = PVZAI(game, state_dict, c.LEVEL, randomized)
        ai.play_game(mode)
    else:
        raise Exception(f"The mode {mode} is invalid.")

def define_time(speed_time):
    c.SPEED_TIME = speed_time
    c.MOVEBAR_CARD_FRESH_TIME = int(6000 / speed_time)
    c.CARD_MOVE_TIME = int(60/speed_time)
    c.PRODUCE_SUN_INTERVAL = int(7000 / speed_time)
    c.FLOWER_SUN_INTERVAL = int(22000 / speed_time)
    c.SUN_LIVE_TIME = int(7000 / speed_time)
    c.ICE_SLOW_TIME = int(2000 / speed_time)
    c.FREEZE_TIME = int(7500 / speed_time)



