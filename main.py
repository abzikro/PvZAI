import pygame as pg
import sys
from PlantsVsZombies.source.main import main

if __name__ == '__main__':
    args = sys.argv
    if len(args) not in [3, 4]:
        print("ERROR: The program should get 2 or 3 arguments.")
    elif args[1] not in ["manual", "train", "rl", "ga", "heu", "rl-ga"]:
        print("ERROR: The first argument should be in [manual, train, rl, ga, random, rl-ga]")
    elif not args[2].isnumeric() or int(args[2]) not in range(1, 21):
        print("ERROR: The second argument should be number between 1 to 20.")
    elif len(args) == 4 and args[3] != "random":
        print("ERROR: The third argument can be only 'random'.")
    else:
        mode = args[1]
        speed = int(args[2])
        randomized = True if len(args) == 4 and args[3] == "random" else False
        main(mode, speed, randomized)
        pg.quit()
