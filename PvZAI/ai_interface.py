import random
import numpy as np
from collections import defaultdict
from PlantsVsZombies.source import constants as c
from PvZAI.PvZGeneticAlgorithm import PvZGeneticAlgorithm
from PvZAI.game_state import GameState
from PvZAI.ai_actions import CollectSunAction, DoNothingAction, PlaceAttackerAction, PlaceSunflowerAction, \
    PlaceDefenderAction, PlaceBomberAction
import pygame as pg
from PlantsVsZombies.source.state.screen import GameLoseScreen, GameVictoryScreen
import json
import pandas as pd

class PVZAI:
    def __init__(self, game, state_dict, level_name, randomized):
        self.game = game
        self.state_dict = state_dict
        self.level_name = level_name
        self.row_q_values = defaultdict(lambda: defaultdict(float))
        self.epsilon = 0.4
        self.alpha = 0.1
        self.states_per_action = int(35 * (20/c.SPEED_TIME))
        self.ga_evolve_freq = 10
        self.ga = PvZGeneticAlgorithm(
                population_size=30,
                generations=90,
                mutation_rate=0.2,
                crossover_rate=0.7,
                alpha=0.89,  # As per the paper
                beta=0.92,  # As per the paper
                theta=2.25  # As per the paper
                )
        self.episode_data = []
        self.q_value_changes = []
        self.ga_generation_scores = defaultdict(list)
        self.best_weights = [1/6, 1/6, 1/6, 1/6, 1/6 ,1/6]
        self.randomized = randomized

    def get_game_state(self):
        game_state = self.game.get_game_state()
        sun_count = game_state['sun_value']
        state = GameState(sun_count, rows= c.ROWS_NUMBER)

        for row in range(c.ROWS_NUMBER):
            row_state = state.rows[row]
            row_plants = [p for p in game_state['plants'].values() if p['position'][1] == row]
            row_zombies = [z for zlist in game_state['zombies'].values() for z in zlist if z['position'][1] == row]
            row_state.plants = len(row_plants)
            row_state.sunflower_count = sum(1 for p in row_plants if p['type'] == c.SUNFLOWER)
            row_state.zombie_count = len(row_zombies)
            row_state.attacker_count = sum(1 for p in row_plants if self.is_attacker(p['type']))
            row_state.defensive_count = sum(1 for p in row_plants if self.is_defender(p['type']))
            row_state.has_car = game_state['cars'][row]

            if row_zombies:
                closest_zombie = min(z['position'][0] for z in row_zombies)
                if closest_zombie < 3:
                    row_state.zombie_proximity = 3
                elif closest_zombie < 6:
                    row_state.zombie_proximity = 2
                else:
                    row_state.zombie_proximity = 1
            else:
                row_state.zombie_proximity = 0

        return state

    def get_available_actions(self, row):
        actions = []
        for plant_ind in range(8):
            plant_card = self.game.get_plant_card(plant_ind)
            if plant_card is not None:
                plant_type = plant_card[0]
                if self.can_place_plant(plant_type, row):
                    if self.is_attacker(plant_type):
                        actions.append(PlaceAttackerAction(plant_type))
                    elif plant_type == c.SUNFLOWER:
                        actions.append(PlaceSunflowerAction(plant_type))
                    elif self.is_defender(plant_type):
                        actions.append(PlaceDefenderAction(plant_type))
                    elif self.is_bomber(plant_type):
                        actions.append(PlaceBomberAction(plant_type))

        actions.append(DoNothingAction())
        return actions

    def select_action(self, state, mode=c.RLGA_AGENT):
        row_actions = []
        for row in range(c.ROWS_NUMBER):
            row_state = state.rows[row]
            available_actions = self.get_available_actions(row)
            row_state_key = self.row_state_to_key(row_state)

            if row_state_key not in self.row_q_values or mode not in [c.RLGA_AGENT, c.RL_AGENT]:
                row_actions.append((random.choice(available_actions), row))
            else:
                if random.random() < self.epsilon:
                    best_action = random.choice(available_actions)
                else:
                    best_action = max(available_actions, key=lambda a: self.row_q_values[row_state_key][
                        self.action_to_key(a)] if self.action_to_key(a) in self.row_q_values[row_state_key] else 0)
                row_actions.append((best_action, row))
        if mode not in [c.GA_AGENT, c.RLGA_AGENT]:
            row = self.calculate_closest_zombie()
        else:
            row = np.argmax(state.get_row_features().dot(np.array(self.best_weights).reshape(-1, 1)))
        return row_actions[row]

    def calculate_closest_zombie(self):
        game_state = self.game.get_game_state()
        zombies_location = []
        for row in range(c.ROWS_NUMBER):
            row_zombies = [z for zlist in game_state['zombies'].values() for z in zlist if z['position'][1] == row]
            if row_zombies:
                closest_zombie = min(z['position'][0] for z in row_zombies)
                zombies_location.append(closest_zombie)
            else:
                zombies_location.append(10)
        min_value = np.min(zombies_location)  # Find the minimum value in the array
        min_indices = np.where(zombies_location == min_value)[0]  # Find all indices of the minimum value
        return np.random.choice(min_indices)

    def row_state_to_key(self, row_state):
        return str(tuple(row_state.to_feature_vector()))

    def action_to_key(self, action):
        if isinstance(action, PlaceAttackerAction):
            return str(('place_attacker', action.attacker))
        elif isinstance(action, PlaceSunflowerAction):
            return str(('place_sunflower', action.sunflower))
        elif isinstance(action, PlaceDefenderAction):
            return str(('place_defender', action.defender))
        elif isinstance(action, PlaceBomberAction):
            return str(('place_bomber', action.bomber))
        elif isinstance(action, CollectSunAction):
            return str(('collect',))
        else:
            return 'do_nothing'

    def perform_action(self, action, row):
        success = False
        if isinstance(action, CollectSunAction):
            success = self.game.state.collect_sun(action.position)
        elif isinstance(action, DoNothingAction):
            for sun_position in self.game.get_game_state()['sun']:
                self.perform_action(CollectSunAction(sun_position), row)
            success = True
        elif self.get_plant_position(action, row) is None:
            return
        elif isinstance(action, PlaceAttackerAction):
            success = self.game.state.place_plant(action.attacker, self.get_plant_position(action, row),
                                                  self.get_card_cost_without_check(action.attacker))
        elif isinstance(action, PlaceSunflowerAction):
            success = self.game.state.place_plant(action.sunflower, self.get_plant_position(action, row),
                                                  self.get_card_cost_without_check(action.sunflower))
        elif isinstance(action, PlaceDefenderAction):
            success = self.game.state.place_plant(action.defender, self.get_plant_position(action, row),
                                                  self.get_card_cost_without_check(action.defender))
        elif isinstance(action, PlaceBomberAction):
            success = self.game.state.place_plant(action.bomber, self.get_plant_position(action, row),
                                                  self.get_card_cost_without_check(action.bomber))
        return success

    def calculate_reward(self, old_state, action, row):
        reward = 0

        if isinstance(action, DoNothingAction):
            reward += 20
            return reward
        old_row_state = old_state.rows[row]
        if isinstance(action, PlaceAttackerAction):
            reward += 25 if 3 > old_row_state.zombie_proximity > 0 else 0
            reward += 10 if old_row_state.zombie_proximity == 1 else 0
            reward += 15 if old_row_state.attacker_count < 1 else 0
            reward += 15 if old_row_state.defensive_count > 0 else 0
        elif isinstance(action, PlaceBomberAction):
            reward += 60 if old_row_state.zombie_proximity == 3 else 0
        elif isinstance(action, PlaceSunflowerAction):
            reward += 40 if old_row_state.zombie_proximity == 0 else 0
            reward -= (10 * old_row_state.sunflower_count)
        elif isinstance(action, PlaceDefenderAction):
            reward += 60 if old_row_state.zombie_proximity > 0 and old_row_state.attacker_count > 0 else 0
            reward -= 20 if old_row_state.defensive_count > 0 else 0

        return reward

    def train(self, num_episodes=200):
        def evaluate_fitness(weights):
            self.best_weights = weights
            total_reward = self.train_genetic_algorithm()
            self.ga_generation_scores[self.ga.current_generation].append(total_reward)
            return total_reward
        self.epsilon = 0.5
        perform_action = 0
        for episode in range(num_episodes):
            self.reset_game()
            total_reward = 0
            self.game.update()
            pg.display.update()
            actions = 0
            while not self.is_game_over():
                state = self.get_game_state()
                if not perform_action % self.states_per_action:
                    action, row = self.select_action(state)
                    total_reward += self.q_update(state, action, row)
                    actions += 1
                else:
                    action, row = DoNothingAction(), -1
                self.perform_action(action, row)
                self.game.update()
                pg.display.update()
                perform_action += 1
            self.epsilon *= 0.98
            print(f"Episode {episode + 1} completed.")
            self.episode_data.append({
                'episode_num': episode + 1,
                'total_reward': total_reward,
                'actions': actions
            })
        self.epsilon = 0
        # self.best_weights = self.ga.evolve(evaluate_fitness)
        #
        # print("Training completed.")
        # np.save('best_weights.npy', np.array(self.best_weights))

        # Save Q-values
        with open('row_q_values.json', 'w') as f:
            json.dump(self.row_q_values, f)
        self.save_training_data()
        print("Weights and Q-values saved.")

    def q_update(self, state, action, row):

        # Calculate the cumulative discounted reward
        row_reward = self.calculate_reward(state, action, row) if row != -1 else 0
        if isinstance(action, (PlaceDefenderAction, PlaceSunflowerAction, PlaceAttackerAction, PlaceBomberAction,
                               DoNothingAction)):
            row_state = state.rows[row]
            state_key = self.row_state_to_key(row_state)
            action_key = self.action_to_key(action)
            old_q = self.row_q_values[state_key][action_key]
            new_q = old_q + self.alpha * (row_reward - old_q)
            self.row_q_values[state_key][action_key] = new_q
            self.q_value_changes.append({
                'change_num': len(self.q_value_changes) + 1,
                'state': state_key,
                'action': action_key,
                'change': (old_q - new_q) ** 2
            })
        return row_reward

    def train_genetic_algorithm(self):
        self.reset_game()
        tick = 0
        self.game.update()
        pg.display.update()
        state = None
        while not self.is_game_over():
            state = self.get_game_state()
            action, row = self.select_action(state) if (tick % self.states_per_action) == 0 else (DoNothingAction(), -1)
            self.perform_action(action, row)
            self.game.update()
            pg.display.update()
            tick += 1

        total_reward = self.game_reward(state)

        print(f"Game over. Total reward: {total_reward}")
        return total_reward

    def play_game(self, mode):
        self.epsilon = 0
        self.load_saved_values()
        self.reset_game()
        self.game.state.initState()
        self.game.event_loop()
        self.game.update()
        pg.display.update()
        self.game.clock.tick(self.game.fps)
        tick = 0
        while not self.game.done:
            state = self.get_game_state()
            action, row = self.select_action(state, mode) if (tick % self.states_per_action) == 0 else (DoNothingAction(), -1)
            self.perform_action(action, row)
            self.game.update()
            pg.display.update()
            tick += 1
            if isinstance(self.game.state, GameVictoryScreen):
                if self.game.game_info[c.LEVEL_NUM] > c.LEVELS:
                    break
                self.reset_game()
            elif isinstance(self.game.state, GameLoseScreen):
                print(f'Game over, the agent lose level {self.game.state.game_info[c.LEVEL_NUM]}')
                return
        print('Game over, the agent win!')


    def reset_game(self):
        # Reset game to initial state
        self.game.setup_states(self.state_dict, self.game.state_name, self.level_name)
        self.game.state = self.state_dict[self.level_name]
        self.game.state.startup(self.game.current_time, self.game.game_info, self.randomized)
        self.game.state.initState()

        # Reinitialize Pygame if necessary
        if not pg.get_init():
            pg.init()

        # Reset display
        if self.game.screen is None:
            self.game.screen = pg.display.set_mode(c.SCREEN_SIZE)

        # Reset clock
        self.game.clock = pg.time.Clock()
        self.game.start_game = pg.time.get_ticks()


    def is_game_over(self):
        return (isinstance(self.game.state, GameLoseScreen) or isinstance(self.game.state, GameVictoryScreen)
                or self.game.is_over())

    def get_card_cost_without_check(self, plant_type):
        return self.game.state.get_card_cost_without_check(plant_type)

    def is_attacker(self, plant_type):
        return plant_type in [c.PEASHOOTER, c.SNOWPEASHOOTER]  # Add other attacker types if needed

    def is_defender(self, plant_type):
        return plant_type in [c.WALLNUT]  # Add other defender types if needed

    def is_bomber(self, plant_type):
        return plant_type in [c.CHERRYBOMB, c.POTATOMINE]

    def can_place_plant(self, plant_type, row):
        # Check if there's an available position in the row and if the plant card is not in cooldown
        return any(self.game.is_cell_empty(x, row,) for x in range(9))

    def get_plant_position(self, action, row):
        game_state = self.game.get_game_state()
        if isinstance(action, (PlaceSunflowerAction, PlaceAttackerAction)):
            for x in range(9):  # Find the leftmost empty position
                if self.game.is_cell_empty(x, row):
                    return x, row
        elif isinstance(action, (PlaceDefenderAction, PlaceBomberAction)):
            zombies = [z for zlist in game_state['zombies'].values() for z in zlist if z['position'][1] == row]
            if zombies:
                nearest_zombie_x = min(z['position'][0] for z in zombies)
                for x in range(min(nearest_zombie_x, 8), -1, -1):  # Find the nearest empty position to the zombie
                    if self.game.is_cell_empty(x, row):
                        return x, row
            else:
                for x in range(8, -1, -1):  # Find the rightmost empty position
                    if self.game.is_cell_empty(x, row):
                        return x, row
        return None  # Return None if no suitable position is found

    def game_reward(self, state):
        if state is None:
            return 0
        if isinstance(self.game.state, GameLoseScreen):
            return 0
        elif isinstance(self.game.state, GameVictoryScreen):
            cars = sum([s.has_car for s in state.rows])
            # time = (self.game.get_current_time() * c.SPEED_TIME) / 50
            return 5 + cars**1.5
        else:
             raise Exception("Game state should be lose or win in the end of the game.")

    def load_saved_values(self):
        try:
            self.best_weights = np.load('PvZAI/best_weights.npy')
            with open('PvZAI/row_q_values.json', 'r') as f:
                self.row_q_values = json.load(f)
            print("Weights and Q-values loaded successfully.")
        except FileNotFoundError:
            print("Saved files not found. Using initial values.")

    def save_training_data(self):
        # Save episode data
        episode_df = pd.DataFrame(self.episode_data)
        episode_df['average_reward'] = episode_df['total_reward'] / episode_df['actions']
        episode_df['SD'] = episode_df['average_reward'].rolling(window=10).std()
        episode_df[['episode_num', 'average_reward', 'SD']].to_csv('episode_data.csv', index=False)

        # Save Q-value changes
        pd.DataFrame(self.q_value_changes).to_csv('../q_value_changes.csv', index=False)

        # Save GA generation scores
        ga_scores = [{'generation_num': gen, 'average_score': np.mean(scores)}
                     for gen, scores in self.ga_generation_scores.items()]
        pd.DataFrame(ga_scores).to_csv('../ga_generation_scores.csv', index=False)
