from guidance import models, select
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer

# Import the visualization handler
from pypokerengine.ui.graphics_visualizer import GraphicVisualizer

class GuidancePokerPlayer(BasePokerPlayer):
    def __init__(self, lm, name):
        self.lm = lm
        self.name = name
        self.poker_agent = self.lm + f"""# Instructions

You are playing a game of Texas Hold'em poker against other opponents. Pick the best action to win the game.

# Game State
"""

    def declare_action(self, valid_actions, hole_card, round_state):
        self.poker_agent += f"""## Round {round_state['round_count']}

Your hole cards: {hole_card}
Community cards: {round_state['community_card']}
Pot size: {round_state['pot']['main']['amount']}
Your stack: {self.stack}
Opponents' stacks: {[p['stack'] for p in round_state['seats'] if p['uuid'] != self.uuid]}
Available actions: {valid_actions}

# Opponents' past actions
"""
        for seat in round_state['seats']:
            if seat['uuid'] != self.uuid:
                self.poker_agent += f"{seat['name']}: {seat['action_histories']}\n"

        self.poker_agent += f"""
What action do you want to take?

Your choice: {select([a['action'] for a in valid_actions], name="choice")}
"""

        action = self.poker_agent["choice"]
        amount = valid_actions[0]['amount'] if action == 'fold' else valid_actions[1]['amount']
        return action, amount

    def receive_game_start_message(self, game_info):
        self.uuid = game_info['seats'][game_info['player_num']]['uuid']
        self.stack = game_info['seats'][game_info['player_num']]['stack']

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

lm = models.LlamaCpp("/Users/nicholasking/code/models/mistral-7b-v0.1.Q8_0.gguf", n_gpu_layers=-1, n_ctx=4096)

config = setup_config(max_round=10, initial_stack=100, small_blind_amount=5)
config.register_player(name="Mistral1", algorithm=GuidancePokerPlayer(lm, "Mistral1"))
config.register_player(name="Mistral2", algorithm=GuidancePokerPlayer(lm, "Mistral2"))
config.register_player(name="Mistral3", algorithm=GuidancePokerPlayer(lm, "Mistral3"))

# Create a GraphicVisualizer instance
visualizer = GraphicVisualizer()

# Start the game with the visualizer
game_result = start_poker(config, verbose=1, visualizer=visualizer)
