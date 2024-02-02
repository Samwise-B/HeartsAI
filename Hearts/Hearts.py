import gym
import random
import numpy as np
import math

maxScore = 100

cards = {
    0: "2s", 1: "3s", 2: "4s", 3: "5s", 4: "6s", 5: "7s", 6: "8s", 7: "9s", 8: "10s", 9: "Js", 10: "Qs", 11: "Ks", 12: "As",
    13: "2c", 14: "3c", 15: "4c", 16: "5c", 17: "6c", 18: "7c", 19: "8c", 20: "9c", 21: "10c", 22: "Jc", 23: "Qc", 24: "Kc", 25: "Ac",
    26: "2d", 27: "3d", 28:"4d", 29: "5d", 30: "6d", 31: "7d", 32:"8d", 33:"9d", 34: "10d", 35:"Jd", 36:"Qd", 37:"Kd", 38: "Ad", 
    39: "2h", 40: "3h", 41: "4h", 42: "5h", 43: "6h", 44: "7h", 45:"8h", 46: "9h", 47:"10h", 48:"Jh", 49:"Qh", 50:"Kh", 51: "Ah"
}

class HeartsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, verbose = False, manual = False):
        self.name = "hearts"
        self.maxScore = maxScore

        self.n_players = 4
        self.current_player_num = 0
        self.players = [Player(0), Player(1), Player(2), Player(3)]

        # initialise observation space
        self.observation_space = gym.spaces.Dict(
            {
                "current_trick": gym.spaces.Discrete(4),
                "player_cards": gym.spaces.Discrete(13),
                "player_scores": gym.spaces.Discrete(4),
                "player_position": gym.spaces.Discrete(1),
                "remaining_cards": gym.spaces.Discrete(52)
            }
        )

        # initialise action space
        self.action_space = gym.spaces.Discrete(13)

        self.remaining_cards = [x for x in range(52)]

        self.current_trick = [-1 for i in range(4)]
        self.current_trick_suit = None
        self.trick_start_pos = 0

        self.terminated = False

    def _get_obs(self):
        # get current trick values
        trick = np.array(self.current_trick)

        # get player's hand
        player_id = self.current_player_num
        player_cards = np.full(13, -1)
        for i, card in enumerate(self.players[player_id].hand): # fix this
            player_cards[i] = card

        # get all players scores
        player_scores = np.empty(4)
        for player in self.players:
            player_scores[player.id] = player.score

        # get player's position
        player_position = (player_id - self.trick_start_pos) % self.n_players
        
        # get remaining cards
        individuals_remaining_cards = []
        for card in self.remaining_cards:
            if card not in player_cards:
                individuals_remaining_cards.append(card)

        return {
            "current_trick": trick,
            "player_cards": player_cards,
            "player_scores": player_scores,
            "player_position": player_position,
            "remaining_cards": individuals_remaining_cards
        }

    def _get_info(self):
        return {}
    
    def legal_actions(self):
        legal_actions = np.zeros(13)
        current_player = self.players[self.current_player_num]
        can_follow_suit = False
        for i, card in enumerate(current_player.hand):
            card_num, card_suit = format_card(card)
            if card_suit == self.current_trick_suit:
                legal_actions[i] = 1
                can_follow_suit = True
        if not can_follow_suit:
            legal_actions = np.ones(13)

        return legal_actions

            

    def reset(self, seed=None):
        # following line to seed self.np_random
        super().reset(seed=seed)

        # reset and shuffle the deck
        self.deck = Deck()
        self.deck.shuffle()

        # reset remaining card tracker
        self.remaining_cards = [x for x in range(52)]
        
        for player in self.players:
            # clear player hand and score
            player.reset()

            # deal player new hand
            player.hand = self.deck.draw(13)

            # if player has the 2 of clubs
            if 13 in player.hand:
                # set current player to the left of player with 2 of clubs
                self.current_player_num = (player.id + 1) % 4
                # add 2 of clubs to current trick and discard from player's hand
                self.current_trick[player.id] = 13
                player.discard(13)

                self.current_trick_suit = "c"

        # return observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):
        terminated = False
        reward = [0] * self.n_players

        if self.legal_actions[action] == 0:
            # handle illegal actions
            pass
        else:
            # add card to trick
            self.current_trick.append(action)

            # remove card from player's hand
            player_id = self.current_player_num
            self.players[player_id].discard(action)
            # remove card from remaining cards list
            self.remaining_cards.remove(action)

            # handle end of trick
            if (self.trick_start_pos - 1) % self.n_players == self.current_player_num:
                # check who won the trick and update their score
                trick_score = 0
                max_card = 0
                winner = None
                for player_id, card in enumerate(self.current_trick):
                    card_num, card_suit = format_card(card)

                    # find the winner of the trick
                    if card_suit == self.current_trick_suit and card_num > max_card:
                        max_card = card_num
                        winner = player_id
                    
                    # handle trick score
                    if card_suit == "h":
                        trick_score += 1
                    elif card_suit == "s" and card_num == 10:
                        trick_score += 13

                # update score and current player
                self.players[winner].score += trick_score
                reward[winner] = trick_score
                self.current_player_num = winner
                self.trick_start_pos = winner

                # handle end of game
                if self.players[winner].score >= 100:
                    self.terminated = True

        self.render()
            
        return self.observation, reward, self.terminated, False, {}


    def render(self):

        if not self.terminated:
            print(f"Player {self.current_player_num}'s Turn")

            print("---- Player Scores ----")
            for player in self.players:
                print(f"Player {player.id}:{player.score}")

            print("---- Player Position ----")
            print(f"=> {(self.current_player_num - self.trick_start_pos) % self.n_players}")

            trick_str = ""
            for card in self.current_trick:
                card_str = card_to_string(card)
                trick_str += f"{card_str} "
            print("---- Current Trick ----")
            print(f"=> {trick_str}")

            player_cards_str = ""
            for card in self.players[self.current_player_num].hand:
                card_str = card_to_string(card)
                player_cards_str += f"{card_str} "

            print("---- Your Cards ----")
            print(f"=> {player_cards_str}")
        else:
            print("---- Player Scores ----")
            min_score = math.inf
            winner = None
            for player in self.players:
                print(f"Player {player.id}:{player.score}")
                if player.score < min_score:
                    min_score = player.score
                    winner = player.id
            
            print(f"Player {winner} wins!")

    def close(self):
        return

def format_card(card):
    # handle empty card
    if card == -1:
        return card, ""
    
    card_num = card % 13
    card_suit = None
    if card < 13:
        card_suit = "s"
    elif card < 26:
        card_suit = "c"
    elif card < 39:
        card_suit = "d"
    elif card < 52:
        card_suit = "h"

    return card_num, card_suit

def card_to_string(card):
    # handle empty card
    if card == -1:
        return card
    else:
        return cards[card]

class Player():
    def __init__(self, id):
        self.id = id
        self.score = 0
        self.hand = []

    def discard(self, card):
        self.hand[self.hand.index(card)] = -1

    def reset(self):
        self.score = 0
        self.hand = []

class Deck():
    def __init__(self):
        self.cards = [x for x in range(52)]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n):
        drawn = []
        for i in range(n):
            drawn.append(self.cards.pop())
        return drawn