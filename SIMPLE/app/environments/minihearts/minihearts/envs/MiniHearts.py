import gym
import random
import numpy as np
import math

from stable_baselines import logger

maxScore = 45
maxCardCount = 24
numPlayerCards = 6
startingCard = 6
QofSpades = 3
QofSpadesValue = 6

cards = {
    0: "2s", 1: "3s", 2: "4s", 3: "5s", 4: "6s", 5: "7s",
    6: "2c", 7: "3c", 8: "4c", 9: "5c", 10: "6c", 11: "7c",
    12: "2d", 13: "3d", 14:"4d", 15: "5d", 16: "6d", 17: "7d", 
    18: "2h", 19: "3h", 20: "4h", 21: "5h", 22: "6h", 23: "7h"
}

class MiniHeartsEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, verbose = False, manual = False):
        super(MiniHeartsEnv, self).__init__()
        self.name = "minihearts"
        self.maxScore = maxScore
        self.verbose = True

        self.n_players = 4
        self.current_player_num = 0
        self.players = [Player(0), Player(1), Player(2), Player(3)]
        self.agent_player_num = None

        # initialise observation space
        self.observation_space = gym.spaces.Box(-1, 1, (
            maxCardCount     # current trick
            + maxCardCount   # player's cards
            + self.n_players              # player's position
            + maxCardCount   # remaining cards
            + numPlayerCards # legal actions
            , )
        )

        # initialise action space
        self.action_space = gym.spaces.Discrete(numPlayerCards)

        self.remaining_cards = [x for x in range(maxCardCount)]

        self.current_trick = [-1 for i in range(self.n_players)]
        self.current_trick_suit = None
        self.trick_start_pos = 0
        self.total_rounds = 0
        self.total_tricks = 0
        self.first_trick_of_round = True
        self.hearts_broken = False

        self.terminated = False

    def _get_obs(self):
        # get current trick values
        trick = np.array(self.current_trick)
        # one-hot encode the trick
        trick_obs = np.zeros(maxCardCount)
        for i, card in enumerate(trick):
            if card != -1:
                trick_obs[card] = 1

        ret = trick_obs

        # get player's hand
        player_id = self.current_player_num
        player_cards = np.full(numPlayerCards, -1)
        player_cards_obs = np.zeros(maxCardCount)
        for i, card in enumerate(self.players[player_id].hand):
            player_cards[i] = card
            player_cards_obs[card] = 1
            
        ret = np.append(ret, player_cards_obs)
        # get all players scores
        # player_scores = np.empty(4)
        # for player in self.players:
        #     player_scores[player.id] = player.score

        # get player's position
        player_position = (self.trick_start_pos - player_id) % self.n_players
        player_pos_obs = np.zeros(self.n_players)
        player_pos_obs[player_position] = 1
        ret = np.append(ret, player_pos_obs)
        
        # get remaining cards
        individuals_remaining_cards = []
        remaining_cards_obs = np.zeros(maxCardCount)
        for card in self.remaining_cards:
            if card not in player_cards:
                individuals_remaining_cards.append(card)
                remaining_cards_obs[card] = 1
            
        ret = np.append(ret, remaining_cards_obs)

        # append legal actions for masking
        legal_actions = self.legal_actions
        ret = np.append(ret, legal_actions)
        #print(ret.shape)
        logger.debug(f"observations:")
        logger.debug(f"trick: {trick_obs}")
        logger.debug(f"player_cards: {player_cards_obs}")
        logger.debug(f"player_position: {player_pos_obs}")
        logger.debug(f"remaining cards: {remaining_cards_obs}")
        logger.debug(f"legal acts: {legal_actions}")


        return ret
                
    def _get_info(self):
        return {}
    
    @property
    def legal_actions(self):
        current_player = self.players[self.current_player_num]
        can_follow_suit = False
        is_leading_trick = not self.current_trick_suit

        legal_actions_any_card = np.zeros(numPlayerCards)
        legal_actions_in_suit = np.zeros(numPlayerCards)
        legal_actions_no_scoring_cards = np.zeros(numPlayerCards)
        legal_actions_no_hearts = np.zeros(numPlayerCards)

        for i, card in enumerate(current_player.hand):
            card_num, card_suit = self.format_card(card)
            # flag any card following suit
            if card_suit == self.current_trick_suit:
                legal_actions_in_suit[i] = 1
                can_follow_suit = True
            # handle cant follow suit cases
            if card != -1:
                # flag any unplayed card
                legal_actions_any_card[i] = 1

                # flag non hearts cards
                if card_suit != "h":
                    legal_actions_no_hearts[i] = 1

                    # no scoring cards
                    if card != QofSpades:
                        legal_actions_no_scoring_cards[i] = 1
        # logger.debug(legal_actions_any_card)
        # logger.debug(legal_actions_in_suit)
        # logger.debug(legal_actions_no_hearts)
        # logger.debug(legal_actions_no_scoring_cards)

        if can_follow_suit:
            return legal_actions_in_suit
        else:
            if self.first_trick_of_round and np.any(legal_actions_no_scoring_cards):
                return legal_actions_no_scoring_cards
            # elif self.first_trick_of_round:
            #     return legal_actions_any_card
            elif is_leading_trick and not self.hearts_broken:
                if np.any(legal_actions_no_hearts):
                    return legal_actions_no_hearts
        return legal_actions_any_card
            # else:
            #     if is_starting_trick and not self.hearts_broken:
            #         if np.any(legal_actions_no_hearts):
            #             return legal_actions_no_hearts
            #     else:
            #         return legal_actions_any_card           

    def reset(self, seed=None):
        # following line to seed self.np_random
        #super().reset(seed=seed)

        for player in self.players:
            # clear player hand and score
            player.reset()

        self.total_rounds = 0
        self.total_tricks = 0

        # reset the round
        self.reset_round()

        # return observation and info
        self.observation = self._get_obs()
        info = self._get_info()

        #if self.render_mode == "human":
        #    self.render()

        # reset termination flag
        self.terminated = False

        return self.observation, info
    
    def reset_round(self):
        # reset and shuffle the deck
        self.deck = Deck()
        self.deck.shuffle()

        self.total_tricks += 1

        # reset remaining card tracker
        self.remaining_cards = [x for x in range(maxCardCount)]

        # reset current trick
        for i, card in enumerate(self.current_trick):
            self.current_trick[i] = -1

        # reset hearts_broken flag
        self.hearts_broken = False

        # reset first trick flag
        self.first_trick_of_round = True
        
        for player in self.players:
            # deal player new hand
            player.hand = self.deck.draw(numPlayerCards)

            # if player has the 2 of clubs
            if startingCard in player.hand:
                # add 2 of clubs to current trick and discard from player's hand
                self.current_trick[player.id] = startingCard
                player.discard(startingCard)

                # set trick suit, start pos and remove 2c from remaining cards
                self.current_trick_suit = "c"
                self.trick_start_pos = player.id
                self.remaining_cards.remove(startingCard)

                # set current player to the left of player with 2 of clubs
                self.current_player_num = (player.id - 1) % self.n_players

    def step(self, action):
        self.terminated = False
        reward = [0] * self.n_players

        self.render_player_hand()

        self.render_trick()

        player_id = self.current_player_num

        if self.legal_actions[action] == 0:
            # handling illegal actions for evaluation callback
            logger.debug(f"Invalid action: {action}, {self.players[player_id].hand}")
            # reward = [-0.01 * player.score + 0.01 * player.turns_taken for player in self.players]
            # reward[self.current_player_num] = -1

            # uncomment for single deal reward
            reward = [(maxCardCount - player.score) / maxCardCount for player in self.players]
            reward[self.current_player_num] = 0

            # reward = [self.total_tricks - player.score for player in self.players]
            # reward[self.current_player_num] = 0
            # reward = [1 for player in self.players]
            # reward[self.current_player_num] = 0
            # binary case
            # scores = [self.players[0].score, self.players[1].score, self.players[2].score, self.players[3].score]
            # reward[scores.index(min(scores))] = 1
            self.terminated = True
        else:
            # set action (index) to card
            action = self.players[player_id].hand[action]
            
            # add card to trick
            self.current_trick[player_id] = action

            # remove card from player's hand
            self.players[player_id].discard(action)
            # remove card from remaining cards list
            self.remaining_cards.remove(action)

            # give player reward for playing card
            #reward[player_id] = 0.01

            logger.debug(f"Played: {self.card_to_string(action)}")

            # handle trick start
            if player_id == self.trick_start_pos:
                card_num, card_suit = self.format_card(action)
                self.current_trick_suit = card_suit
                self.total_tricks += 1
            
            # handle trick end
            if (self.trick_start_pos + 1) % self.n_players == self.current_player_num:
                # check if first trick of round
                if self.first_trick_of_round:
                    self.first_trick_of_round = False

                # check who won the trick and update their score
                trick_score = 0
                max_card = 0
                winner = None
                for player_id, card in enumerate(self.current_trick):
                    card_num, card_suit = self.format_card(card)

                    # check if hearts have been broken
                    if card_suit == "h" and not self.hearts_broken:
                        self.hearts_broken = True
                        logger.debug("hearts broken!")

                    # find the winner of the trick
                    if card_suit == self.current_trick_suit and card_num >= max_card:
                        max_card = card_num
                        winner = player_id
                    
                    # handle trick score
                    if card_suit == "h":
                        trick_score += 1
                    elif card_suit == "s" and card_num == QofSpades:
                        trick_score += QofSpadesValue

                # update score and current player
                self.players[winner].score += trick_score
                self.current_player_num = winner
                self.trick_start_pos = winner

                #reward[winner] += -0.01 * trick_score

                logger.debug(f"Player {winner} won the trick.")

                logger.debug("---- Player Scores ----")
                for player in self.players:
                    logger.debug(f"Player {player.id}:{player.score}")

                # reset the trick suit
                self.current_trick_suit = None

                # reset the trick
                for i, card in enumerate(self.current_trick):
                    self.current_trick[i] = -1

                # handle end of game and end of round
                if self.players[winner].score >= maxScore:
                    self.terminated = True
                    self.total_rounds += len(self.remaining_cards) / maxCardCount
                    logger.debug(f"Total Tricks Played: {self.total_tricks}")
                    logger.debug(f"Total Rounds Played: {self.total_rounds}")
                    # handle reward (only binary case)
                    # scores = [self.players[0].score, self.players[1].score, self.players[2].score, self.players[3].score]
                    # reward[scores.index(min(scores))] = 1
                    # handle reward (terminal, non-binary case)
                    reward = self.score_game()
                elif len(self.remaining_cards) == 0:
                    # comment in to make environment run for a single deal and comment out reset_round
                    self.terminated = True
                    reward = self.score_game()
                    #self.reset_round()
                    self.total_rounds += 1
            else:
                # move to next player
                self.current_player_num = (player_id - 1) % self.n_players
        #self.render()

        self.observation = self._get_obs()
            
        return self.observation, reward, self.terminated, False
    
    def score_game(self):
        reward = [0] * self.n_players
        max_reward = -math.inf
        min_reward = math.inf
        for i, player in enumerate(self.players):
            # reward[i] = self.total_tricks - player.score
            # if reward[i] > max_reward:
            #     max_reward = reward[i]
            # if reward[i] < min_reward:
            #     min_reward = reward[i]
            reward[i] = (maxCardCount - player.score) / maxCardCount
        
        # for i, player in enumerate(self.players):
        #     reward[i] = (reward[i] - min_reward) / (max_reward - min_reward)

        return reward
    
    def rules_move(self):
        player_id = self.current_player_num
        # print(self.render_trick())
        # print(self.render_player_hand())

        # handle non-leading tricks
        if self.trick_start_pos != player_id:
            # check if player can follow suit of the trick
            # and if they have any non-scoring cards
            can_follow_suit = False
            has_non_scoring_card = False
            for i, card in enumerate(self.players[player_id].hand):
                card_num, card_suit = self.format_card(card)
                if card_suit == self.current_trick_suit:
                    can_follow_suit = True
                if card_suit != "h" and card != QofSpades:
                    has_non_scoring_card = True

            #print("has non scoring card:", has_non_scoring_card)
            
            # handle can play any card
            if not can_follow_suit:
                max_card = -1
                max_card_ind = -1
                max_heart_ind = -1
                max_heart = -1
                for i, card in enumerate(self.players[player_id].hand):
                    card_num, card_suit = self.format_card(card)
                    if card == QofSpades and (not self.first_trick_of_round or self.first_trick_of_round and not has_non_scoring_card):
                        # if player has Qs, play it
                        return [1 if j == i else 0 for j in range(numPlayerCards)]
                    elif card_suit == "h":
                        # if player has a heart, track highest heart
                        if max_heart < card_num:
                            max_heart = card_num
                            max_heart_ind = i
                    elif card_num > max_card and card != QofSpades:
                        # track highest non-scoring card
                        max_card = card_num
                        max_card_ind = i
                # handle return array for max heart / card
                if (not self.first_trick_of_round or self.first_trick_of_round and not has_non_scoring_card) and max_heart != -1:
                    return [1 if i == max_heart_ind else 0 for i in range(numPlayerCards)]
                else:
                    return [1 if i == max_card_ind else 0 for i in range(numPlayerCards)]
            else:
                # handle, when player can follow suit
                # get max card of the trick suit
                max_trick_card = -1
                for i, card in enumerate(self.current_trick):
                    card_num, card_suit = self.format_card(card)
                    if card_suit == self.current_trick and card_num > max_trick_card:
                        max_trick_card = card_num
                max_card = -1
                max_card_i = -1
                min_card = math.inf
                min_card_i = math.inf
                # handle following suit
                for i, card in enumerate(self.players[player_id].hand):
                    card_num, card_suit = self.format_card(card)
                    if card_suit == self.current_trick_suit:
                        # if card is of the same suit
                        if card_num > max_card and card_num < max_trick_card:
                            # update max card that is less than max_trick_card
                            max_card = card_num
                            max_card_i = i
                        if card_num < min_card:
                            min_card = card_num
                            min_card_i = i
                # if player has a card less than the current max in the trick, play it
                # otherwise, play the minimum card in the suit
                if max_card_i != -1:
                    return [1 if i == max_card_i else 0 for i in range(numPlayerCards)]
                else:
                    return [1 if i == min_card_i else 0 for i in range(numPlayerCards)]
        else:
            # handle leading trick, play random card
            indexes = []
            # enumerate all indexes of hand which are unplayed cards
            for i, card in enumerate(self.players[player_id].hand):
                if self.legal_actions[i] == 1:
                    indexes.append(i)
            # get a random index
            i = indexes[random.randint(0, len(indexes)-1)]
            return [1 if i == j else 0 for j in range(numPlayerCards)]


    def render(self, mode='human', close=False):
        if not self.terminated:
            logger.debug(f"Player {self.current_player_num}'s Turn")    
        else:
            #logger.debug("---- Player Scores ----")
            min_score = math.inf
            winner = None
            for player in self.players:
                #logger.debug(f"Player {player.id}:{player.score}")
                if player.score < min_score:
                    min_score = player.score
                    winner = player.id
            
            logger.debug(f"Player {winner} wins!")

    def render_player_hand(self):
        player_cards_str = ""
        for card in self.players[self.current_player_num].hand:
            card_str = self.card_to_string(card)
            player_cards_str += f"{card_str} "

        logger.debug(f"---- Player {self.current_player_num} Cards ----")
        logger.debug(f"=> {player_cards_str}")

    def render_trick(self):
        trick_str = ""
        for card in self.current_trick:
            card_str = self.card_to_string(card)
            trick_str += f"{card_str} "
        logger.debug("---- Current Trick ----")
        logger.debug(f"=> {trick_str}")
        player_pos_str = ["  " if i != self.current_player_num else "^" for i in range(self.n_players)]
        logger.debug(f"=> {' '.join(player_pos_str)}   (position)")
        #logger.debug(f"=> {(self.current_player_num - self.trick_start_pos) % self.n_players}")

    def close(self):
        return

    def format_card(self, card):
        # handle empty card
        if card == -1:
            return card, ""
        
        num_suit_cards = (maxCardCount / self.n_players)
        card_num = card % num_suit_cards
        card_suit = None
        if card < num_suit_cards:
            card_suit = "s"
        elif card < num_suit_cards * 2:
            card_suit = "c"
        elif card < num_suit_cards * 3:
            card_suit = "d"
        elif card < num_suit_cards * 4:
            card_suit = "h"

        return card_num, card_suit

    def card_to_string(self, card):
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
        self.cards = [x for x in range(maxCardCount)]

    def shuffle(self):
        random.shuffle(self.cards)

    def draw(self, n):
        drawn = []
        for i in range(n):
            drawn.append(self.cards.pop())
        return drawn