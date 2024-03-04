from hearts.envs.Hearts import HeartsEnv, Deck, Player
import unittest
import numpy as np
import copy

class TestObservationMethod(unittest.TestCase):

    def setUp(self):
        self.env = HeartsEnv()
        self.obs, _ = self.env.reset()

    # check observation type
    def test_return_type(self):
        self.assertIsInstance(self.obs, np.ndarray, "_get_obs() should return an ndarray")
    
    # check obs length
    def test_return_length(self):
        self.assertEqual(len(self.obs), 173, f"get_obs() should return a numpy array of length 173. Instead returned {len(self.obs)} elements")
    
    # check current trick properly represented in observation
    def test_trick_encoding(self):
        obs_current_trick = self.obs[0:52]
        for i in range(52):
            if i in self.env.current_trick:
                self.assertEqual(obs_current_trick[i], 1, "observation missing card in current trick")
            else:
                self.assertEqual(obs_current_trick[i], 0, "observation contains card not in current trick")
    
    # check correct values of observation
    def test_playerhand_encoding(self):
        player_id = self.env.current_player_num
        player_hand = self.env.players[player_id].hand
        obs_player_hand = self.obs[52:104]
        for i in range(52):
            if i in player_hand:
                self.assertEqual(obs_player_hand[i], 1, f"player's card missing from observation: {i} (i) - {obs_player_hand[i]} == 0 - (Player's Hand) {player_hand}")
            else:
                self.assertEqual(obs_player_hand[i], 0, f"observation contains card not in player's hand: {i} (i) - {obs_player_hand[i]} == 1 - (Player's Hand) {player_hand}")

    # check players position is correct
    def test_playerposition_encoding(self):
        player_position = (self.env.current_player_num - self.env.trick_start_pos) % self.env.n_players
        obs_position = self.obs[104: 108]
        for i in range(4):
            if i == player_position:
                assert obs_position[i] == 1, "player position false negative"
            else:
                assert obs_position[i] == 0, "player position false positive"
    
    # check remaining cards correct
    def test_remainingcards_encoding(self):
        player_id = self.env.current_player_num
        player_hand = self.env.players[player_id].hand
        remaining_cards = self.env.remaining_cards
        obs_remaining_cards = self.obs[108: 160]
        for i in range(52):
            if i in remaining_cards and i not in player_hand:
                assert obs_remaining_cards[i] == 1, "observation doesn't include a remaining card"
            else:
                assert obs_remaining_cards[i] == 0, "observation includes a card that is not remaining"
            
    # check legal actions are correct
    # trick_suit = env.current_trick_suit
    # obs_legal_actions = obs[160:]
    # for i in range(13):
    #     card = player_hand[i]
    #     card_num, card_suit = env.format_card(card)
                
class TestFormatCardMethod(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, _ = self.env.reset()
    
    def test_negative_case(self):
        self.assertEqual(self.env.format_card(-1), (-1, ""), f"format card producing incorrect values. card: {-1} expected value {(-1, '')}")
    
    def test_normal_case(self):
        self.assertEqual(self.env.format_card(13), (0, "c"), f"format card producing incorrect values. card: {13} expected value {(0, 'c')}")

    def test_lower_bound(self):
        self.assertEqual(self.env.format_card(0), (0, "s"), f"format card producing incorrect values. card: {0} expected value {(0, 's')}")

    def test_upper_bound(self):
        self.assertEqual(self.env.format_card(51), (12, "h"), f"format card producing incorrect values. card: {51} expected value {(12, 'h')}")

class TestDeckClass(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, _ = self.env.reset()
        self.deck = Deck()
        self.deck.shuffle()
        self.old_cards = copy.deepcopy(self.deck.cards)
        self.deck.shuffle()
        self.cards = self.deck.cards

    def test_shuffle_method(self):
        self.assertNotEqual(self.old_cards, self.cards, "deck not producing random values")

    # test that the draw method has correct length and deck has no duplicates
    def test_draw_method(self):
        players = []
        for i in range(4):
            player_hand = self.deck.draw(13)

            # check length is correct
            self.assertEqual(len(player_hand), 13, "draw returns incorrect number of cards")

            # check player's hand does not contain duplicate cards
            self.assertEqual(len(player_hand), len(set(player_hand)), "player hand contains duplicates")

class TestResetMethods(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        self.old_players = copy.deepcopy(self.env.players)

    def test_reset_round_method(self):
        self.env.reset_round()

        for i, player in enumerate(self.env.players):
            # check score has reset and player has a new hand
            self.assertNotEqual(player.hand, self.old_players[i].hand, "player hand has not changed")

            # check that every player has correct starting cards
            if i != self.env.trick_start_pos:
                self.assertNotIn(-1, player.hand, "player have correct starting cards")
            else:
                self.assertEqual(len(set(player.hand)), 13, "starting player has multiple missing cards")

            # check trick start pos correctly assigned
            if -1 in player.hand:
                self.assertEqual(self.env.trick_start_pos, i, "trick start position incorrect")

            # check scores are preserved between rounds
            self.assertEqual(player.score, self.old_players[i].score, "score is not preserved between rounds")

        # check that 13 is in current trick
        for i, card in enumerate(self.env.current_trick):
            if i == self.env.trick_start_pos:
                self.assertEqual(card, 13, "starting trick has wrong initial value")
            else:
                self.assertEqual(card, -1, "starting trick has wrong initial values")

        # check current trick suit is clubs
        self.assertEqual(self.env.current_trick_suit, "c", "starting trick is not clubs")

        # check current player number is correct
        self.assertEqual(self.env.current_player_num, (self.env.trick_start_pos - 1) % self.env.n_players, "current player number incorrect")

        # check remaining card tracker is correct
        self.assertEqual(len(set(self.env.remaining_cards)), 51, "remaining card tracker incorrect")

    def test_reset_method(self):
        # check player score was reset
        for i, player in enumerate(self.old_players):
            self.assertEqual(player.score, 0, "player score has not reset")

class TestNormalStep(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        self.player_id = self.env.current_player_num
        self.player_cards = self.env.players[self.player_id].hand

        # play a card
        self.card = None
        for i, card in enumerate(self.player_cards):
            if card != -1 and self.env.legal_actions[i] == 1:
                self.obs, self.reward, self.terminated, _ = self.env.step(i)
                self.index = i
                self.card = card
                break

    
    def test_trick_contains_card(self):
        # check trick contains card
        self.assertEqual(self.env.current_trick[self.player_id], self.card, "trick does not contain card played.")

    def test_playernum_updated(self):
        # check current player was updated
        self.assertEqual(self.env.current_player_num, (self.player_id - 1) % self.env.n_players, "current player incorrectly updated")
        
class TestTrickStart(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        self.card = 0
        for i, player in enumerate(self.env.players):
            for j in range(13):
                player.hand[j] = self.card
                self.card += 1

        self.env.current_player_num = 0
        self.env.current_trick = [-1 for i in range(4)]
        self.env.current_trick_suit = None
        self.env.trick_start_pos = 0

        self.player_id = 0
        self.card = 5
        self.obs, self.reward, self.terminated, _ = self.env.step(self.card)

    def test_trick_suit(self):
        # check trick suit was properly updated
        self.assertEqual(self.env.current_trick_suit, "s", "trick suit not updated on start")

    def test_trick_contains_card(self):
        # check trick contains player card
        self.assertEqual(self.env.current_trick[self.player_id], self.card, "trick does not contain card played.")

    def test_player_updated(self):
        # check current player was updated
        self.assertEqual(self.env.current_player_num, (self.player_id - 1) % self.env.n_players, "current player incorrectly updated")

    def test_remaining_cards_updated(self): 
        # check remaining cards was updated
        self.assertNotIn(5, self.env.remaining_cards, "card not removed from remaining cards")

    def test_obs_type(self):
        self.assertIsInstance(self.obs, np.ndarray, "step should return observation as an ndarray")

    def test_reward_type(self):
        self.assertIsInstance(self.reward, list, "reward should be a list of rewards for each player")

    def test_termination_type(self):
        self.assertIsInstance(self.terminated, bool, "termination flag should be a boolean value")

    def test_termination_value(self):
        self.assertEqual(self.terminated, False, "terminated flag should be false")

    def test_reward_value(self):
        self.assertEqual(self.reward, [0 for i in range(4)], "Should be 0 reward for each player")

class TestStepEndTrick(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        card = 0
        for i, player in enumerate(self.env.players):
            for j in range(13):
                player.hand[j] = card
                card += 1

        self.env.current_player_num = 0
        self.env.current_trick = [-1 for i in range(4)]
        self.env.current_trick_suit = None
        self.env.trick_start_pos = 0

        # player 0 plays Qs
        self.obs, self.reward, self.terminated, _ = self.env.step(10) # Qs

        # player 3 plays Jh
        self.obs, self.reward, self.terminated, _ = self.env.step(9) # Jh

        # player 2 plays 6d
        self.obs, self.reward, self.terminated, _ = self.env.step(4) # 6d

        # player 1 plays 3c
        self.obs, self.reward, self.terminated, _ = self.env.step(1) # 3c

    def test_reward(self):
        self.assertEqual(sum(self.reward), -0.14, "reward not correct")

    def test_player_score_updated(self):
        self.assertEqual(self.env.players[0].score, 14, "player score not updated")

    def test_trick_start_updated(self):
        self.assertEqual(self.env.trick_start_pos, 0, "trick start position not updated to winner")

    def test_current_player_updated(self):
        self.assertEqual(self.env.current_player_num, 0, "current player not updated to winner")

    def test_trick_suit_reset(self):
        self.assertEqual(self.env.current_trick_suit, None, "trick's suit not reset")

    def test_trick_reset(self):
        self.assertEqual(self.env.current_trick, [-1 for i in range(4)], "trick not reset")

    def test_termination_value(self):
        self.assertEqual(self.terminated, False, "terminated flag should be false")


class TestStepEndRound():
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        card = 0
        self.old_players = copy.deepcopy(self.env.players)
        for i, player in enumerate(self.env.players):
            player.hand[i] = card
            card += 1

        self.env.current_player_num = 0
        self.env.current_trick = [-1 for i in range(4)]
        self.env.current_trick_suit = None
        self.env.trick_start_pos = 0
        self.env.remaining_cards = [i if i < 4 else 0 for i in range(52)]

        player_id = 0
        card = 0 # 2s
        obs, reward, terminated, _ = self.env.step(0)

        player_id = 1
        card = 1 # 3s
        obs, reward, terminated, _ = self.env.step(0)

        player_id = 2
        card = 2 # 4s
        obs, reward, terminated, _ = self.env.step(0)

        player_id = 3
        card = 3 # 5s
        obs, reward, terminated, _ = self.env.step(0)
    
    def test_round_end(self):
        for i, player in enumerate(self.env.players):
            self.assertNotEqual(player.hand, self.old_players[i].hand, "round has not been reset")

class TestStepGameEnd(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        card = 0
        
        for i, player in enumerate(self.env.players):
            player.hand = [-1 for i in range(13)]
            player.hand[0] = card
            card += 1
        self.old_players = copy.deepcopy(self.env.players)

        self.env.current_player_num = 0
        self.env.current_trick = [-1 for i in range(4)]
        self.env.current_trick_suit = None
        self.env.trick_start_pos = 0
        self.env.remaining_cards = [i if i < 4 else 0 for i in range(52)]

        card = 0 # 2s
        obs, reward, terminated, _ = self.env.step(0)

        card = 3 # 5s
        self.env.players[3].score = 100
        obs, reward, terminated, _ = self.env.step(0)

        card = 2 # 4s
        obs, reward, terminated, _ = self.env.step(0)

        card = 1 # 3s
        self.obs, self.reward, self.terminated, _ = self.env.step(0)

    def test_end_of_game(self):
        self.assertEqual(self.terminated, True, "termination flag not set at game end")

class TestNormalLegalActions(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()

        for i, player in enumerate(self.env.players):
            player.hand = [-1 for i in range(13)]

        self.env.players[0].hand[0] = 18

        self.env.current_player_num = 0
        self.env.current_trick = [-1 for i in range(4)]
        self.env.current_trick[1] = 13 # start trick with 2c
        self.env.current_trick_suit = "c"
        self.env.trick_start_pos = 1

        self.legal_actions = self.env.legal_actions

    def test_legal_action_type(self):
        self.assertIsInstance(self.legal_actions, np.ndarray, "legal actions not returning a numpy array")

    def test_legal_action_len(self):
        self.assertEqual(len(self.legal_actions), 13)

    def test_legal_action_boolean(self):
        for i in self.legal_actions:
            if i == 0:
                self.assertEqual(i, 0, "legal actions contains non boolean values")
            else:
                self.assertEqual(i, 1, "legal actions contains non boolean values")
    
    def test_legal_actions_values(self):
        player_id = self.env.current_player_num
        for i, card in enumerate(self.env.players[player_id].hand):
            card_num, card_suit = self.env.format_card(card)
            if card_suit == "c":
                self.assertEqual(self.legal_actions[i], 1, "legal action contains card not of trick suit")

class TestLegalActionsAnyCard(unittest.TestCase):
    def setUp(self):
        self.env = HeartsEnv()
        self.obs, self.info = self.env.reset()
        card = 0
        for i, player in enumerate(self.env.players):
            for j in range(13):
                player.hand[j] = card
                card += 1

        self.env.current_player_num = 0
        self.env.current_trick = [-1 for i in range(4)]
        self.env.current_trick[1] = 13 # start trick with 2c
        self.env.current_trick_suit = "c"
        self.env.trick_start_pos = 1

        self.legal_actions = self.env.legal_actions

    def test_legal_actions_values(self):
        player_id = 0
        for i, card in enumerate(self.env.players[player_id].hand):
            if card != -1:
                self.assertEqual(self.legal_actions[i], 1, "legal actions contains invalid cards")

def test_card_to_string(env, card, exp_val):
    pass

if __name__ == "__main__":
    unittest.main()
    print("All tests passed.")
