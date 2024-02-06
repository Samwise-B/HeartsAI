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

class TestStepMethod(unittest.TestCase):
    def setUp(self):
        pass


def test_card_to_string(env, card, exp_val):
    pass

if __name__ == "__main__":
    unittest.main()
    print("All tests passed.")
