from hearts.envs.Hearts import HeartsEnv
import numpy as np

# _get_obs() unit test
def obs_return():
    env = HeartsEnv()
    obs, info = env.reset()

    # check observation type
    assert isinstance(obs, np.ndarray), "_get_obs() should return an ndarray"

    # check obs length
    assert len(obs) == 173, f"get_obs() should return a numpy array of length 173. Instead returned {len(obs)} elements"

    # check current trick properly represented in observation
    current_trick = env.current_trick
    obs_current_trick = obs[0:52]
    for i in range(52):
        if i in env.current_trick:
            assert obs_current_trick[i] == 1, "observation missing card in current trick"
        else:
            assert obs_current_trick[i] == 0, "observation contains card not in current trick"

    # check correct values of observation
    player_id = env.current_player_num
    player_hand = env.players[player_id].hand
    obs_player_hand = obs[52:104]
    for i in range(52):
        if i in player_hand:
            assert obs_player_hand[i] == 1, f"player's card missing from observation: {i} (i) - {obs_player_hand[i]} == 0 - (Player's Hand) {player_hand}"
        else:
            assert obs_player_hand[i] == 0, f"observation contains card not in player's hand: {i} (i) - {obs_player_hand[i]} == 1 - (Player's Hand) {player_hand}"

    # check players position is correct
    player_position = (player_id - env.trick_start_pos) % env.n_players
    obs_position = obs[104: 108]
    for i in range(4):
        if i == player_position:
            assert obs_position[i] == 1, "player position false negative"
        else:
            assert obs_position[i] == 0, "player position false positive"   
    
    # check remaining cards correct
    remaining_cards = env.remaining_cards
    obs_remaining_cards = obs[108: 160]
    for i in range(52):
        if i in remaining_cards:
            assert obs_remaining_cards[i] == 1, "observation doesn't include a remaining card"
        else:
            assert obs_remaining_cards[i] == 0, "observation includes a card that is not remaining"
            
    # check legal actions are correct
    # trick_suit = env.current_trick_suit
    # obs_legal_actions = obs[160:]
    # for i in range(13):
    #     card = player_hand[i]
    #     card_num, card_suit = env.format_card(card)


    


if __name__ == "__main__":
    obs_return()
    print("All tests passed.")
