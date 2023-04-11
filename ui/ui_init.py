from utils.read_config import config
from ui.hex import HexUI
from ui.nim import NimUI

def ui_setup():
    if config.game == "hex":
        return HexUI(config.board_size)
    elif config.game == "nim":
        return NimUI(config.nr_of_piles)
    else:
        raise Exception("Game not implemented")