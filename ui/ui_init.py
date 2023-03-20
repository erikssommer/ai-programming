from utility.read_config import config
from ui.hex import HexUI

def ui_setup():
    if config.game == "hex":
        return HexUI(config.board_size)
    elif config.game == "nim":
        raise Exception("UI for nim not implemented")
    else:
        raise Exception("Game not implemented")