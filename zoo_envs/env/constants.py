# Description: Constants for the zoo_envs package
import pathlib
from pygame import image
from typer.colors import MAGENTA

STILL = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

MOVES = [STILL, UP, DOWN, LEFT, RIGHT]

TILE_SIZE = 32
BACKGROUND_COLOR = (255, 185, 137)
CLEAR = (0, 0, 0, 0)
GRID_LINE_COLOR = (200, 150, 100, 200)
SCREEN_SIZE = (800, 800)

STAG_COLOR = (255, 0, 0)
PLANT_COLOR = (0, 255, 0)
AGENT_COLORS = {"agent_0": (0, 0, 255), "agent_1": (255, 0, 255), "agent_2": (255, 255, 0)}

STAG_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "stag_v2.png"
PLANT_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "plant_no_fruit.png"
POISONED_STAG_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "stag_poisoned_v2.png"

HARE_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "hare.png"

HUNTER_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "hunter.png"

BLUE_HUNTER_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "blue_hunter.png"
MAGENTA_HUNTER_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "magenta_hunter.png"
YELLOW_HUNTER_SPRITE = pathlib.Path(__file__).parent.parent / "assets" / "yellow_hunter.png"

SPRITE_DICT = {
    "agent_0": BLUE_HUNTER_SPRITE,
    "agent_1": MAGENTA_HUNTER_SPRITE,
    "agent_2": YELLOW_HUNTER_SPRITE,
}