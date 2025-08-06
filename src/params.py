
VAL_PCT = 0.15
TEST_PCT = 0.15

# Mapping from RGB colors to class IDs
COLOR_TO_CLASS = {
    (255, 0, 0): 1,       # red (tomato)
    (0, 255, 0): 2,       # green (leaf)
    (0, 0, 255): 3,       # blue (vase)
    (0, 125, 125): 4,     # floor
    (0, 255, 255): 5      # trunk
}

# Mapping from class ID to RGB color (used for visualization)
CLASS_TO_COLOR = {
    0: (0, 0, 0),         # background (black)
    1: (255, 0, 0),       # red
    2: (0, 255, 0),       # green
    3: (0, 0, 255),       # blue
    4: (0, 125, 125),     # dark yellow
    5: (0, 255, 255)      # yellow
}
