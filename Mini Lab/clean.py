# clean.py
import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import pandas as pd
from ucimlrepo import fetch_ucirepo

decode_map = {
    'class': {'p': 'poisonous', 'e': 'edible'},
    'cap-shape': {'b': 'bell', 'c': 'conical', 'x': 'convex', 'f': 'flat',
                  's': 'sunken', 'p': 'spherical', 'o': 'others'},
    'cap-surface': {'i': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth',
                    'h': 'shiny', 'l': 'leathery', 'k': 'silky', 't': 'sticky',
                    'w': 'wrinkled', 'e': 'fleshy'},
    'cap-color': {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
                  'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow',
                  'l': 'blue', 'o': 'orange', 'k': 'black'},
    'does-bruise-or-bleed': {'t': 'yes', 'f': 'no'},
    'gill-attachment': {'a': 'adnate', 'x': 'adnexed', 'd': 'decurrent', 'e': 'free',
                        's': 'sinuate', 'p': 'pores', 'f': 'none', '?': 'unknown'},
    'gill-spacing': {'c': 'close', 'd': 'distant', 'f': 'none'},
    'gill-color': {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
                   'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow',
                   'l': 'blue', 'o': 'orange', 'k': 'black', 'f': 'none'},
    'stem-root': {'b': 'bulbous', 's': 'swollen', 'c': 'club', 'u': 'cup',
                  'e': 'equal', 'z': 'rhizomorphs', 'r': 'rooted', 'f': 'fibrous'},
    'stem-surface': {'i': 'fibrous', 'g': 'grooves', 'y': 'scaly', 's': 'smooth',
                     'h': 'shiny', 'l': 'leathery', 'k': 'silky', 't': 'sticky',
                     'w': 'wrinkled', 'e': 'fleshy', 'f': 'none'},
    'stem-color': {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
                   'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow',
                   'l': 'blue', 'o': 'orange', 'k': 'black', 'f': 'none'},
    'veil-type': {'p': 'partial', 'u': 'universal'},
    'veil-color': {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
                   'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow',
                   'l': 'blue', 'o': 'orange', 'k': 'black', 'f': 'none'},
    'has-ring': {'t': 'ring', 'f': 'none'},
    'ring-type': {'c': 'cobwebby', 'e': 'evanescent', 'r': 'flaring', 'g': 'grooved',
                  'l': 'large', 'p': 'pendant', 's': 'sheathing', 'z': 'zone',
                  'y': 'scaly', 'm': 'movable', 'f': 'none', '?': 'unknown'},
    'spore-print-color': {'n': 'brown', 'b': 'buff', 'g': 'gray', 'r': 'green', 'p': 'pink',
                          'u': 'purple', 'e': 'red', 'w': 'white', 'y': 'yellow',
                          'l': 'blue', 'o': 'orange', 'k': 'black'},
    'season': {'s': 'spring', 'u': 'summer', 'a': 'autumn', 'w': 'winter'},
    'habitat': {'g': 'grasses', 'l': 'leaves', 'm': 'meadows', 'p': 'paths',
                'h': 'heaths', 'u': 'urban', 'w': 'waste', 'd': 'woods'}
}


def load_raw() -> pd.DataFrame:
    secondary_mushroom = fetch_ucirepo(id=848)
    X = secondary_mushroom.data.features
    y = secondary_mushroom.data.targets
    return pd.concat([X, y], axis=1)


def clean_mushrooms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Decode letter codes to words (matches group notebook section 3)
    for col, mapping in decode_map.items():
        df[col] = df[col].map(mapping)

    # Drop exact duplicates (146 rows, -> 60,923)
    df = df.drop_duplicates().reset_index(drop=True)

    # Fill missing categoricals with an explicit "Missing" category
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Missing")

    # Engineered features (group notebook section 4.7)
    df["has-stem"] = ~((df["stem-height"] == 0) & (df["stem-width"] == 0))
    df["has-stem"] = df["has-stem"].map({True: "yes", False: "no"})
    df["size-score"] = df["cap-diameter"] + df["stem-height"] + df["stem-width"] / 10

    return df