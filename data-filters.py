
# This code primarily serves to outline methodology and logic for filtering are larger population of census blocks
# based on data-quality crieria of using a collection of many landsat images to equally represent summertime tree canopy
# We first validate that there were a relatively equal number of scenes availble for each census block, then we also calculate
# the average number of pixels available for each census block across all images
# this helps to ensure that the data is not biased by a few images with many pixels and many images with no pixels,
# which would not be representative of the entire summer season or blocks that may only have trees in a fraction of the 
# block. We also filter out blocks with very low population density, low tree canopy, and small land area to ensure that
# the analysis or statistical models are using the most ecologically comparable blocks.

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "Data"


# ============================================================
# INPUT FILE
# ============================================================

input_file = (
    DATA_DIR
    / "remote_sensing_analysis"
    / "moore_tree_quality_blocks.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(input_file)


# ============================================================
# LANDSAT PIXEL-COUNT COLUMNS
# ============================================================
# Pixel counts are calculated during Zonal Statistics procedure
# Using a center within approach for inclusion of pixels in a block sample

pixel_count_columns = [
    "160417L7_C","160418L8_C","160519L7_C","160527L8_C",
    "160612L8_C","160621L8_C","160723L8_C","160807L7_C",
    "160808L8_C","160823L7_C", "160824L8_C","160916L8_C",
    "160925L8_C","161010L7_C","161019L7_C","170507L8_C",
    "170515L7_C","170523L8_C","170616L7_C","170624L8_C",
    "170702L7_C", "170709L7_C","170717L8_C","170718L7_C",
    "170802L8_C","170803L7_C","170826L7_C","170904L7_C",
    "170920L7_C","171022L7_C","180407L7_C","180501L8_C",
    "180517L8_C","180525L7_C", "180704L8_C","180713L8_C",
    "180805L8_C","180806L7_C","181009L7_C","181016L7_C"
]


# ============================================================
# CALCULATE AVERAGE PIXEL COUNT
# ============================================================

# Zero indicates that no valid pixels were available for that
# scene, so zeros are excluded from the average.

pixel_counts_no_zero = (
    df[pixel_count_columns]
    .replace(0, np.nan)
)

df["avg_pix_co"] = (
    pixel_counts_no_zero
    .mean(axis=1, skipna=True)
)

# ============================================================
# CALCULATE SCENE COUNT
# ============================================================

# Count scenes with at least one valid pixel.
# Zero-valued scenes are not counted.

df["scene_co"] = (
    df[pixel_count_columns]
    .ne(0)
    .sum(axis=1)
)


# ============================================================
# CALCULATE SCENE AND PIXEL PROPORTIONS
# ============================================================

# Total number of possible scenes represented by the columns above.
TOTAL_SCENES = len(pixel_count_columns)

df["scene_co_p"] = (
    df["scene_co"]
    / TOTAL_SCENES
)

# max_pix_co represents the maximum possible number of 30 m
# pixels within each census block and was calculated separately
# from a complete NDVI composite.

df["avg_pix_p"] = (
    df["avg_pix_co"]
    / df["max_pix_co"].replace(0, np.nan)
)


# ============================================================
# VERIFY DATA-QUALITY FILTERS
# ============================================================

failed_filters = df[
    # Practical Filters
    (df["pop_dens"] < 2.5)
    | (df["canopy_per"] < 0.1)
    | (df["ALAND10_x"] < 9000)
    # Data-quality filters
    | (df["scene_co_p"] < 0.7)
    | (df["avg_pix_p"] < 0.7)
]


# ============================================================
# SUMMARY
# ============================================================

if len(failed_filters) == 0:
    print(
        "Data-quality check passed: "
        "0 rows fail the filtering criteria."
    )
else:
    print(
        f"WARNING: {len(failed_filters):,} rows fail "
        f"the filtering criteria."
    )