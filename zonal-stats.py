# This code can only be run once image-export.py has been ran successfully and those files have been looked over and projected to desired CRS.
# Images will beed to be downloaded from drive and stored locally in working directory. 

# Code will calculate zonal statistics for each raster in the images folder and save the results to a geopackage file. 
# The output file will contain the mean and count of pixel values for each zone in the input shapefile.


import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "Data"
IMAGE_DIR = PROJECT_DIR / "images"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

input_file = (
    DATA_DIR
    / "remote_sensing_analysis"
    / "moore_tree_quality_blocks.shp"
)

output_file = (
    OUTPUT_DIR
    / "moore_tree_quality_blocks_ndvi_zonal_stats.gpkg"
)


# ============================================================
# LOAD CENSUS BLOCKS
# ============================================================

zones = gpd.read_file(input_file)


# ============================================================
# FIND NDVI RASTERS
# ============================================================

raster_files = sorted(IMAGE_DIR.glob("*.tif"))

if len(raster_files) == 0:
    raise FileNotFoundError(
        f"No .tif raster files found in: {IMAGE_DIR}"
    )

print(f"Found {len(raster_files)} raster files.")


# ============================================================
# ZONAL STATISTICS
# ============================================================

for raster_file in raster_files:

    print(f"Processing: {raster_file.name}")

    with rasterio.open(raster_file) as src:

        # Check that raster and census blocks use the same CRS
        if zones.crs != src.crs:
            raise ValueError(
                f"CRS mismatch for {raster_file.name}\n"
                f"Blocks CRS: {zones.crs}\n"
                f"Raster CRS: {src.crs}"
            )

        raster_data = src.read(1)

        stats = zonal_stats(
            zones.geometry,
            raster_data,
            affine=src.transform,
            nodata=src.nodata,
            stats=["mean", "count"],
            all_touched=False,
        )

    # Use raster filename without .tif as the column prefix
    raster_name = raster_file.stem

    mean_column = f"{raster_name}_mean"
    count_column = f"{raster_name}_count"

    zones[mean_column] = [
        stat["mean"] for stat in stats
    ]

    zones[count_column] = [
        stat["count"] for stat in stats
    ]


# ============================================================
# EXPORT RESULTS
# ============================================================

zones.to_file(
    output_file,
    driver="GPKG",
)

print()
print("=" * 80)
print("NDVI ZONAL STATISTICS COMPLETE")
print("=" * 80)
print(f"Input census blocks: {input_file}")
print(f"Raster folder:       {IMAGE_DIR}")
print(f"Output:              {output_file}")
print(f"Rasters processed:   {len(raster_files)}")