# The workflow integrates:

# Landsat 7 ETM+ (Enhanced Thematic Mapper Plus)
# Landsat 8 OLI (Operational Land Imager)
# Landsat 8 TIRS (Thermal Infrared Sensor)
# The case study focuses on Cuyahoga County, Ohio.

#Preprocessing Notes
#To improve compatibility between Landsat 7 and Landsat 8 datasets, band-level transformations described in Roy et al. (2016) are applied to Landsat 7 surface reflectance data.

#Reference: Roy, D. P., et al. (2016).
#Characterization of Landsat-7 to Landsat-8 reflective wavelength and normalized difference vegetation index continuity.
#Remote Sensing of Environment.https://doi.org/10.1016/j.rse.2015.12.024

# This code excludes landsat scenes with overall cloud cover greater than 20%.
# This code also removes cloud and cloud-shadow pixels using the QA_PIXEL band. Missing data at scene or pixel levels are resolved using the 
# data-filter code included in this repository. The exported images are saved to your Google Drive in a folder named "landsat_imagery_cq" assuming
# you have GEE account, space on google drive, and theinitialized and authenticated the GEE Python API and replaced the appropriate lines of code below
# with your project ID and assets
# All exported images should be hand inspected before aggregating or calculation of zonal statistics. Our final included images were less than that
# exported ultimately due to Cuyahoga County intersecting several landsat tiles thus having some scenes where only a small fraction of the county 
# is included in the image. Examples of this can be identified often by having two images from the same date but different tiles. However,
# this is contingent on the size of your area of interest, where small areas that span less than a landsat tile may not have this issue.

import ee

# ============================================================================
# 0. INITIALIZE EARTH ENGINE
# ============================================================================

# Run ee.Authenticate() once if needed.
# ee.Authenticate()

ee.Initialize(project="ee- YOUR USERNAME")  # Replace with your GEE project ID


# ============================================================================
# 1. STUDY AREA
# ============================================================================
Cuyahoga_County_Boundary = ee.FeatureCollection(
    "path to your asset or geometry that is your area of interest"
)

COUNTY = Cuyahoga_County_Boundary.geometry()


# ============================================================================
# 2. COMMON SETTINGS
# ============================================================================

# Same year range used in original scripts
YEAR_RANGE = ee.Filter.calendarRange(2016, 2018, "year")

# Scene-level cloud filter
CLOUD_FILTER = ee.Filter.lt("CLOUD_COVER", 20)

# Original Landsat 7 NDVI date range
L7_NDVI_DATE_RANGE = ee.Filter.dayOfYear(91, 305)

# Original Landsat 8 NDVI date range
L8_NDVI_DATE_RANGE = ee.Filter.dayOfYear(91, 305)

# Original Landsat 8 LST date range
L8_LST_DATE_RANGE = ee.Filter.dayOfYear(121, 273)

# Drive folder for individual NDVI scenes
DRIVE_FOLDER = "landsat_imagery_cq"


# ============================================================================
# 3. CLOUD MASK FUNCTION
# ============================================================================

def cloud_mask(image):
    """
    Mask cloud and cloud-shadow pixels using QA_PIXEL
    bits 3 and 4.
    """

    qa = image.select("QA_PIXEL")

    mask = (
        qa.bitwiseAnd(1 << 3)
        .Or(qa.bitwiseAnd(1 << 4))
    )

    return image.updateMask(mask.Not())


# ============================================================================
# 4. LANDSAT 7 NDVI
# ============================================================================

def add_ndvi_l7(image):
    """
    Calculate Landsat 7 NDVI after:
      1. Applying Collection 2 surface reflectance scale/offset
      2. Harmonizing L7 ETM+ to L8 OLI using Roy et al. transformation
      3. Masking invalid reflectance and NDVI values
    """

    # Convert DN to surface reflectance
    sr_b4 = (
        image.select("SR_B4")
        .multiply(0.0000275)
        .add(-0.2)
    )

    sr_b3 = (
        image.select("SR_B3")
        .multiply(0.0000275)
        .add(-0.2)
    )

    # Roy et al. (2016) spectral harmonization
    nir = (
        sr_b4
        .multiply(0.8462)
        .add(0.0412)
    )

    red = (
        sr_b3
        .multiply(0.9047)
        .add(0.0061)
    )

    # Calculate NDVI
    ndvi = (
        nir.subtract(red)
        .divide(nir.add(red))
        .rename("NDVI")
    )

    # Valid reflectance / NDVI mask
    valid_mask = (
        red.gt(0)
        .And(nir.gt(0))
        .And(ndvi.gte(-1))
        .And(ndvi.lte(1))
    )

    return image.addBands(
        ndvi.updateMask(valid_mask)
    )


def mask_to_county(image):
    """
    Clip Landsat 7 imagery to county boundary.
    Matches original L7 script behavior.
    """

    clipped = image.clip(COUNTY)

    return clipped.updateMask(
        clipped.mask()
    )


# Load Landsat 7 imagery
l7_collection = (
    ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
    .filterBounds(COUNTY)
    .filter(L7_NDVI_DATE_RANGE)
    .filter(YEAR_RANGE)
    .filter(CLOUD_FILTER)
    .map(cloud_mask)
    .select(["SR_B4", "SR_B3"])
)

# Calculate NDVI and mask to county
l7_ndvi = (
    l7_collection
    .map(add_ndvi_l7)
    .map(mask_to_county)
)

# Number of Landsat 7 scenes
l7_count = l7_ndvi.size().getInfo()

print(f"Number of Landsat 7 NDVI scenes: {l7_count}")


# ============================================================================
# 5. EXPORT LANDSAT 7 NDVI TIME SERIES
# ============================================================================

l7_sorted = l7_ndvi.sort("system:time_start")

l7_list = l7_sorted.toList(l7_count)

for i in range(l7_count):

    scene = ee.Image(l7_list.get(i))

    date = (
        ee.Date(scene.get("system:time_start"))
        .format("YYMMdd")
        .getInfo()
    )

    description = f"{date}L7"

    task = ee.batch.Export.image.toDrive(
        image=scene.select("NDVI"),
        description=description,
        folder=DRIVE_FOLDER,
        region=COUNTY,
        scale=30,
        crs="EPSG:4326",
        fileFormat="GeoTIFF"
    )

    task.start()

    print(f"Started L7 NDVI export: {description}")


# Print L7 acquisition dates
l7_dates = (
    l7_ndvi
    .aggregate_array("system:time_start")
    .getInfo()
)

print("\nLandsat 7 NDVI Acquisition Dates:")

for date in l7_dates:

    formatted_date = (
        ee.Date(date)
        .format("YYYY-MM-dd")
        .getInfo()
    )

    print(formatted_date)


# ============================================================================
# 6. LANDSAT 8 NDVI
# ============================================================================

def add_ndvi_l8(image):
    """
    Calculate Landsat 8 NDVI using scaled surface reflectance.
    """

    red = (
        image.select("SR_B4")
        .multiply(0.0000275)
        .add(-0.2)
    )

    nir = (
        image.select("SR_B5")
        .multiply(0.0000275)
        .add(-0.2)
    )

    # Calculate NDVI
    ndvi = (
        nir.subtract(red)
        .divide(nir.add(red))
        .rename("NDVI")
    )

    # Valid reflectance / NDVI mask
    valid_mask = (
        red.gt(0)
        .And(nir.gt(0))
        .And(ndvi.gte(-1))
        .And(ndvi.lte(1))
    )

    return image.addBands(
        ndvi.updateMask(valid_mask)
    )


# Load Landsat 8 imagery
l8_ndvi_collection = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(COUNTY)
    .filter(L8_NDVI_DATE_RANGE)
    .filter(YEAR_RANGE)
    .filter(CLOUD_FILTER)
    .map(cloud_mask)
    .select(["SR_B5", "SR_B4"])
)

# Calculate NDVI
l8_ndvi = l8_ndvi_collection.map(add_ndvi_l8)

# Number of Landsat 8 scenes
l8_ndvi_count = l8_ndvi.size().getInfo()

print(f"\nNumber of Landsat 8 NDVI scenes: {l8_ndvi_count}")


# ============================================================================
# 7. EXPORT LANDSAT 8 NDVI TIME SERIES
# ============================================================================

l8_sorted = l8_ndvi.sort("system:time_start")

l8_list = l8_sorted.toList(l8_ndvi_count)

for i in range(l8_ndvi_count):

    scene = ee.Image(l8_list.get(i))

    date = (
        ee.Date(scene.get("system:time_start"))
        .format("YYMMdd")
        .getInfo()
    )

    description = f"{date}L8"

    task = ee.batch.Export.image.toDrive(
        image=scene.select("NDVI"),
        description=description,
        folder=DRIVE_FOLDER,
        region=COUNTY,
        scale=30,
        crs="EPSG:4326",
        fileFormat="GeoTIFF"
    )

    task.start()

    print(f"Started L8 NDVI export: {description}")


# Print L8 acquisition dates
l8_dates = (
    l8_ndvi
    .aggregate_array("system:time_start")
    .getInfo()
)

print("\nLandsat 8 NDVI Acquisition Dates:")

for date in l8_dates:

    formatted_date = (
        ee.Date(date)
        .format("YYYY-MM-dd")
        .getInfo()
    )

    print(formatted_date)


# ============================================================================
# 8. LANDSAT 8 LAND SURFACE TEMPERATURE
# ============================================================================

def apply_lst_scale_factors(image):
    """
    Convert Landsat Collection 2 ST_B10 DN values
    to land surface temperature in degrees Celsius.

    Kelvin:
        DN * 0.00341802 + 149.0

    Celsius:
        Kelvin - 273.15
    """

    thermal_c = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)
    )

    return image.addBands(
        thermal_c,
        None,
        True
    )


# Load Landsat 8 thermal imagery
l8_lst_collection = (
    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .select(["ST_B10", "QA_PIXEL"])
    .filterBounds(COUNTY)
    .filter(L8_LST_DATE_RANGE)
    .filter(YEAR_RANGE)
    .map(cloud_mask)
    .filter(CLOUD_FILTER)
)

# Print raw filtered LST collection size
l8_lst_count = l8_lst_collection.size().getInfo()

print(
    f"\nNumber of Landsat 8 LST images after filtering: "
    f"{l8_lst_count}"
)


# ============================================================================
# 9. CALCULATE MEAN LST COMPOSITE
# ============================================================================

landsat_st = l8_lst_collection.map(
    apply_lst_scale_factors
)

# Pixel-wise mean
mean_landsat_st = landsat_st.mean()

# Clip to county
clip_mean_st = mean_landsat_st.clip(COUNTY)

# Select LST band
values_st = clip_mean_st.select("ST_B10")


# ============================================================================
# 10. EXPORT LANDSAT 8 MEAN LST
# ============================================================================

lst_task = ee.batch.Export.image.toDrive(
    image=clip_mean_st,
    folder=DRIVE_FOLDER,
    description="LST_cuyahoga_2017",
    scale=30,
    region=COUNTY,
    crs="EPSG:4326", # data was reprojected after export in QGIS
    fileFormat="GeoTIFF"
)

lst_task.start()

print("\nStarted LST export: LST_cuyahoga_2017")


# ============================================================================
# 11. PRINT LST ACQUISITION DATES
# ============================================================================

lst_dates = (
    l8_lst_collection
    .aggregate_array("system:time_start")
    .getInfo()
)

print("\nLandsat 8 LST Acquisition Dates:")

for date in lst_dates:

    formatted_date = (
        ee.Date(date)
        .format("YYYY-MM-dd")
        .getInfo()
    )

    print(formatted_date)


# ============================================================================
# 12. SUMMARY
# ============================================================================

print("\n================================================")
print("PROCESSING COMPLETE")
print("================================================")

print(f"Landsat 7 NDVI scenes: {l7_count}")
print(f"Landsat 8 NDVI scenes: {l8_ndvi_count}")
print(f"Landsat 8 LST scenes:  {l8_lst_count}")

print("\nEarth Engine export tasks have been submitted.")
print("Check the Earth Engine Tasks page / Google Drive for outputs.")