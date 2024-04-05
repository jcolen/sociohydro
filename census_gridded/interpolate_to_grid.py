import os
from glob import glob
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.interpolate as interpolate
from shapely.geometry import Point
import h5py

import warnings
warnings.filterwarnings('ignore')

sc_list = pd.read_csv("states_counties.csv")
years = [1980, 1990, 2000, 2010, 2020]

for scidx, sc in tqdm(sc_list.iterrows(), total=len(sc_list)):
    try:
        print(f"{sc.state}, {sc.county}")
        county_folder = f"{sc.state}_{sc.county}"
        gis_files = sorted(glob(os.path.join("counties", county_folder, "*.geojson")))
        pop_files = sorted(glob(os.path.join("counties", county_folder, "*.csv")))

        # First loop - establish geometry
        x_min, y_min = 1e10, 1e10
        x_max, y_max = -1e10, -1e10
        for yidx, gis_file in enumerate(tqdm(gis_files)):
            gis = gpd.read_file(gis_file)
            
            boundary = gpd.GeoDataFrame([gis.unary_union])
            boundary.geometry = boundary[0]
            boundary.crs = gis.crs

            # get indices of all census tracts in nearby area
            # nearby = census tract inside regions 1.5x size of county
            x0, y0, x1, y1 = boundary.boundary[0].bounds
            x_min = min(x0, x_min)
            x_max = max(x1, x_max)
            y_min = min(y0, y_min)
            y_max = max(y1, y_max)
        
        # Establish grid for this county
        xi = np.arange(x_min, x_max, 1000)
        yj = np.arange(y_min, y_max, 1000)
        xgrid, ygrid = np.meshgrid(xi, yj)
        grid_points = gpd.GeoDataFrame({"geometry": [Point(x, y) for x, y in zip(xgrid.ravel(), ygrid.ravel())]})

        for yidx, (gis_file, pop_file) in enumerate(zip(tqdm(gis_files), pop_files)):
            gis = gpd.read_file(gis_file)
            pop = pd.read_csv(pop_file)

            points = np.array([[p.x, p.y] for p in gis.centroid.values])
            
            boundary = gpd.GeoDataFrame([gis.unary_union])
            boundary.geometry = boundary[0]
            boundary.crs = gis.crs

            # points inside region = 0, points outside = 1
            mask = np.ones(np.prod(xgrid.shape))
            grid_points.crs = boundary.crs
            within = gpd.sjoin(grid_points, boundary, how="right", predicate="within")
            mask[within.index_left.values] = 0
            mask = mask.reshape(xgrid.shape)

            # points inside county = 0, points outside county = 1
            gis_county = gis[(pop.STATEA == sc.state_code) & (pop.COUNTYA == sc.county_code)]
            boundary_county = gpd.GeoDataFrame([gis_county.unary_union])
            boundary_county.geometry = boundary_county[0]
            boundary_county.crs = gis_county.crs

            county_mask = np.ones(np.prod(xgrid.shape))
            grid_points.crs = boundary_county.crs
            within = gpd.sjoin(grid_points, boundary_county, how="right", predicate="within")
            county_mask[within.index_left.values] = 0
            county_mask = county_mask.reshape(xgrid.shape)

            white_grid = interpolate.griddata(points, pop.white, (xgrid, ygrid),
                                            fill_value=np.nan, method="linear")
            white_grid_masked = np.ma.array(white_grid, mask=mask,
                                            fill_value=np.nan).filled()
            white_grid_county = np.ma.array(white_grid, mask=county_mask,
                                            fill_value=np.nan).filled()
            
            black_grid = interpolate.griddata(points, pop.black, (xgrid, ygrid),
                                            fill_value=np.nan, method="linear")
            black_grid_masked = np.ma.array(black_grid, mask=mask,
                                            fill_value=np.nan).filled()
            black_grid_county = np.ma.array(black_grid, mask=county_mask,
                                            fill_value=np.nan).filled()

            aapi_grid = interpolate.griddata(points, pop.aapi, (xgrid, ygrid),
                                            fill_value=np.nan, method="linear")
            aapi_grid_masked = np.ma.array(aapi_grid, mask=mask,
                                        fill_value=np.nan).filled()
            aapi_grid_county = np.ma.array(aapi_grid, mask=county_mask,
                                            fill_value=np.nan).filled()

            hispanic_grid = interpolate.griddata(points, pop.hispanic, (xgrid, ygrid),
                                                fill_value=np.nan, method="linear")
            hispanic_grid_masked = np.ma.array(hispanic_grid, mask=mask,
                                            fill_value=np.nan).filled()
            hispanic_grid_county = np.ma.array(hispanic_grid, mask=county_mask,
                                            fill_value=np.nan).filled()

            total_grid = interpolate.griddata(points, pop.total, (xgrid, ygrid),
                                            fill_value=np.nan, method="linear")
            total_grid_masked = np.ma.array(total_grid, mask=mask,
                                            fill_value=np.nan).filled()
            total_grid_county = np.ma.array(total_grid, mask=county_mask,
                                            fill_value=np.nan).filled()

            with h5py.File(os.path.join("gridded", county_folder + ".hdf5"), "a") as d:
                    year_group = d.create_group(f"{years[yidx]}")
                    year_group.create_dataset("x_grid", data=xgrid)
                    year_group.create_dataset("y_grid", data=ygrid)
                
                    year_group.create_dataset("mask", data=mask)
                    year_group.create_dataset("county_mask", data=county_mask)
                    # white
                    year_group.create_dataset("white_grid", data=white_grid)
                    year_group.create_dataset("white_grid_masked", data=white_grid_masked)
                    year_group.create_dataset("white_grid_county", data=white_grid_county)
                    # black
                    year_group.create_dataset("black_grid", data=black_grid)
                    year_group.create_dataset("black_grid_masked", data=black_grid_masked)
                    year_group.create_dataset("black_grid_county", data=black_grid_county)
                    # aapi
                    year_group.create_dataset("aapi_grid", data=aapi_grid)
                    year_group.create_dataset("aapi_grid_masked", data=aapi_grid_masked)
                    year_group.create_dataset("aapi_grid_county", data=aapi_grid_county)
                    # hispanic
                    year_group.create_dataset("hispanic_grid", data=hispanic_grid)
                    year_group.create_dataset("hispanic_grid_masked", data=hispanic_grid_masked)
                    year_group.create_dataset("hispanic_grid_county", data=hispanic_grid_county)
                    # total
                    year_group.create_dataset("total_grid", data=total_grid)
                    year_group.create_dataset("total_grid_masked", data=total_grid_masked)
                    year_group.create_dataset("total_grid_county", data=total_grid_county)
    except Exception as e:
        print(e)
        print(f'Skipping {sc.state, sc.county}')
                
