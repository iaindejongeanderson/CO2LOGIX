#%% Import required packages

import geopandas as gpd
import numpy as np
import CoolProp.CoolProp as CP
import pandas as pd
from shapely.geometry import box

#%% Import CO2LOGIX modules

from model.pressure import run_model
from model.growth import solve_tm
from model.utils import estimate_frac_pres

params = {}

#%% Inputs to growth model

params['L'] = 880 # carrying capacity
params['growth_rates'] = [0.086,0.13,0.22] # growth rates to iterate over
params['scenarios'] = ['reference','high','very high'] # names for each scenario

#%% Inputs to pressure model

params['grid_size'] = 1000 # in metres

# fluid properties

params['h_grad'] = 10 # hydrostatic pressure gradient (MPa/km)
params['l_grad'] = 23 # lithostatic pressure gradient (MPa/km)
params['t_grad'] = 25 # temperature gradient (degC/km)
params['t_surf'] = 15 # surface or seabed temperature (degC)
params['c_w'] = 5E-10 # brine compressibility

# simulation properties
params['inj_rate'] = 1 # in Mt/year
params['model_years'] = 100
params['inj_years'] = 25 # per well
params['domain'] = 'open' # or closed
params['rc'] = 10000 # closure radius, only used if domain is closed

# %% Import geospatial data

# Import shapefile

S_gdf = gpd.read_file(r'Input_data\saline_aquifers.shp')

# Import aquifer properties

data = pd.read_csv(r'Input_data\input_table.csv')
S_gdf['phi'] = data['Porosity']
S_gdf['h'] = data['Gross thickness'] * data['Net-to-gross']
S_gdf['k_md'] = data['Permeability']
S_gdf['z'] = data['Depth']


#%% Calculate proportion of total wells to place in each aquifer outline (based on area)

params['xmin'], params['ymin'], params['xmax'], params['ymax'] = S_gdf.total_bounds

# make a meshgrid of given grid size
x_coords = np.arange(params['xmin'] + params['grid_size'] / 2.0, params['xmax'], params['grid_size'])
y_coords = np.arange(params['ymin'] + params['grid_size'] / 2.0, params['ymax'], params['grid_size'])
params['gx'], params['gy'] = np.meshgrid(x_coords, y_coords) 
S_gdf['area_m2'] = S_gdf.geometry.area
S_gdf['well_fraction'] = S_gdf['area_m2'] / S_gdf['area_m2'].sum()
del x_coords,y_coords

#%% Some precomputations

# convert permeability to m2

S_gdf['k'] = S_gdf['k_md'] * 9.869233e-16 # permeability

# estimate rock compressibility (Hall 1953 method)

S_gdf['c_r'] = ((1.782/S_gdf['phi']**0.438)*1E-6)*145.038*1E-6 

# calculate reference pressure and temperature

S_gdf['p_ref'] = params['h_grad'] * S_gdf['z']/1000 # reference pressure at aquifer depth (MPa)
S_gdf['t_ref'] = params['t_grad'] * S_gdf['z']/1000 + params['t_surf'] # reference temperature at aquifer depth (degC)

# fluid properties from CoolProp
    
S_gdf['u_w'] = CP.PropsSI('V', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'Water')
S_gdf['u_c'] = CP.PropsSI('V', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'CO2')
S_gdf['rho_c'] = CP.PropsSI('D', 'T', S_gdf['t_ref'].values + 273.15, 'P', S_gdf['p_ref'].values * 1e6, 'CO2')

# volumetric flow rate

S_gdf['Q'] = params['inj_rate'] * 1e9 / S_gdf['rho_c'] / 365 / 86400


# total compressibility

S_gdf['c_tot'] = S_gdf['c_r'] + S_gdf['phi'] * params['c_w']

# inputs to Nordbotten solution

S_gdf['gamma'] = S_gdf['u_c'] / S_gdf['u_w']
S_gdf['omega'] = ((S_gdf['u_c'] + S_gdf['u_w']) / (S_gdf['u_c'] - S_gdf['u_w']) * np.log(np.sqrt(S_gdf['gamma'])) - 1.0)
S_gdf['D'] = S_gdf['k']/(S_gdf['c_tot']*S_gdf['u_w'])
S_gdf['Q'] = params['inj_rate'] * 1e9 / S_gdf['rho_c'] / 365 / 86400
S_gdf['p_c'] = (S_gdf['Q'] * S_gdf['u_w']) / (2.0 * np.pi * S_gdf['h'] * S_gdf['k']) / 1e6

# fracture pressure and pressure limit
S_gdf['sv'] = params['l_grad'] * (S_gdf['z'] / 1000)
S_gdf['p_frac'] = estimate_frac_pres(S_gdf['sv'], S_gdf['p_ref'])


#%% Make a grid of p_frac and p_ref - for normalising results at end

x_flat = params['gx'].ravel()
y_flat = params['gy'].ravel()

half_size = params['grid_size'] / 2

# Create polygons centered at (x, y)
polygons = [
    box(x - half_size, y - half_size, x + half_size, y + half_size)
    for x, y in zip(x_flat, y_flat)
]

# Create a GeoDataFrame of grid cells
grid_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=S_gdf.crs)

# Perform the spatial join
pfrac_join = gpd.sjoin(grid_gdf, S_gdf[['geometry', 'p_frac']], how='left', predicate='intersects')
pref_join = gpd.sjoin(grid_gdf, S_gdf[['geometry', 'p_ref']], how='left', predicate='intersects')

# Drop duplicates so each cell appears only once
pfrac_join = pfrac_join.drop_duplicates(subset='geometry')
pref_join = pref_join.drop_duplicates(subset='geometry')
p_frac_flat = pfrac_join['p_frac'].values
p_ref_flat = pref_join['p_ref'].values

params['p_frac_grid'] = p_frac_flat.reshape(params['gx'].shape)
params['p_ref_grid'] = p_ref_flat.reshape(params['gx'].shape)

del x_flat,y_flat,half_size,polygons,grid_gdf

#%% Run model, iterating over growth scenario

for k_growth,name in zip(params['growth_rates'],params['scenarios']):
    
    params['k_growth'] = k_growth
    

    t = 1
    y = 2
    
    params['t0'] = solve_tm(params['L'], 
                            params['k_growth'],
                            t, y)
    
    
    iteration = 1
    max_iterations = 101
    results = []
    
    
    while iteration < max_iterations:
        
        print(f'Iteration: {iteration}')
       
        
        all_dP_grids, max_dP_frac_pc, wells, capacity = run_model(S_gdf,params)
        
        results.append({
            "max_dP": max_dP_frac_pc,
            "capacity": capacity,
            "wells": wells
        })
        
        iteration += 1
        
    max_df_out = pd.DataFrame([r['max_dP'] for r in results]).T
    capacity = pd.DataFrame([r['capacity'] for r in results]).T
    wells = pd.DataFrame([r['wells'] for r in results]).T
            
    dfs = {
        "dP max": max_df_out,
        "capacity": capacity,
        "wells": wells
    }     
        
    with pd.ExcelWriter(f"output/output_{name}.xlsx", engine="openpyxl") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)   
        
 


