import numpy as np
from collections import defaultdict
from model.growth import generate_well_schedule_logistic
from model.utils import make_distance_matrix
from shapely.geometry import Point
import geopandas as gpd

def delta_p_nordbotten(r, psi, gamma, R, rc, p_char_mpa, domain):
    
    # Initialize output array with zeros
    PD = np.zeros_like(r, dtype=float)

    # Define masks for where to perform each calculation
    
    # Mask where distance matrix contains a number
    valid_r = ~np.isnan(r)
    
    # Mask for region inside plume radius (r <= psi)
    in_plume = valid_r & (r <= psi)
    

    # Open boundary calculation
    
    if domain=='open':
        
        # mask for outside plume but within R
        
        out_plume = valid_r & (r > psi) & (r <= R)
        
        # PD within CO2 plume
        
        PD[in_plume] = gamma * np.log(psi/r[in_plume]) + np.log(R / psi)
        
        # PD outside CO2 plume
        
        PD[out_plume] = np.log(R / r[out_plume])
        
    # Closed boundary calculation
        
    if domain == 'closed':
        
        # mask for outside plume but within rc
        
        out_plume = valid_r & (r > psi) & (r <= rc)
        
        # PD within CO2 plume
        
        PD[in_plume] = gamma * np.log(psi/r[in_plume]) + np.log(rc / psi) + ((2 * R**2)/(2.25 * rc**2)) - (3/4)
        
        # PD outside plume
        
        PD[out_plume] = np.log(rc / r[out_plume]) + ((2 * R**2)/(2.25 * rc**2)) - (3/4)
    
    # multiply by characteristic pressure
    
    PD = PD * p_char_mpa

    PD[np.isnan(r)] = np.nan

    return PD


def run_model(S_gdf,params):
  
    # Well locations - get all xy locations and make gdf
    
    grid_points = gpd.points_from_xy(x=params['gx'].ravel(), 
                                     y=params['gy'].ravel())
    
    points_gdf = gpd.GeoDataFrame(geometry=grid_points, crs=S_gdf.crs)

    # do a spatial join with S_gdf to get fraction of wells in each storage unit
    
    points_with_poly_info = gpd.sjoin(points_gdf, S_gdf[['well_fraction', 'geometry']], 
                                      how='inner', predicate='within')
    
    # randomly sample based on area weights
    
    sampled_points = points_with_poly_info.sample(n=params['L']+100,
                                                  weights=points_with_poly_info['well_fraction'],
                                                  replace=False,
                                                  random_state=42)
    
    # populate candidate  locations with xy coordinates 
    
    x_sampled = sampled_points.geometry.x.to_numpy()
    y_sampled = sampled_points.geometry.y.to_numpy()
    candidate_locations = np.column_stack((x_sampled, y_sampled))

    
    used_index = [0]
     
    # create list of wells using logistic model
    
    wells,wells_per_year = generate_well_schedule_logistic(params['model_years'],
                                                           params['L'],
                                                           params['k_growth'],
                                                           params['t0'],
                                                           params['inj_years'],
                                                           candidate_locations,
                                                           used_index
                                                           )
    # sample aquifer index, and pressure grids per well
    
    for i, w in enumerate(wells,1):
        coords = np.array([w['x'],w['y']])
        w['aquifer_idx'] = S_gdf.contains(Point(coords)).tolist().index(True)
        w['p_frac'] = S_gdf['p_frac'][w['aquifer_idx']]
        w['p_ref'] = S_gdf['p_ref'][w['aquifer_idx']]
    
    # reformat well list to a dictionary organised per-year
    
    active_by_year = defaultdict(list)
    
    all_years = range(min(w['start_year'] for w in wells), 
                      max(w['start_year'] for w in wells) + params['inj_years'] + 1)
    
    for year in all_years:
        for well in wells:
            if well['start_year'] <= year <= well['end_year']:
                active_by_year[year].append({
                    'x': well['x'],
                    'y': well['y'],
                    'start_year': well['start_year'],
                    'end_year': well['end_year'],
                    'injection_age': year - well['start_year'] + 1,
                    'active': True,
                    'aquifer_idx':well['aquifer_idx'],
                    'p_frac':well['p_frac'],
                    'p_ref':well['p_ref']
                    
                })
            elif year > well['end_year']:
                active_by_year[year].append({
                    'x': well['x'],
                    'y': well['y'],
                    'start_year': well['start_year'],
                    'end_year': well['end_year'],
                    'injection_age': well['end_year'] - well['start_year'] + 1,
                    'active': False,
                    'aquifer_idx':well['aquifer_idx'],
                    'p_frac':well['p_frac'],
                    'p_ref':well['p_ref']
                })

    # make arrays from the various storage unit specific properties
    
    D_arr    = S_gdf['D'].to_numpy()
    Q_arr   = S_gdf['Q'].to_numpy()
    phi_arr = S_gdf['phi'].to_numpy()
    h_arr    = S_gdf['h'].to_numpy()
    omega_arr = S_gdf['omega'].to_numpy()
    gamma_arr = S_gdf['gamma'].to_numpy()
    p_c_arr  = S_gdf['p_c'].to_numpy()
    
    
    all_dP_grids = []
    max_dP_frac_pc = []
    capacity = []
    
    ## Main calculation ##
    
    for year, wells_in_year in active_by_year.items():
        
        dP_out = np.zeros_like(params['gx'], dtype=float)
            
        xs = [w['x'] for w in wells_in_year]
        ys = [w['y'] for w in wells_in_year]
       
        aquifer_idxs = [w['aquifer_idx'] for w in wells_in_year]
        pressure_ages = []
        valid_inj = []
        
        # make a vector of pressure ages for calculation
        
        for w in wells_in_year:
            
            if w['active']:
                pressure_ages.append(w['injection_age'])
                valid_inj.append(True)
            else:
                pressure_ages.append(params['inj_years'])  # fallback
                valid_inj.append(False)

        # convert pressure ages to seconds
        
        t_s = np.array(pressure_ages) * 86400 * 365
        
        # Main dP calculation
        
        for i in range(len(xs)): # for every well in that timestep
        
            # sample properties using aquifer index
            
            D    = D_arr[aquifer_idxs[i]] # 
            Q    = Q_arr[aquifer_idxs[i]] 
            phi  = phi_arr[aquifer_idxs[i]] # porosity
            h = h_arr[aquifer_idxs[i]]  # thickness
            omega = omega_arr[aquifer_idxs[i]] 
            gamma = gamma_arr[aquifer_idxs[i]] 
            p_c = p_c_arr[aquifer_idxs[i]] # characteristic pressure
            
            # calculated parameters
            
            R = np.sqrt(2.25 * D * t_s[i]) # pressure influence radius
            csi = np.sqrt(Q * t_s[i] / (np.pi * phi * h))
            psi = np.exp(omega) * csi
            
            

            if params['domain'] == 'open':
                
                dist_local, mask_y, mask_x = make_distance_matrix(xs[i], ys[i], R, params['gx'], params['gy'])
            
            elif params['domain'] == 'closed':
                
                dist_local, mask_y, mask_x = make_distance_matrix(xs[i], ys[i], params['rc'], params['gx'], params['gy'])
            
            # single-well solution
            
            dP_local = delta_p_nordbotten(
                dist_local,
                psi,
                gamma,
                R,
                params['rc'],
                p_c,
                params['domain'],
                )
            
            # adding single-well solutions for this timestep
            
            dP_out[np.ix_(mask_y, mask_x)] += dP_local
                
            
        # normalisation to fracture pressure
            
        dP_out[dP_out == 0] = np.nan
        dP_frac_pc = ((dP_out+params['p_ref_grid'])/params['p_frac_grid'])*100
        
        # putting together export data
        
        all_dP_grids.append(dP_frac_pc)  
        max_dP_frac_pc.append(np.nanmax(dP_frac_pc))
        capacity.append(sum(valid_inj))
        
    
    return all_dP_grids, max_dP_frac_pc, wells_per_year, capacity