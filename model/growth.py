import numpy as np

def logistic_function(L, t, k, t0):

    return L / (1 + np.exp(-k *(t - t0)))

def logistic_derivative_analytical(L,t,k,t0):
    
    fx = logistic_function(L,t,k,t0)
    
    return k * fx * (1 - fx / L)

def generate_well_schedule_logistic(years, L,
                                    growth_rate, 
                                    peak_year,
                                    injection_duration,
                                    candidate_locations,
                                    used_index):
    
    all_wells = []
    num_wells_drilled_annually = []
    
    for year in range(1, years + 1):
        
        wells = logistic_derivative_analytical(L,year,growth_rate,peak_year)
        
        
        num_to_drill = int(np.ceil(wells))
        num_wells_drilled_annually.append(num_to_drill)
        
        new_wells = pooled_well_generator(num_to_drill,
                                   candidate_locations,
                                   used_index)

        for loc in new_wells:
            well = {
                'x': loc[0],
                'y': loc[1],
                'start_year': year,
                'end_year': year + injection_duration - 1
            }
            all_wells.append(well)
         
    return all_wells,num_wells_drilled_annually

def solve_tm(L, k, t, y):
    
    return t + np.log((L / y) - 1) / k
    
def pooled_well_generator(n,candidate_locations,used_index):
    # selects well locations from a predefined pool then updates used_index
    # (to ensure those wells are not selected next time its called)
    
    wells = candidate_locations[used_index[0] : used_index[0] + n]
    if len(wells) < n:
        raise ValueError("not enough wells")
    used_index[0] += n
    return wells

