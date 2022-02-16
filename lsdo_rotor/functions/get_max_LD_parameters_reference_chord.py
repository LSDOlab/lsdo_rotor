import numpy as np 

def get_max_LD_parameters_reference_chord(interp_object, reference_chord, reference_radius, Vx, RPM, altitude):
    
    # Trained airfoil surrogate model 
    interp      = interp_object
    
    # Computing tangential velocity of reference chord at reference radius 
    n           = RPM/60
    Vt          = n * 2 * np.pi * reference_radius

    # Computing Re# for reference chord 
    L           = 6.5
    R           = 287
    T0          = 288.16
    P0          = 101325
    g0          = 9.81
    mu0         = 1.735e-5
    S1          = 110.4
    h           = altitude * 1e-3
    T           = T0 - L * h
    P           = P0 * (T/T0)**(g0/L/R)
    rho         = P/R/T  
    mu          = mu0 * (T/T0)**(3/2) * (T0 + S1)/(T + S1) 
    W           = (Vx**2 + Vt**2)**0.5  
    Re          = rho * W * reference_chord / mu

    # Finding max L/D and corresponding AoA, Cl, Cd 
    alpha_range                     = np.linspace(-2*np.pi/180,10*np.pi/180,100)
    Re_alpha_design_space           = np.zeros((len(alpha_range),2))
    Re_alpha_design_space[:,0]      = alpha_range
    Re_alpha_design_space[:,1]      = Re/2e6

    Re_alpha_prediction             = interp.predict_values(Re_alpha_design_space)
    LD_design_space                 = Re_alpha_prediction[:,0] / Re_alpha_prediction[:,1]
    LD_max                          = np.max(LD_design_space)

    alpha_LD_max_index              = np.where(LD_design_space == LD_max)
    alpha_max_LD                    = alpha_range[alpha_LD_max_index]

    max_Re_alpha_combination        = np.array([alpha_max_LD , Re/2e6],dtype=object)
    max_Re_alpha_combination        = max_Re_alpha_combination.reshape((1,2))

    Cl_Cd_prediction                = interp.predict_values(max_Re_alpha_combination)

    Cl_max                          = Cl_Cd_prediction[0,0]
    Cd_min                          = Cl_Cd_prediction[0,1]

    
    return alpha_max_LD, Cl_max, Cd_min



