import numpy as np

def get_physical_tile(coords):
    '''
    Get the VTM physical tile name for a given point in 3D space.
    
    Parrameters:
        coords (np.array) : 3 element numpy array of [X,Y,Z] in metres.
    
    Returns:
        string or None: The name of the VTM physical tile, if the coordinates \
        are on one, otherwise None.
    '''    
    name = None
    component = 'Unknown'
    
    # Put input cartesian coords in to cylindrical
    R = np.sqrt(coords[0]**2 + coords[1]**2)
    Z = coords[2]
    phi = np.arctan2(coords[1],coords[0])
    if phi < 0.:
        phi = phi + 2*3.14159
    
    # Which component / tile are we on?
    # ------------- Divertor ---------------
    if R > 2.144 and R < 2.302 and Z > -1.328 and Z < -1.274:
        component = 'divertor'
        tile = 0
    elif R > 2.2294 and R < 2.414 and Z > -1.505 and Z < -1.229:
        component = 'divertor'
        tile = 1
    elif R > 2.390 and R < 2.440 and Z > -1.69 and Z < -1.511:
        component = 'divertor'
        tile = 3
    elif R > 2.314 and R < 2.524 and Z < -1.700:
        component = 'divertor'
        tile = 4
    elif R > 2.5739 and R < 2.633 and Z > -1.707 and Z < -1.585:
        component = 'divertor'
        tile = '5A'
    elif R > 2.633 and R < 2.693 and Z > -1.707 and Z < -1.585:
        component = 'divertor'
        tile = '5B'
    elif R > 2.693 and R < 2.752 and Z > -1.707 and Z < -1.585:
        component = 'divertor'
        tile = '5C'
    elif R > 2.752 and R < 2.814 and Z > -1.667 and Z < -1.585:
        component = 'divertor'
        tile = '5D'
    elif R > 2.813 and R < 2.988 and Z < -1.705:
        component = 'divertor'
        tile = 6
    elif R > 2.879 and R < 2.901 and Z > -1.683 and Z < -1.506:
        component = 'divertor'
        tile = 7
    elif R > 2.885 and R < 3.012 and Z > -1.499 and Z < -1.329:
        component = 'divertor'
        tile = 8
    elif R > 3.058 and R < 3.199 and Z > -1.298 and Z < -1.206:
        component = 'divertor'
        tile = 9
    elif R > 3.198 and R < 3.313 and Z > -1.219 and Z < -1.203:
        component = 'divertor'
        tile = 10
    # ---------------------- Not Divertor -----------------------
    elif R > 3.803 and R < 4.067 and phi > 0.5491 and phi < 0.558:
        component = 'REION8'
    elif R > 3.803 and R < 4.067 and phi > 3.6903 and phi < 3.7025:
        component = 'REION4'
    elif R > 3.265 and R < 3.94 and Z > -1.171 and Z < 1.355:
        component = 'PL'
    elif R < 1.967 and Z > -0.626 and Z < 1.249:
        # Note this is also where the inner wall protection tiles get lumped in.
        component = 'IWGL'
    elif R > 2.195 and R < 3.010 and Z > 1.82 and Z < 2.03:
        component = 'UDP'
    elif Z > 1.321 and Z < 1.82 and R > (0.4573*Z + 1.2766) and R < 2.181:
        component = 'UIWP'
    
    # Generate the tile name
    if component == 'divertor':
    
        # Work out what divertor module we're on and how far we are round it
        module_number = 12*phi/3.14159 - 7./2.
        if module_number < 1:
            module_number = module_number + 24
        module_fraction,module_number = np.modf(module_number)
        module_number = int(module_number)
        
        if tile in [3,7]:
            tiles = ['A','B']
        elif tile in ['5A','5B','5C','5D']:
            tiles = ['']
        else:
            tiles = ['A','B','C','D']
        
        name = 'DIV_M' + '{:02d}'.format(module_number) + 'T' + str(tile) + tiles[ int(np.floor(module_fraction * len(tiles)))]
    
    
    # Upper dump plates: the fact that the VTM splits them up in to
    # 5 things per octant does not really make sense, but we have to
    # conform to what the VTM thinks.
    elif component == 'UDP':
        
        octant = 4 * phi / 3.14159 + 7.5
        if octant >= 9.0:
            octant = octant - 8
            
        octant_fraction,octant = np.modf(octant)
        octant = int(octant)
        
        if octant_fraction < 0.13077:
            DP = 'A'
        elif octant_fraction < 0.36619:
            DP = 'B'
        elif octant_fraction < 0.6337:
            DP = 'C'
        elif octant_fraction < 0.86925:
            DP = 'D'
        else:
            DP = 'E'
        
        name = 'DP_' + '{:1d}'.format(octant) + DP
    
    # Outer limiters: check toroidal angle to see which limiter we're on.
    elif component == 'PL':
        beam = None
        
        if phi > 0.4532 and phi < 0.53776:
            beam = '8B'
            beamtype = 'wOPL'
        elif phi > 0.5313 and phi < 0.5491:
            beam = '8B'
            beamtype = 'wOPL_wing'
        elif phi > 1.0178 and phi < 1.119:
            beam = '8D'
            beamtype = 'wOPL'
        elif phi > 1.6313 and phi < 1.732:
            beam = '1D'
            beamtype = 'wOPL'
        elif phi > 1.955 and phi < 1.9712:
            beam = 'Oct1_2'
            beamtype = 'A2_SepTile'
        elif phi >  2.19 and phi < 2.245:
            beam = '2B-R'
            beamtype = 'PLTile'
        elif phi > 2.47 and phi < 2.515:
            beam = '2D-R'
            beamtype='PLTile'
        elif phi > 2.7411 and phi < 2.7572:
            beam = 'Oct2_3'
            beamtype = 'A2_SepTile'
        elif phi > 2.982 and phi < 3.029:
            beam = '3B'
            beamtype = 'nPL'
        elif phi > 3.039 and phi < 3.087:
            beam = 'R'
            beamtype = 'LHCD'
        elif phi > 3.087 and phi < 3.195:
            beam = 'T_B'
            beamtype = 'LHCD'
        elif phi > 3.195 and phi < 3.247:
            beam = 'L'
            beamtype = 'LHCD'
        elif phi > 3.603 and phi < 3.67929:
            beam = '4B'
            beamtype = 'wOPL'
        elif phi > 3.67929 and phi < 3.6947:
            beam = '4B'
            beamtype = 'wOPL_wing'
        elif phi > 4.1594 and phi < 4.2574:
            beam = '4D'
            beamtype = 'wOPL'
        elif phi > 4.773 and phi < 4.872:
            beam = '5D'
            beamtype = 'wOPL'
        elif phi > 5.0971 and phi < 5.11292:
            beam = 'Oct5_6'
            beamtype = 'A2_SepTile'
        elif phi > 5.335 and phi < 5.437:
            beam = '6B'
            beamtype = 'wOPL'
        elif phi > 5.558 and phi < 5.6585:
            beam = '6D'
            beamtype = 'wOPL'
        elif phi > 5.8827 and phi < 5.8983:
            beam = 'Oct6_7'
            beamtype = 'A2_SepTile'
        elif phi > 6.122 and phi < 6.227:
            beam = '7B'
            beamtype = 'wOPL'
        
        # Limiter tils.
        if beam is not None:
            '''
            # NOTE: This part of the code reflects reality, which is
            # that there are 16 tiles on these limiters.
            # HOWEVER, the VTM thinks there are 23, so we have
            # to conform to its version of reality rather than our own.
            if beamtype == 'PLTile':
                
                tile_z = np.array([-1.161,-1.083,-0.951,-0.809,-0.658,-0.499,-0.334,-0.165,0.009,0.186,0.364,0.541,0.716,0.888,1.056,1.217])
                tile = np.argwhere(Z > tile_z).max() + 1
                
                name = 'PLTile_' + beam + '{:02d}'.format(tile)
            '''
            if beamtype == 'LHCD':
                if beam == 'R':
                    z_bottom = -0.8432 * phi + 1.971
                    heights = [0.084, 0.155, 0.226, 0.299, 0.372, 0.446, 0.521, 0.595, 0.669, 0.744, 0.822, 0.897,0.972,1.049,1.125,1.198,1.272]
                    
                    z_edges = np.array([z_bottom] + list(z_bottom+np.array(heights)))
                    try:               
                        tile = np.argwhere(Z > z_edges).max() + 1
                    except:
                        tile = None
                            
                elif beam == 'L':
                    z_bottom = -0.9643 * phi + 2.481
                    heights = [0.063,0.132, 0.201,0.276,0.352,0.416,0.489,0.562,0.636,0.711,0.786,0.861,0.937,1.013,1.089,1.164,1.252]
                    
                    z_edges = np.array([z_bottom] + list(z_bottom+np.array(heights)))
                    try:
                        tile = np.argwhere(Z > z_edges).max() + 1
                    except:
                        tile = None
                        
                elif beam == 'T_B':
                    if Z > -0.612 and Z < -0.458:
                        beam = 'B'
                        if phi > 0.3085 and phi < 0.3110:
                            tile = 1
                        elif phi < 3.1323:
                            tile = 2
                        elif phi < 3.1545:
                            tile = 3
                        elif phi < 3.17688:
                            tile = 4
                        elif phi < 3.1973:
                            tile = 5
                        else:
                            tile = None
                    elif Z > 0.463 and Z < 0.628:
                        beam = 'T'
                        if phi > 3.087 and phi < 3.10828:
                            tile = 1
                        elif phi < 3.12947:
                            tile = 2
                        elif phi < 3.15066:
                            tile = 3
                        elif phi < 3.17785:
                            tile = 4
                        elif phi < 3.1941:
                            tile = 5
                        else:
                            tile = None
                    else:
                        tile = None
                
                if tile is not None:    
                    name = 'LH_' + beam + '{:02d}'.format(tile) 
                    
                    
            elif beamtype == 'A2_SepTile':
                
                if Z > -0.668 and Z < -0.393:
                    tile = 1
                elif Z > -0.668 and Z < -0.099:
                    tile = 2
                elif Z > -0.668 and Z < 0.205:
                    tile = 3
                elif Z > -0.668 and Z < 0.511:
                    tile = 4
                elif Z > -0.668 and Z < 0.816:
                    tile = 5
                else:
                    tile = None
                    
                if tile is not None:
                    name = beamtype + '_' + beam + '_Tile_' + str(tile)
                
            else:
                lim_centre_vec = np.array([R - 1.9168,Z-0.31])
                lim_centre_R = np.sqrt(np.sum(lim_centre_vec**2))
                if lim_centre_R > 2.05:
                    tile = None
                else:
                    lim_centre_phi = np.arctan(lim_centre_vec[1]/lim_centre_vec[0])
                    tile_boundary_phi = np.array([-0.9,-0.7499,-0.693,-0.6355, -0.5782, -0.5212, -0.4647, -0.4067, -0.3493, -0.2919, -0.2345, -0.1771, -0.1197, -0.0624, -0.005, 0.0524, 0.1098, 0.1672, 0.2247, 0.2819, 0.3392, 0.3966, 0.4539])
                    tile = np.argwhere(lim_centre_phi > tile_boundary_phi).max() + 1
                
                if tile is not None:
                    name = beamtype + '_' + beam +  '{:02d}'.format(tile)
                               
    # Inner limiters: check toroidal angle to see which limiter we're on.                             
    elif component == 'IWGL':
        
        lim_centre_vec = np.array([R - 4.826,Z-0.280])
        lim_centre_R = np.sqrt(np.sum(lim_centre_vec**2))
        if lim_centre_R > 3.085:
            return None
        elif phi > 0.198 and phi < 0.362:
            beam = '7Z'
            tiles = range(15,20)
        elif phi > 0.594 and phi < 0.757:
            beam = '8X'
            tiles = range(1,20)
        elif phi > 0.985 and phi < 1.149:
            beam = '8Z'
            tiles = range(1,20)
        elif phi > 1.378 and phi < 1.545:
            beam = '1X'
            tiles = range(15,20)
        elif phi > 1.773 and phi < 1.944:
            beam = '1Z'
            tiles = range(1,20)
        elif phi > 2.169 and phi < 2.331:
            beam = '2X'
            tiles = range(1,20)
        elif phi > 2.555 and phi < 2.717:
            beam = '2Z'
            tiles = range(16,20)
        elif phi > 2.956 and phi < 3.114:
            beam = '3X'
            tiles = range(1,20)
        elif phi > 3.345 and phi < 3.504:
            beam = '3Z'
            tiles = range(15,20)
        elif phi > 3.736 and phi < 3.898:
            beam = '4X'
            tiles = range(1,20)
        elif phi > 4.127 and phi < 4.288:
            beam = '4Z'
            tiles = range(1,20)
        elif phi > 4.517 and phi < 4.682:
            beam = '5X'
            tiles = range(15,20)
        elif phi > 4.915 and phi < 5.080:
            beam = '5Z'
            tiles = range(1,20)
        elif phi > 5.311 and phi < 5.475:
            beam = '6X'
            tiles = range(1,20)
        elif phi > 5.499 and phi < 5.864:
            beam = '6Z'
            tiles = range(16,20)
        elif phi > 6.096 and phi < 6.261:
            beam = '7X'
            tiles = range(1,20)
        else:
            beam = None
            
        if beam is None:
            component = 'Unknown'
            name = None
        else:
            tile_z = np.array([-0.677,-0.622,-0.521,-0.427,-0.329,-0.232,-0.135,-0.036,0.062,0.161,0.260,0.359,0.457,0.556,0.654,0.752,0.851,0.950,1.049,1.146])
            tile = np.argwhere(Z > tile_z).max()
            
            if tile in tiles:   
                name = 'IWGL_' + beam + '{:02d}'.format(tile)
            else:
                tile_z = np.array([-0.677,-0.570,-0.484,-0.386,-0.286,-0.186,-0.084,0.017,0.118,0.220,0.322,0.423,0.523,0.624,0.719])
                tile = np.argwhere(Z > tile_z).max() + 1
                name = 'IWGL_' + beam + '{:02d}'.format(tile)
                
    elif component in ['REION4','REION8']:
        if Z > -0.350 and Z < 0.015:
            region = 'L'
        elif Z > 0.015 and Z < 0.25:
            region = 'M'
        elif Z > 0.25 and Z < 0.617:
            region = 'U'
        else:
            region = None
            
        if region is not None:
            name = component + region
            
    elif component == 'UIWP':
        if Z > 1.603:
            t_b = 'T'
        else:
            t_b = 'B'
            
        octant = 4 * (phi + 0.1039) / 3.14159 + 7.5
        if octant >= 9.0:
            octant = octant - 8
            
        octant_fraction,octant = np.modf(octant)
        octant = int(octant)
        
        if octant_fraction < 0.25:
            n = 1
        elif octant_fraction < 0.5:
            n = 2
        elif octant_fraction < 0.625:
            n = 3
        elif octant_fraction < 0.75:
            n = 4
        else:
            n = 5
        name = 'IU{:d}{:d}{:s}'.format(octant,n,t_b)
        
        
                
    return name