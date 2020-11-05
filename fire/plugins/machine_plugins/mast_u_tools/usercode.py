import numpy as np

# MAST coordinate formatter, includes sector number.
def format_coord(coords):

    phi = np.arctan2(coords[1],coords[0])
    if phi < 0.:
        phi = phi + 2*3.14159
    phi = phi / 3.14159 * 180

    sector = (3 - np.floor(phi/30)) % 12
    if sector == 0:
        sector = 12
    
    formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(coords[0],coords[1],coords[2])
    formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(np.sqrt(coords[0]**2 + coords[1]**2),coords[2],phi)
    formatted_coord = formatted_coord + '\n Sector {:.0f}'.format(sector)

    return  formatted_coord