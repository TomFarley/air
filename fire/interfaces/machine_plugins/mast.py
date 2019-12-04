"""MAST specific plugin fuctions for FIRE.

"""
import numpy as np

# MAST coordinate formatter, includes sector number.
def get_machine_sector(phi, input_in_deg=True):
    if not input_in_deg:
        phi = np.rad2deg(phi)
    if phi < 0:
        phi = phi + 360

    # MAST has 12 sectors going clockwise from North
    # Phi coordinate goes anticlockwise starting at phi=0 along x axis (due East)
    n_sectors = 12
    sector_angle = 360/n_sectors
    sector = (3 - np.floor(phi / sector_angle)) % n_sectors
    if sector == 0:
        sector = 12
    return sector

def cartesian_to_toroidal(x, y):
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return r, phi

def format_coord(coord):
    x, y, z = coord[0], coord[1], coord[2]
    r, phi = cartesian_to_toroidal(x, y)

    sector = get_machine_sector(phi)

    formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(x, y, z)
    formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(r, z, phi)
    formatted_coord = formatted_coord + '\n Sector {:.0f}'.format(sector)

    return formatted_coord