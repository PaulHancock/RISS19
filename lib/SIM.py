from __future__ import print_function, division



def position_gen(number):
    """
    A function to generate a number of random points in RA/DEC
    Input:  Number of points to generate
    Output: List of RA/DEC (2,1) array
    """

    return pos_arr

def region_gen(positions, region):
    """
    Takes in a list of positions and removes points outside the MIMAS region
    Input:  RA/DEC positions and MIMAS region file.
    Output: List of RA/DEC inside the correct region.
    """
    return region_arr


def flux_gen(fdl, region_pos, source_dist):
    """
    Function to distribute flux across all points
    Input:  Flux Density limit, RA/DEC positions, source distribution function
    Output: Flux for each RA/DEC point
    """
    return flux_arr

def soruce_type(region_pos, soruce_size):
    """
    Function to determine if a source is of type AGN or SFR
    Input:  RA/DEC list (source_size?)
    Output: AGN (1?) or SFR (0?)
    """
    return type_arr

