from numpy import array as ary; from numpy import log as ln
from numpy import cos, sin, pi, sqrt, exp, arccos;
import numpy as np; tau = 2*pi
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.ndimage import convolve
from functools import reduce

"""
Solves the laplace equation in an n-D square grid geometry.
The following file's documentation will be written in the language of solving the heat diffusion equation's equilibrium temperature
in an N-D grid. But the program doesn't lose any generality 
"""

def nd_kernel(n):
    """
    creates the N-d kernel, which spreads the signal out in all cardinal directions.
    e.g. if n = 2 this will be 

    0 1 0
    1 0 1
    0 1 0
    """
    n = int(n)
    total_size = 3**n
    mid_point = int((3**n - 1)/2)
    kern = np.zeros(total_size, dtype=bool)
    for i in range(n):
        kern[mid_point-3**i] = True
        kern[mid_point+3**i] = True
    new_shape = 3*np.ones(n, dtype=int) 
    unnormed_kern = kern.reshape(new_shape)
    return unnormed_kern/unnormed_kern.sum()

class SolutionGrid(object):
    def __init__(self, boundary_values):
        """
        Parameters
        ----------
        boundary_values:an array of mixed datatype,
                        where np.nan denotes no boundary condition enfored at this point,
                        otherwise a float is used to denotes that this point must always have this much temperature.
        """
        self.solvable_values = np.isnan(boundary_values)
        self.fixed_values = ~solvable_values
        self.values = fixed_values

    def iterate(, speed=0.5):
        self.values += self.get_difference()*0.5

def initialize(boundary_values):
    """
    Interpolate in all straight line directions so that all values are found.
    boundary_values:an array of mixed datatype,
        where np.nan denotes no boundary condition enfored at this point,
        otherwise a float is used to denotes that this point must always have this much temperature.    
    """
    is_boundary_value = ~np.isnan(boundary_values)
    matrix_copy = boundary_values.copy()
    matrix_copy[~is_boundary_value] = np.nan

    initialization_success, valid_interps = [], []
    for n in range(boundary_values.ndim):
        bd_value_in_row = is_boundary_value.any(axis=n)
        len_n_axis = boundary_values.shape[n]
        has_boundary_interpable_value = np.tile(bd_value_in_row, [len_n_axis for ax_num in range(boundary_values.ndim) if ax_num==n else 1])
        initialization_success.append(has_boundary_interpable_value)

        interpolated_values = interpolate_matrix(matrix_copy.transpose(), bd_value_in_row).transpose # something something # fill nan in with
        valid_interps.append(interpolated_values)

    initialized_pos = reduce(np.logical_or, initialization_success)
    final_interpolated_values = np.nanmean(valid_interps)

    return final_interpolated_values, initialized_pos

def interpolate_matrix(matrix):
    """
    turn the entire matrix into a list of 1d-series,
        then perfom interpolation on these series if the bool value is true.
    then build them back up into a matrix.
    """

def spread_mask(mask):
    float_mask = ary(mask, dtype=float)
    kernel = nd_kernel(mask.ndim)
    return ary(convolve(float_mask, kernel)+float_mask, dtype=bool)

def calculate_dist_from_eqm(distance_from_eqm, variable_mask):
    """
    Calculates how far each one is from the equilibrium

    Parameters:
    -----------
    The matrix of 
    variable_mask: mask denoting where the variables are. Of the same shape as here.
    """


if __name__=='__main__':

if False:
    import matplotlib.animation as manimation
    writer = manimation.writers['ffmpeg'](fps=15, metadata={"title":"#animation", "comment":"#", "artist":"Ocean Wong"})

    grid_kws = {"width_ratios": (.9, .05), "wspace": .3}
    fig, (ax, cbar_ax) = plt.subplots(1,2, gridspec_kw=grid_kws)

    with writer.saving(fig, "Video_name.mp4", 300):
        for data in data_list:
            sns.heatmap(data, ax=ax, cbar_ax=cbar_ax)
            fig.suptitle("Stuff")
            ax.set_xlabel("")
            ax.set_ylabel("")

            writer.grab_frame()
            ax.clear()
            cbar_ax.clear()
    fig.clf()