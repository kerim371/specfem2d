#!/usr/bin/env python
#
# This program implements the analytical solution for the displacement vector in a 2D plane-strain elastic medium
# with a vertical force source located in (0,0),
# from Appendix B of Carcione et al., Wave propagation simulation in a linear viscoelastic medium, GJI, vol. 95, p. 597-611 (1988),
# modified to compute the solution in the time domain instead of in the frequency domain
# (and thus for the elastic case only, instead of for the viscoelastic case).
# The amplitude of the force is called F and is defined below.
#
# This is a python version of the original Fortran code `analytical_solution_elastic_only_2D_plane_strain_Pilant_in_the_time_domain.f90`.
# It adds the possibility to convolve with a Ricker wavelet, as in the original version, and also with a Gaussian source time function.
# A corresponding parameter `time_function_type` can be set in the Parameter section of this file.
#
# Original Fortran program written by Dimitri Komatitsch, CNRS, Marseille, France, May 2018.
import sys,os
import time

import numpy as np
import matplotlib.pyplot as plt

###########################################################################
# Parameters
# To see how small the contribution of the near-field term is,
# here the user can ask not to include it, to then compare with the full result obtained with this flag set to False
DO_NOT_COMPUTE_THE_NEAR_FIELD = False

# For time-domain calculation
# Number of time steps used to discretize the time interval
# Use a very high value here, because the convolution algorithm that is implemented is very basic
# and thus needs a tiny time interval in order to be accurate
nt = 50000    # e.g. 100000
Tmax = 1.0    # duration in seconds

# Density of the medium
rho = 2000.0

# Unrelaxed (f = +infinity) values
# These values for the unrelaxed state are computed from the relaxed state values (Vp = 3000, Vs = 2000, rho = 2000)
# given in Carcione et al. 1988 GJI vol 95 p 604 Table 1
Vp = 3297.849
Vs = 2222.536

# Amplitude of the force source
F = 1.0

# Definition source parameters
f0 = 18.0
t0 = 1.2 / f0

# relative position of the receiver to source (x1 = xr - xs / x2 = zr - zs)
x1 = 500.0
x2 = 500.0

# source time function
# 1 == Ricker, 3 == Gaussian
time_function_type = 1

###########################################################################

# Duration of a time step
deltat = Tmax / (nt - 1)

# constants
pi = np.pi

def ricker_wavelet(t, f0):
    """Ricker wavelet function"""
    # Ricker wavelet (second derivative of a Gaussian)
    # note: same definition as in file src/specfem2D/comp_source_time_function.f90
    #       for routine comp_source_time_function_Ricker() (and comp_source_time_function_d2Gaussian())
    a = pi**2 * f0**2
    return (1.0 - 2.0 * a * t**2) * np.exp(-a * t**2)

def gaussian_wavelet(t, f0):
    """Gaussian wavelet function"""
    # Gaussian wavelet i.e. second integral of a Ricker wavelet
    # note: same definition as in file src/specfem2D/comp_source_time_function.f90
    #       for routine comp_source_time_function_Gaussian()
    a = pi**2 * f0**2
    return -np.exp(-a * t**2) / (2.0 * a)

def heaviside(t):
    """Heaviside step function"""
    if hasattr(t, '__len__'):  # array input
        return np.where(t > 0.0, 1, 0)
    else:  # scalar input
        return 1 if t > 0.0 else 0

def G1(r, t, v1, v2, do_not_compute_near_field=False):
    """
    Green's function G1 from equation (B2a) of Carcione et al., Wave propagation simulation
    in a linear viscoelastic medium, Geophysical Journal, vol. 95, p. 597-611 (1988)
    """
    tau1 = r / v1
    tau2 = r / v2

    heaviside_tau1 = heaviside(t - tau1)
    heaviside_tau2 = heaviside(t - tau2)

    G1_val = 0.0

    if heaviside_tau1 == 1:
        G1_val += 1.0 / (v1**2 * np.sqrt(t**2 - tau1**2))

    if not do_not_compute_near_field:
        if heaviside_tau1 == 1:
            G1_val += np.sqrt(t**2 - tau1**2) / (r**2)

        if heaviside_tau2 == 1:
            G1_val -= np.sqrt(t**2 - tau2**2) / (r**2)

    return G1_val

def G2(r, t, v1, v2, do_not_compute_near_field=False):
    """
    Green's function G2 from equation (B2a) of Carcione et al., Wave propagation simulation
    in a linear viscoelastic medium, Geophysical Journal, vol. 95, p. 597-611 (1988)
    """
    tau1 = r / v1
    tau2 = r / v2

    heaviside_tau1 = heaviside(t - tau1)
    heaviside_tau2 = heaviside(t - tau2)

    G2_val = 0.0

    if heaviside_tau2 == 1:
        G2_val -= 1.0 / (v2**2 * np.sqrt(t**2 - tau2**2))

    if not do_not_compute_near_field:
        if heaviside_tau1 == 1:
            G2_val += np.sqrt(t**2 - tau1**2) / (r**2)

        if heaviside_tau2 == 1:
            G2_val -= np.sqrt(t**2 - tau2**2) / (r**2)

    return G2_val

def u1(t, v1, v2, x1, x2, rho, F, do_not_compute_near_field=False):
    """Displacement in x1 direction"""
    # Source-receiver distance
    r = np.sqrt(x1**2 + x2**2)

    return F * x1 * x2 * (G1(r, t, v1, v2, do_not_compute_near_field) +
                          G2(r, t, v1, v2, do_not_compute_near_field)) / (2.0 * pi * rho * r**2)

def u2(t, v1, v2, x1, x2, rho, F, do_not_compute_near_field=False):
    """Displacement in x2 direction"""
    # Source-receiver distance
    r = np.sqrt(x1**2 + x2**2)

    return F * (x2*x2*G1(r, t, v1, v2, do_not_compute_near_field) -
                x1*x1*G2(r, t, v1, v2, do_not_compute_near_field)) / (2.0 * pi * rho * r**2)


def convolve_with_stf(Green_u):
    print(f"convolving with source time function...")
    if time_function_type == 1:
        print(f"  time function type : {time_function_type} - Ricker wavelet\n")
    elif time_function_type == 3:
        print(f"  time function type : {time_function_type} - Gaussian wavelet\n")
    else:
        print(f"source time function not implemented yet for type {time_function_type}")
        sys.exit(1)

    start_time = time.time()

    # Prepare output arrays
    time_output = []
    convolution_output = []

    # To avoid writing a huge file, since we purposely used a huge number of time steps,
    # write only every 10 time steps
    for it in range(1, nt + 1, 10):  # 1-based indexing like Fortran
        time_val = (it - 1) * deltat - t0

        if (it - 1) % 5000 == 0:
            print(f'  computing {it} out of {nt}')

        # start at time zero
        if time_val < 0.0: continue

        convolution_value = 0.0

        #for j in range(it - nt, it):
        #    if j < 1:  # Skip invalid indices (Fortran arrays start at 1)
        #        continue
        #    tau_j = (j - 1) * deltat  # Convert to 0-based time indexing
        #
        #    # Convolve with a Ricker wavelet
        #    ricker = ricker_wavelet(tau_j, f0)
        #    convolution_value += Green_ux[it - j] * ricker * deltat
        #
        #if time >= 0.0:
        #    time_output_ux.append(time)
        #    convolution_output_ux.append(convolution_value)

        tau = []
        for j in range(it - nt - 1, it):
            tau_j = j * deltat
            tau.append(tau_j)
        tau = np.array(tau)

        # source time function
        if time_function_type == 1:
            # Convolve with a Ricker wavelet
            stf = ricker_wavelet(tau, f0)
        elif time_function_type == 3:
            # Convolve with a Gaussian
            stf = gaussian_wavelet(tau, f0)
        else:
            print(f"source time function not implemented yet for type {time_function_type}")
            sys.exit(1)

        #print("tau",tau)
        #print("stf",stf)

        # convolution
        if 1 == 0:
            # simple, explicit convolution
            # elapsed time: 276.60 (s)
            for i,stf_val in enumerate(stf):
                # i -> j
                j = it - nt + i

                # Skip invalid indices (Fortran arrays start at 1)
                if it - j < 1: continue

                # indexing: it - j == it - (it - nt + i)
                #                  == nt - i
                #convolution_value += Green_u[it - j] * stf_val * deltat
                convolution_value += Green_u[it - j] * stf_val
            convolution_value *= deltat

        if 1 == 1:
            # using numpy dot function
            # elapsed time: 75.58 (s)
            # Flip source time function for convolution
            convolution_value = np.dot(Green_u, stf[::-1])
            convolution_value *= deltat

        if 1 == 0:
            # numpy convolution
            # elapsed time: 75.74 (s)
            # Flip source time function for convolution
            convolution = np.correlate(Green_u, stf[::-1],mode='valid') * deltat
            #print("convolution: ",convolution.shape)
            convolution_value = convolution[0]

        # store values
        if time_val >= 0.0:
            time_output.append(time_val)
            convolution_output.append(convolution_value)


        # Flip source time function for convolution
        #convolution = np.correlate(Green_u, stf[::-1], mode='same') * deltat
        #print("convolution: ",convolution.shape)

        # Create output with decimation (every 10th point like original)
        #decimate = 10
        #time_output = time_array[::decimate].tolist()
        #convolution_output = convolution[::decimate].tolist()

    end_time = time.time()
    print(f"  elapsed time: {end_time - start_time:.2f} (s)")
    print("")

    return time_output,convolution_output


def main():
    # user info
    print("-------------------------------------------------------------------------")
    print("analytical solution for displacement in 2D plane-strain elastic medium:")
    print("-------------------------------------------------------------------------")
    print(f"  Force   : located at (0,0) - pointing in vertical direction")
    print(f"  Receiver: located at (x,z) = ({x1}, {x2})")
    print("")

    # source time function
    if time_function_type == 1:
        print(f"  source time function: type = {time_function_type} - Ricker wavelet")
    elif time_function_type == 3:
        print(f"  source time function: type = {time_function_type} - Gaussian wavelet")
    else:
        print(f"Invalid source time function type {time_function_type}, it is not implemented yet.")
        print("Please modify the Parameter setting in this file...")
        sys.exit(1)

    # near field term
    if DO_NOT_COMPUTE_THE_NEAR_FIELD:
        print("  far-field solution only (rather than the full Green function)")
    else:
        print("  full solution (including the near-field term of the Green function)")
    print("")

    # **********
    # Compute Ux
    # **********

    print("computing Ux component...")

    # Store the Green function (using 1-based indexing equivalent)
    Green_ux = np.zeros(nt + 1)  # Extra element to handle 1-based indexing
    for it in range(1, nt + 1):  # 1-based indexing like Fortran
        time = (it - 1) * deltat - t0
        Green_ux[it] = u1(time, Vp, Vs, x1, x2, rho, F, DO_NOT_COMPUTE_THE_NEAR_FIELD)

    # convolve with source time function
    time_output_ux, convolution_output_ux = convolve_with_stf(Green_ux)

    # Save results for Ux
    if DO_NOT_COMPUTE_THE_NEAR_FIELD:
        filename_ux = 'analytical_solution_elastic_Carcione_without_near_field_Ux.dat'
    else:
        filename_ux = 'analytical_solution_elastic_Carcione_with_near_field_Ux.dat'

    header_info = f"""# created by script: ./analytical_solution_elastic_Carcione.py
# x1, x2      = {x1} {x2}
# rho, Vp, Vs = {rho} {Vp} {Vs}
# f0          = {f0}
# STF type    = {time_function_type}
# nt          = {nt}
"""

    with open(filename_ux, 'w') as f:
        # header
        f.write(header_info)
        f.write("# format:\n")
        f.write("#time  #Ux-displacement\n")
        # data
        for t, val in zip(time_output_ux, convolution_output_ux):
            f.write(f'{t:.6e} {val:.6e}\n')

    print(f"written to: {filename_ux}")
    print("")

    # **********
    # Compute Uz
    # **********

    print("computing Uz component...")

    # Store the Green function (using 1-based indexing equivalent)
    Green_uz = np.zeros(nt + 1)  # Extra element to handle 1-based indexing
    for it in range(1, nt + 1):  # 1-based indexing like Fortran
        time = (it - 1) * deltat - t0
        Green_uz[it] = u2(time, Vp, Vs, x1, x2, rho, F, DO_NOT_COMPUTE_THE_NEAR_FIELD)

    # convolve with source time function
    time_output_uz, convolution_output_uz = convolve_with_stf(Green_uz)

    # Save results for Uz
    if DO_NOT_COMPUTE_THE_NEAR_FIELD:
        filename_uz = 'analytical_solution_elastic_Carcione_without_near_field_Uz.dat'
    else:
        filename_uz = 'analytical_solution_elastic_Carcione_with_near_field_Uz.dat'

    with open(filename_uz, 'w') as f:
        # header
        f.write(header_info)
        f.write("# format:\n")
        f.write("#time  #Uz-displacement\n")
        # data
        for t, val in zip(time_output_uz, convolution_output_uz):
            f.write(f'{t:.6e} {val:.6e}\n')

    print(f"written to: {filename_uz}")
    print("")

    # plotting
    if '--show' in sys.argv:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(time_output_ux, convolution_output_ux, 'b-', linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Ux displacement')
        plt.title('Horizontal displacement component (Ux)')
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(time_output_uz, convolution_output_uz, 'r-', linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Uz displacement')
        plt.title('Vertical displacement component (Uz)')
        plt.grid(True)

        plt.tight_layout()

        if DO_NOT_COMPUTE_THE_NEAR_FIELD:
            filename = 'analytical_solution_elastic_Carcione_without_near_field_displacement.png'
        else:
            filename = 'analytical_solution_elastic_Carcione_with_near_field_displacement.png'

        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'plotted to: {filename}')

        plt.show()


if __name__ == '__main__':
    # source time function arguments override default setting
    if '--Ricker' in sys.argv:
        time_function_type = 1
    if '--Gaussian' in sys.argv:
        time_function_type = 3

    # main routine
    main()
