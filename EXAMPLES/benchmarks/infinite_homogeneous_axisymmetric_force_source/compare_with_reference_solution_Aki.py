#!/usr/bin/env python
#
# this script compares the numerical solution with an analytical solution for displacement in a 3D elastic medium
# due to a point force.
#
# the solution is taken from:
# Aki & Richards (2009), Quantitative Seismology
# chapter 4.2: Solution for the Elastodynamic Green Function in
#              a Homogeneous, Isotropic, Unbounded Medium
#
# equation (4.23): displacement solution for point force X_0(t) in x_j direction

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

######################################################################
## User Parameters

## setup (to match Par_file setting)

# positioning
# source
xs = 0.0
zs = 2500.0

# receiver
xr = 2500.0
zr = 1500.0

# source frequency
f0 = 3.0

# velocity model
rho = 2500.0
vp = 3400.0
vs = 1963.0

# time stepping
dt = 0.6e-3
nt = 4500

# numerical solution
rec_xfile = "OUTPUT_FILES/AA.S0003.BXX.semd"
rec_zfile = "OUTPUT_FILES/AA.S0003.BXZ.semd"

# analytical solution w/ near-field term
add_nearfield = True

# source time function type (0 == Gaussian normalized; 1 == Gaussian SPECFEM2D; 2 == Ricker SPECFEM2D)
specfem2d_stf = 2

######################################################################

# constants
pi = np.pi

def delta(i, j):
    if i==j:
      return 1.0
    else:
      return 0.0

def gauss_specfem2d(t, f0):
    # SPECFEM2D
    # note: the default Gaussian source time function (time_function_type == 3) in SPECFEM2D
    #       is given by routine comp_source_time_function_Gaussian() in comp_source_time_function.f90
    #       It is defined as the second integral of a Ricker wavelet to be able to apply the USE_TRICK_FOR_BETTER_PRESSURE
    #       feature for pressure outputs.
    #       The second integral of the Ricker expression then has a minus sign in front, thus the Gaussian goes "downwards"
    #       instead of the more intuitive "upward", positive Gaussian function.
    #
    #       This needs to be considered when using another Gaussian function, like the normal distribution as second option
    #       here below.
    a = pi**2 * f0**2
    comp_source_time_function_Gaussian = - np.exp(-a * t**2) / (2.0 * a)
    return comp_source_time_function_Gaussian

def gauss_normal(t, tau):
    # Gaussian normal distribution (with mean 0 and standard deviation tau)
    #
    # Here, we define the function to be positive, i.e. gauss_normal(t, tau) >= 0.0
    # To match the negative sign of the above SPECFEM2D version of the Gaussian, we could flip the sign for this normal Gaussian,
    # or we can flip the force vector from positive +Z to negative -Z direction. In this script, we choose the latter.
    # when evaluating the near-field terms, which require integration of this function, the sign needs to be considered carefully.
    return np.exp(-0.5 * t**2 / tau**2) / (tau * np.sqrt(2.0 * pi))

def ricker_specfem2d(t, f0):
    # SPECFEM2D
    # Ricker wavelet (second derivative of a Gaussian)
    a = pi**2 * f0**2
    comp_source_time_function_d2Gaussian = (1.0 - 2.0 * a * t**2) * np.exp(-a * t**2)
    return comp_source_time_function_d2Gaussian

def compute_stf(t_arr, tau, f0):
    if specfem2d_stf == 0:
        # normal gaussian distribution
        stf = gauss_normal(t_arr,tau)
    elif specfem2d_stf == 1:
        # specfem2d gaussian
        stf = gauss_specfem2d(t_arr,f0)
    elif specfem2d_stf == 2:
        # specfem2d ricker
        stf = ricker_specfem2d(t_arr,f0)
    else:
        print(f"Error: Invalid source time function type {specfem2d_stf} - not implemented yet")
        sys.exit(1)
    return stf

def get_radiation_pattern(gamma, comp, field):
    # gamma = dict({'x': rx, 'y': ry, 'z': rz})
    # comp = 'xx' , 'xy', ..
    # field = 'FP', 'FS', ..
    if len(comp) == 2:
      n = comp[0]
      p = comp[1]

      if field == 'FP':
        return gamma[n] * gamma[p]
      elif field == 'FS':
        return delta(n, p) - gamma[n] * gamma[p]
      elif field  == 'NF':
        return 3.0 * gamma[n] * gamma[p] - delta(n, p)
      else:
        return None

    elif len(comp) == 3:
      n = comp[0]
      p = comp[1]
      q = comp[2]

      if field == 'FP':
        return gamma[n] * gamma[p] * gamma[q]
      elif field == 'FS':
        return (delta(n, p) - gamma[n] * gamma[p]) * gamma[q]
      elif field == 'MP':
        return 6.0 * gamma[n] * gamma[p] * gamma[q] \
               - gamma[n] * delta(p, q) \
               - gamma[p] * delta(n, q) \
               - gamma[q] * delta(n, p)
      elif field == 'MS':
        return - 6.0 * gamma[n] * gamma[p] * gamma[q] \
               + gamma[n] * delta(p, q) \
               + gamma[p] * delta(n, q) \
               + 2.0 * gamma[q] * delta(n, p)
      elif field == 'NF':
        return  15.0 * gamma[n] * gamma[p] * gamma[q] \
               - 3.0 * gamma[n] * delta(p, q) \
               - 3.0 * gamma[p] * delta(n, q) \
               - 3.0 * gamma[q] * delta(n, p)
      else:
        return None

    else:
      return None


def get_fullspace_solution_time_domain(x, y, z,
                                       fx, fy, fz,
                                       la, mu, rho,
                                       t_arr, f0):
    # Aki & Richards (2009), Quantitative Seismology
    # chapter 4.2: Solution for the Elastodynamic Green Function in
    #              a Homogeneous, Isotropic, Unbounded Medium
    #
    # equation (4.23): displacement solution for point force X_0(t) in x_j direction

    # direction cosines gamma_i = x_i / r
    r = np.sqrt(x**2+y**2+z**2)
    rx = x/r
    ry = y/r
    rz = z/r

    gamma = dict({'x': rx, 'y': ry, 'z': rz})

    vp = np.sqrt((la + mu * 2.0) / rho)
    vs = np.sqrt(mu / rho)

    # amplitude factors in the Aki & Richards solution
    amp_p = 1.0 / (4.0 * pi * (la + mu * 2.0))
    amp_s = 1.0 / (4.0 * pi * mu)
    amp_nf = 1.0 / (4.0 * pi * rho)

    # source function amplification factor
    if specfem2d_stf == 0:
        # normalized Gaussian
        print(f"  source time function: Gaussian normal distribution")
        tau = 1.0 / (pi * f0 * np.sqrt(2.0))    # -> f0 = 1.0 / (pi * tau * np.sqrt(2.0)
        amp = np.sqrt(2.0 * pi) * tau           # == 1 / (np.sqrt(pi) * f0)
        # multiply by 1 / (2.0 * pi * pi * f0 * f0) to make the source time function same as SPECFEM2D
        amp *= 1.0 / (2.0 * pi*pi * f0*f0)
        print(f"  source time function amplitude correction: amp = {amp}")

    elif specfem2d_stf == 1:
        # SPECFEM2D Gaussian
        print(f"  source time function: Gaussian (SPECFEM2D)")
        tau = None
        amp = 1.0                               # no amplification needed, same source time function as SPECFEM2D

    elif specfem2d_stf == 2:
        # SPECFEM2D Ricker
        print(f"  source time function: Ricker (SPECFEM2D)")
        tau = None
        amp = 1.0                               # no amplification needed, same source time function as SPECFEM2D

    else:
        print(f"Error: Invalid source time function type {specfem2d_stf} - not implemented yet")
        sys.exit(1)
    
    print("")

    # amplification
    amp_p *= amp
    amp_s *= amp
    amp_nf *= amp

    # travel times
    tp = r / vp     # r / alpha
    ts = r / vs     # r / beta

    # source time function values X_0(t - r/alpha),..
    spec_p = compute_stf(t_arr-tp, tau, f0)
    spec_s = compute_stf(t_arr-ts, tau, f0)

    fx_x = amp_p/r * spec_p * get_radiation_pattern(gamma, 'xx', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'xx', 'FS')
    fy_x = amp_p/r * spec_p * get_radiation_pattern(gamma, 'xy', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'xy', 'FS')
    fz_x = amp_p/r * spec_p * get_radiation_pattern(gamma, 'xz', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'xz', 'FS')

    fx_y = amp_p/r * spec_p * get_radiation_pattern(gamma, 'yx', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'yx', 'FS')
    fy_y = amp_p/r * spec_p * get_radiation_pattern(gamma, 'yy', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'yy', 'FS')
    fz_y = amp_p/r * spec_p * get_radiation_pattern(gamma, 'yz', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'yz', 'FS')

    fx_z = amp_p/r * spec_p * get_radiation_pattern(gamma, 'zx', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'zx', 'FS')
    fy_z = amp_p/r * spec_p * get_radiation_pattern(gamma, 'zy', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'zy', 'FS')
    fz_z = amp_p/r * spec_p * get_radiation_pattern(gamma, 'zz', 'FP') + \
           amp_s/r * spec_s * get_radiation_pattern(gamma, 'zz', 'FS')

    # include near-field
    if add_nearfield:
        # integrated function value int_{r/alpha}^{r/beta} tau * X_0(t - tau) dtau
        #                            = [erf(r/beta) - erf(r/alpha)] - [gauss(t-r/beta) - gauss(t-r/alpha)] for X_0() = gauss()
        if specfem2d_stf == 0:
            # normalized Gaussian
            fac = pi * f0
            term1 = t_arr / 2.0 * (erf((ts - t_arr) * fac) - erf((tp - t_arr) * fac))
            term2 = - tau * tau * (gauss_normal(t_arr - ts, tau) - gauss_normal(t_arr - tp, tau))
            # total contribution
            spec_nf = term1 + term2

        elif specfem2d_stf == 1:
            # SPECFEM2D Gaussian
            # The integral required is \int_{t_p}^{t_s} \tau X_0(t-\tau) d\tau
            # Using substitution s = t - \tau, the integral becomes \int_{t-t_s}^{t-t_p} (t-s) X_0(s) ds
            # where X_0(s) = exp(-a0 s**2) / (2 * a0).
            # This integral resolves to:
            # (t * sqrt(pi) / (4 * a0**1.5)) * (erf(sqrt(a0)*(t-t_p)) - erf(sqrt(a0)*(t-t_s)))
            # + (1 / (4 * a0**2)) * (exp(-a0*(t-t_p)**2) - exp(-a0*(t-t_s)**2))
            a = pi**2 * f0**2
            a_sqrt = pi * f0

            # due to the minus sign, - exp(-a0 s**2), we can flip the sign of these terms,
            # or use \int_{t-t_p}^{t-t_s} instead of \int_{t-t_s}^{t-t_p}
            term1 = t_arr * np.sqrt(pi) / (4.0 * a*a_sqrt) * (erf((t_arr - ts) * a_sqrt) - erf((t_arr - tp) * a_sqrt))
            term2 = (1.0 / (4 * a**2)) * (np.exp(-a * (t_arr - ts)**2) - np.exp(-a * (t_arr - tp)**2))
            # or with gauss_specfem2d(t, f0) == - np.exp(-a * t**2) / (2.0 * a)
            #term2 = 1.0 / (2.0 * a) * (gauss_specfem2d(t_arr - tp, f0) - gauss_specfem2d(t_arr - ts, f0))

            # total contribution
            spec_nf = term1 + term2

        elif specfem2d_stf == 2:
            # SPECFEM2D Ricker
            # The integral required is \int_{r/alpha}^{r/beta} tau X_0(t-tau) dtau
            # Using substitution s = t - tau (ds = -dtau), the integral becomes
            #   - \int_{t-r/alpha}^{t-r/beta} (t-s) X_0(s) ds = - t \int X_0(s) ds + \int s X_0(s) ds
            # where X_0(s) = (1 - 2 * a * s**2)*exp(-a * s**2) is the Ricker wavelet
            # and a = pi**2 * f0**2
            a = pi**2 * f0**2
            sqrt_a = pi * f0

            # Integration limits int_{r/alpha}^{r/beta} tau X0(t - tau)
            # after substitution s = t - τ
            sA = t_arr - tp     # t - r/alpha
            sB = t_arr - ts     # t - r/beta

            # term1: - t \int X_0(s) ds = - t * s * exp(- a * s**2) with  s = t - tau and tau = r/alpha, r/beta
            #        and integration \int_{t-r/alpha}^{t-r/beta} .. ds = [ F(s) ]_{t-r/alpha}^{t-r/beta}
            #                                                          = [ F(s) ]_{sA}^{sB} = F(sB) - F(sA)
            #                                                          with F(s) = - t * s * exp(-a * s**2)
            term1 = - t_arr * sB * np.exp(-a * sB**2) \
                    + t_arr * sA * np.exp(-a * sA**2)

            # term2: \int s X_0(s) ds = + 1/(2*a) * (1 + 2 * a * s**2) exp(-a * s**2)
            term2 =  1.0 / (2.0 * a) * (1.0 + 2.0 * a * sB**2) * np.exp(-a * sB**2) \
                   - 1.0 / (2.0 * a) * (1.0 + 2.0 * a * sA**2) * np.exp(-a * sA**2)

            # total contribution
            spec_nf = term1 + term2

        else:
            print(f"Error: Invalid source time function type {specfem2d_stf} - not implemented yet")
            sys.exit(1)

        fx_x += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'xx', 'NF')
        fy_x += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'xy', 'NF')
        fz_x += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'xz', 'NF')

        fx_y += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'yx', 'NF')
        fy_y += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'yy', 'NF')
        fz_y += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'yz', 'NF')

        fx_z += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'zx', 'NF')
        fy_z += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'zy', 'NF')
        fz_z += amp_nf/r**3 * spec_nf * get_radiation_pattern(gamma, 'zz', 'NF')

    npts = len(t_arr)
    u = np.zeros(shape=(3,npts),dtype=float)

    u[0,:] = fx_x * fx + fy_x * fy + fz_x * fz
    u[1,:] = fx_y * fx + fy_y * fy + fz_y * fz
    u[2,:] = fx_z * fx + fy_z * fy + fz_z * fz

    return u

def compare_solutions():
    """
    compares numerical solution to analytical solution
    """
    print("comparing solutions...")
    print("")

    ## numerical solution
    print("numerical solution:")

    for comp in [ 'x', 'z']:
        if comp == 'x':
            filename = rec_xfile
        elif comp == 'z':
            filename = rec_zfile

        print(f"  receiver file: {filename}\n")

        # load trace
        A = np.loadtxt(filename)

        t_num = A[:,0]    # time
        trace_num = A[:,1] # x/z-trace

        print(f"  time  : min/max = {t_num.min()} / {t_num.max()}")
        print(f"  values: min/max = {trace_num.min()} / {trace_num.max()}")
        print("")

        # array index
        if comp == 'x':
          icomp = 0
        elif comp == 'z':
          icomp = 2

        # create numerical solution array
        if icomp == 0:
            npts = len(t_num)
            seis_num = np.zeros(shape=(3,npts),dtype=float)

        # store trace
        seis_num[icomp,:] = trace_num.copy()

    ## analytical solution
    print("analytical solution:")
    print(f"  near field : {add_nearfield}")

    # lambda/mu parameters
    la2mu = rho * vp * vp
    mu = rho * vs * vs
    la = la2mu - 2.0 * mu
    # debug check velocity conversions
    #vp_check = np.sqrt((la + mu * 2.0) / rho)
    #vs_check = np.sqrt(mu / rho)
    #print(f"  vp, vs (check): {vp_check} / {vs_check}")
    print(f"  vp, vs     : {vp} / {vs}")
    print(f"  lambda, mu : {la} / {mu}")
    print("")

    # Gaussian defined by exp(-0.5*(t/tau)**2)
    print(f"  frequency  : {f0}")

    # vector r
    r_x = xr - xs
    r_y = 0.0
    r_z = zr - zs

    # force vector
    if specfem2d_stf == 0:
        # normalized Gaussian
        stf_name = "normalized Gaussian"
        # uses Gaussian normal distribution (which is a positive function, i.e., gauss_normal(t,tau) >= 0.0).
        # to be able to match the SPECFEM2D solution, we have to deal with the negative Gaussian source time function used
        # by the code, thus we can either define the normal distribution with a negative sign as well,
        # or equivalently flip the force vector direction from +Z to -Z direction.
        force_x = 0.0
        force_y = 0.0
        force_z = -1.0
    elif specfem2d_stf == 1:
        # SPECFEM2D Gaussian
        stf_name = "SPECFEM2D Gaussian"
        # uses Gaussian from SPECFEM2D (which is negative defined, i.e., gauss_specfem2D(t, f0) <= 0.0).
        # we consider a force vector in positive Z-direction to compute the analytical solution.
        force_x = 0.0
        force_y = 0.0
        force_z = 1.0
    elif specfem2d_stf == 2:
        # SPECFEM2D Ricker
        stf_name = "SPECFEM2D Ricker"
        force_x = 0.0
        force_y = 0.0
        force_z = 1.0
    else:
        print(f"Error: Invalid source time function type {specfem2d_stf} - not implemented yet")
        sys.exit(1)

    # time for analytical solution
    t_ref = t_num
    # or define explicitly
    # t0 = 1.2 / f0
    # t_ref = np.arange(nt) * dt - t0

    # gets analytical solution
    u_analytical = get_fullspace_solution_time_domain(r_x, r_y, r_z,
                                                      force_x, force_y, force_z,
                                                      la, mu, rho,
                                                      t_ref, f0)

    # file output
    header_info = f"""# created by script: ./compare_with_reference_solution_Aki.py
# rx, ry, rz  = {r_x} {r_y} {r_z}
# rho, Vp, Vs = {rho} {vp} {vs}
# f0          = {f0}
# STF type    = {specfem2d_stf} - {stf_name}
# dt, nt      = {dt} {nt}
# format:
#time  #displacement
"""

    # compares X/Z-components
    for comp in ['x','z']:
        print(f"comparing {comp}-component to analytical solution...")
        # array index
        if comp == 'x':
          icomp = 0
        elif comp == 'z':
          icomp = 2

        trace_num = seis_num[icomp,:]      # takes x/z-component from numerical solution
        trace_ref = u_analytical[icomp,:]  # takes x/z-component from analytical solution

        print(f"  time  : min/max = {t_ref.min()} / {t_ref.max()}")
        print(f"  values: min/max = {trace_ref.min()} / {trace_ref.max()}\n")

        # file output
        filename = f"REF_ANALYTIC/U{comp}_analytical.dat"
        with open(filename, 'w') as f:
            # header
            f.write(header_info)
            # data
            for t, val in zip(t_ref, trace_ref):
                f.write(f'{t:.6e} {val:.6e}\n')
        print(f"  written to: {filename}")
        print("")

        # plotting comparison w/ solution trace
        print(f"  plotting {comp}-component...\n")

        # plotting
        plt.figure(figsize=(18,10))
        plt.plot(t_num, trace_num, 'r', label='SPECFEM2D solution')
        plt.plot(t_ref, trace_ref, 'k--', linewidth=0.8, label='analytical solution')
        plt.title(f"Displacement - {comp}")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # main routine
    compare_solutions()
