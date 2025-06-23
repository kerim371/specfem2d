!=====================================================================
!
!               S p e c f e m 3 D  V e r s i o n  3 . 0
!               ---------------------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, July 2012
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%%%%%%%%%%%% calls with hdur as an argument are below %%%%%%%%%%%%%%%%%%%%
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  double precision function comp_source_time_function_heaviside_hdur(t,hdur)

  implicit none

  double precision, intent(in) :: t,hdur

  double precision, external :: netlib_specfun_erf

  ! compared with calling these same functions below with f0,
  ! one has the relationship hdur = 1 / (PI * f0), or equivalently f0 = 1 / (PI * hdur)

  ! quasi Heaviside, small Gaussian moment-rate tensor with hdur
  comp_source_time_function_heaviside_hdur = 0.5d0 * (1.0d0 + netlib_specfun_erf(t/hdur))

  end function comp_source_time_function_heaviside_hdur



! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
! %%%%%%%%%%%%% calls with f0 as an argument are below %%%%%%%%%%%%%%%%%%%%
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  double precision function comp_source_time_function_Gaussian(t,f0)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: t,f0
  double precision :: a

  ! Gaussian wavelet i.e. second integral of a Ricker wavelet
  a = PI**2 * f0**2
  comp_source_time_function_Gaussian = - exp(-a * t**2) / (2.d0 * a)

  end function comp_source_time_function_Gaussian

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_dGaussian(t,f0)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: t,f0
  double precision :: a

  ! first integral of a Ricker wavelet
  a = PI**2 * f0**2
  comp_source_time_function_dGaussian = t * exp(-a * t**2)

  end function comp_source_time_function_dGaussian

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d2Gaussian(t,f0)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: t,f0
  double precision :: a

  ! Ricker wavelet (second derivative of a Gaussian)
  a = PI**2 * f0**2
  comp_source_time_function_d2Gaussian = (1.d0 - 2.d0 * a * t**2) * exp(-a * t**2)

  end function comp_source_time_function_d2Gaussian

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d3Gaussian(t,f0)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: t,f0
  double precision :: a

  ! first derivative of a Ricker wavelet (third derivative of a Gaussian)
  a = PI**2 * f0**2
  comp_source_time_function_d3Gaussian = 2.d0 * a * t * (- 3.d0 + 2.d0 * a * t**2) * exp(-a * t**2)

  end function comp_source_time_function_d3Gaussian

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d4Gaussian(t,f0)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: t,f0

  ! local variables
  double precision :: a

  ! second derivative of a Ricker wavelet
  a = PI**2 * f0**2
  comp_source_time_function_d4Gaussian = - 2.d0 * a * (3.d0 - 12.d0 * a * t*t + 4.d0 * a**2 * t*t*t*t) * exp(-a * t**2)

  end function comp_source_time_function_d4Gaussian

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_Ricker(t,f0)

! Ricker wavelet (second derivative of a Gaussian)

  implicit none

  double precision, intent(in) :: t,f0

  double precision, external :: comp_source_time_function_d2Gaussian

  ! Ricker wavelet
  comp_source_time_function_Ricker = comp_source_time_function_d2Gaussian(t,f0)

  !! another source time function that is improperly called 'Ricker' in some old papers,
  !! e.g., 'Finite-Frequency Kernels Based on Adjoint Methods' by Liu & Tromp, BSSA (2006), is:
  ! comp_source_time_function_Ricker = -2.d0*PI*PI*f0*f0*f0*t * exp(-PI*PI*f0*f0*t*t)

  end function comp_source_time_function_Ricker

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_dRicker(t,f0)

  implicit none

  double precision, intent(in) :: t,f0

  double precision, external :: comp_source_time_function_d3Gaussian

  ! first derivative of a Ricker wavelet
  comp_source_time_function_dRicker = comp_source_time_function_d3Gaussian(t,f0)

  end function comp_source_time_function_dRicker

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d2Ricker(t,f0)

  implicit none

  double precision, intent(in) :: t,f0

  double precision, external :: comp_source_time_function_d4Gaussian

  ! second derivative of a Ricker wavelet
  comp_source_time_function_d2Ricker = comp_source_time_function_d4Gaussian(t,f0)

  end function comp_source_time_function_d2Ricker


!------------------------------------------------------------
!
! specialized source time functions
!
!------------------------------------------------------------



  double precision function comp_source_time_function_ocean_I(t,f0)

! ocean acoustics type I

  use constants, only: PI,QUARTER,HALF,ONE,TWO
  implicit none

  double precision, intent(in) :: t,f0
  double precision :: Tc,omega_coa,omegat

  Tc = 4.d0 / f0

  omega_coa = TWO * PI * f0
  omegat = omega_coa * t

  if (t > 0.d0 .and. t < Tc) then
    ! source time function from Computational Ocean Acoustics
    comp_source_time_function_ocean_I = HALF * sin(omegat) * (ONE - cos(QUARTER * omegat))

    ! alternative
    !comp_source_time_function_ocean_I = HALF / omega_coa / omega_coa * &
    !      ( sin(omegat) - 8.d0 / 9.d0 * sin(3.d0/ 4.d0 * omegat) - 8.d0 / 25.d0 * sin(5.d0 / 4.d0 * omegat) )
  else
    comp_source_time_function_ocean_I = 0.d0
  endif

  end function comp_source_time_function_ocean_I

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_ocean_II(t,f0)

! ocean acoustics type II

  use constants, only: PI,ZERO,QUARTER,HALF,ONE,TWO
  implicit none

  double precision, intent(in) :: t,f0
  double precision :: Tc,omega_coa,omegat

  Tc = 4.d0 / f0

  omega_coa = TWO * PI * f0
  omegat = omega_coa * t

  if (t > 0.d0 .and. t < Tc) then
    ! source time function from Computational Ocean Acoustics
    comp_source_time_function_ocean_II = HALF / omega_coa / omega_coa * ( - sin(omegat) + 8.d0 / 9.d0 * sin(3.d0 / 4.d0 * omegat) &
                                           + 8.d0 / 25.d0 * sin(5.d0 / 4.d0 * omegat) - 1.d0 / 15.d0 * omegat )

    ! alternative
    !comp_source_time_function_ocean_II = HALF / omega_coa / omega_coa * ( sin(omegat) - 8.d0 / 9.d0 * sin(3.d0/ 4.d0 * omegat) &
    !                       - 8.d0 / 25.d0 * sin(5.d0 / 4.d0 * omegat) - 1.d0 / 15.d0 * t + 1.d0 / 15.d0 * Tc )
  else if (t > 0.d0) then
    comp_source_time_function_ocean_II = - HALF / omega_coa / 15.d0 * Tc
  else
    comp_source_time_function_ocean_II = ZERO
  endif

  ! alternative - For source 1 OASES
  !Tc = 1.d0 / f0
  !
  !if (t > 0.d0 .and. t < Tc) then
  !  comp_source_time_function_ocean_II = 0.75d0 - cos(omegat) + 0.25d0*cos(TWO * omegat)
  !else
  !  comp_source_time_function_ocean_II = ZERO
  !endif

  end function comp_source_time_function_ocean_II


!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d2ocean_II(t,f0)

! second derivative of ocean acoustics type II

  use constants, only: PI,ZERO,QUARTER,HALF,ONE,TWO
  implicit none

  double precision, intent(in) :: t,f0
  double precision :: Tc,omega_coa,omegat

  Tc = 4.d0 / f0

  omega_coa = TWO * PI * f0
  omegat = omega_coa * t

  if (t > 0.d0 .and. t < Tc) then
    ! second derivate of source time function from Computational Ocean Acoustics
    comp_source_time_function_d2ocean_II = HALF * (ONE - cos(omegat / 4.0d0)) * sin(omegat)
  else
    comp_source_time_function_d2ocean_II = ZERO
  endif

  end function comp_source_time_function_d2ocean_II

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_burst(t,f0,burst_band_width)

! burst type source time function

  use constants, only: PI,ZERO,ONE,TWO

  implicit none

  double precision,intent(in) :: t,f0,burst_band_width
  double precision :: Tc,Nc,omega_coa,omegat

  Nc = TWO * f0 / burst_band_width
  Tc = Nc / f0

  omega_coa = TWO * PI * f0
  omegat = omega_coa * t

  if (t > 0.d0 .and. t < Tc) then
    comp_source_time_function_burst = 0.5d0 * (ONE - cos(omegat / Nc)) * sin(omegat)
  else
    comp_source_time_function_burst = ZERO
  endif

  end function comp_source_time_function_burst

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d2burst(t,f0,burst_band_width)

! second derivative of burst type source time function

  use constants, only: PI,ZERO,ONE,TWO

  implicit none

  double precision,intent(in) :: t,f0,burst_band_width
  double precision :: Tc,Nc,omega_coa,omegat

  Nc = TWO * f0 / burst_band_width
  Tc = Nc / f0

  omega_coa = TWO * PI * f0
  omegat = omega_coa * t

  if (t > 0.d0 .and. t < Tc) then ! t_used > 0 t_used < Nc/f0) then
    comp_source_time_function_d2burst = 0.5d0 * (omega_coa)**2 * sin(omegat) * cos(omegat / Nc) / Nc**2 &
                                         - 0.5d0 * (omega_coa)**2 * sin(omegat) * (ONE - cos(omegat / Nc)) &
                                         + (omega_coa)**2 * cos(omegat) * sin(omegat / Nc) / Nc
  else
    comp_source_time_function_d2burst = ZERO
  endif

  ! alternative - Integral of burst
  !if (t > 0.d0 .and. t < Tc) then
  !  comp_source_time_function_d2burst = - ( Nc*( (Nc+1.0d0)*cos((omega_coa*(Nc-1.0d0)*t_used)/Nc) + &
  !                                          (Nc-1.0d0)*cos((omega_coa*(Nc+1.0d0)*t_used)/Nc)) - &
  !                                           TWO*(Nc**2-1.0d0)*cos(omega_coa*t_used) ) / (8.0d0*PI*f0*(Nc-1)*(Nc+1))
  !else
  !  stf = ZERO
  !endif

  ! Double integral of burst
  !if (t > 0.d0 .and. t < Tc) then
  !  comp_source_time_function_d2burst = - ( -sin(TWO*f0*Pi*t_used)/(8.0d0*f0**TWO*Pi**2) + &
  !                                          (Nc**2*sin((TWO*f0*(Nc-1)*PI*t_used)/Nc))/(16.0d0*f0**2*(Nc-1)**2*Pi**2) + &
  !                                          (Nc**2*sin((TWO*f0*(Nc+1)*PI*t_used)/Nc))/(16.0d0*f0**2*(Nc+1)**2*Pi**2) )
  !else
  !  stf = ZERO
  !endif

  end function comp_source_time_function_d2burst

!
!------------------------------------------------------------
!

  double precision function sinc(a)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: a

  if (abs(a) < 1.0d-10) then
    sinc = 1.0d0
  else
    sinc = sin(a)/a
  endif

  end function sinc

!
!------------------------------------------------------------
!

  double precision function cos_taper(a,hdur)

  use constants, only: PI

  implicit none

  double precision, intent(in) :: a,hdur

  double precision :: b

  b = abs(a)
  cos_taper = 0.d0

  if (b <= hdur) then
    cos_taper = 1.d0
  else if (b > hdur .and. b < 2.d0 * hdur) then
    cos_taper = cos(PI * 0.5d0 * (b-hdur)/hdur)
  endif

  end function cos_taper

!
!------------------------------------------------------------
!

  double precision function marmousi_ormsby_wavelet(a)

  use constants, only: PI

  implicit none

  double precision :: sinc

  double precision, intent(in) :: a

  double precision :: f1,f2,f3,f4,b,c,tmp

  ! 5-10-60-80 Hz Ormsby Wavelet for Marmousi2 Model (Gray S. Martin, 2006)
  ! Please find the Ormsby Wavelet here http://subsurfwiki.org/wiki/Ormsby_filter

  f1 = 5.0d0  ! low-cut frequency
  f2 = 10.0d0 ! low-pass frequency
  f3 = 60.0d0 ! high-pass frequency
  f4 = 80.0d0 ! high-cut frequency

  b = sinc( f1 * a )
  tmp = PI * f1
  tmp = tmp * tmp
  tmp = tmp / PI /( f2 - f1 )
  c = b * b * tmp

  b = sinc( f2 * a )
  tmp = PI * f2
  tmp = tmp * tmp
  tmp = tmp / PI /( f2 - f1 )
  c = c - b * b * tmp

  b = sinc( f3 * a )
  tmp = PI * f3
  tmp = tmp * tmp
  tmp = tmp / PI /( f4 - f3 )
  c = c - b * b * tmp

  b = sinc( f4 * a )
  tmp = PI * f4
  tmp = tmp * tmp
  tmp = tmp / PI /( f4 - f3 )
  c = c + b * b * tmp

  marmousi_ormsby_wavelet = c

  end function marmousi_ormsby_wavelet


!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_marmousi(t,f0)

! source time function with Marmousi Ormsby wavelet

  use constants, only: PI

  implicit none

  double precision,intent(in) :: t,f0
  double precision :: hdur
  double precision, external :: cos_taper, marmousi_ormsby_wavelet

  hdur = 1.0 / f0  ! hdur = 1.0 / 35.0

  comp_source_time_function_marmousi = cos_taper(t,hdur) * marmousi_ormsby_wavelet(PI * t)

  end function comp_source_time_function_marmousi
!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_mono(t,f0)

  ! monochromatic source time function

  use constants, only: PI,TAPER_MONOCHROMATIC_SOURCE

  implicit none

  double precision,intent(in) :: t,f0
  double precision :: tt,omega_coa,taper_val
  integer :: taper

  omega_coa = 2.d0 * PI * f0
  tt = omega_coa * t

  taper = ceiling(TAPER_MONOCHROMATIC_SOURCE * f0)
  if (t < taper / f0) then
    taper_val = 0.5d0 - 0.5d0 * cos(tt / taper / 2.d0)
  else
    taper_val = 1.d0
  endif

  comp_source_time_function_mono = sin(tt) * taper_val

  end function comp_source_time_function_mono

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_d2mono(t,f0)

  ! second derivative of monochromatic source time function

  use constants, only: PI,TAPER_MONOCHROMATIC_SOURCE

  implicit none

  double precision,intent(in) :: t,f0
  double precision :: tt,omega_coa,taper_val
  integer :: taper

  omega_coa = 2.d0 * PI * f0
  tt = omega_coa * t

  taper = ceiling(TAPER_MONOCHROMATIC_SOURCE * f0)
  if (t < taper / f0) then
    taper_val = 0.5d0 - 0.5d0 * cos(tt / taper / 2.d0)
  else
    taper_val = 1.d0
  endif

  comp_source_time_function_d2mono = - omega_coa*omega_coa * sin(tt) * taper_val

  end function comp_source_time_function_d2mono

!
!------------------------------------------------------------
!

  double precision function comp_source_time_function_ext(it_index,isource)

! reads from external source time function file

  use specfem_par, only: myrank, NSTEP, name_of_source_file

  implicit none

  integer,intent(in) :: it_index,isource

  double precision :: dummy_t,stf_val
  integer :: file_unit,ier
  character(len=250) :: error_msg
  character(len=150), parameter :: error_msg1 = 'Error opening the file that contains the external source: '

  ! file unit for external source time function files
  file_unit = 800 + isource

  ! opens external file to read in source time function
  if (it_index == 1) then
    ! reads in from external source time function file
    open(unit=file_unit,file=trim(name_of_source_file(isource)),status='old',action='read',iostat=ier)
    if (ier /= 0) then
      print *,'Error opening source time function file: ',trim(name_of_source_file(isource))
      error_msg = trim(error_msg1)//trim(name_of_source_file(isource))
      call exit_MPI(myrank,error_msg)
    endif
  endif

  ! reads in 2-column file values (time value in first column will be ignored)
  ! format: #time #stf-value
  read(file_unit,*,iostat=ier) dummy_t, stf_val
  if (ier /= 0) then
    print *,'Error reading source time function file: ',trim(name_of_source_file(isource)),' at line ',it_index
    print *,'Please make sure the file contains the same number of lines as the number of timesteps NSTEP ',NSTEP
    call exit_MPI(myrank,'Error reading source time function file')
  endif

  ! closes external file
  if (it_index == NSTEP) close(file_unit)

  comp_source_time_function_ext = stf_val

  end function comp_source_time_function_ext
