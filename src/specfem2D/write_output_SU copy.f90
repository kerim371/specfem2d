!========================================================================
!
!                            S P E C F E M 2 D
!                            -----------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                              CNRS, France
!                       and Princeton University, USA
!                 (there are currently many more authors!)
!                           (c) October 2017
!
! This software is a computer program whose purpose is to solve
! the two-dimensional viscoelastic anisotropic or poroelastic wave equation
! using a spectral-element method (SEM).
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 3 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
! The full text of the license is available in file "LICENSE".
!
!========================================================================

  ! subroutine write_output_SU(x_source,z_source,irec,buffer_binary,number_of_components,seismo_offset,seismo_current,seismotype_l)

  ! use specfem_par, only: NSTEP,nrec,DT,st_xval, &
  !                        P_SV,st_zval,NTSTEP_BETWEEN_OUTPUT_SAMPLE

  ! implicit none

  ! double precision,intent(in) :: x_source,z_source
  ! integer,intent(in) :: irec,number_of_components,seismo_offset,seismo_current,seismotype_l

  ! ! to write seismograms in single precision SEP and double precision binary
  ! double precision, dimension(seismo_current,nrec,number_of_components),intent(in) :: buffer_binary

  ! ! local parameters
  ! double precision :: sampling_deltat
  ! integer :: isample,ioffset
  ! integer, dimension(28) :: header1
  ! real(kind=4), dimension(30) :: header4
  ! integer(kind=2) :: header2(2),header3(2)
  ! real(kind=4), dimension(seismo_current) :: single_precision_seismo

  ! ! header
  ! header1(:) = 0
  ! header2(:) = 0
  ! header3(:) = 0
  ! header4(:) = 0

  ! ! write SU headers (refer to Seismic Unix for details)
  ! header1(1)  =  irec                          ! receiver ID
  ! header1(10) = NINT(st_xval(irec)-x_source)  ! offset
  ! header1(19) = NINT(x_source)                ! source location xs
  ! header1(20) = NINT(z_source)                ! source location zs
  ! header1(21) = NINT(st_xval(irec))           ! receiver location xr
  ! header1(22) = NINT(st_zval(irec))           ! receiver location zr

  ! if (nrec > 1) header4(18) = SNGL(st_xval(2)-st_xval(1)) ! receiver interval

  ! ! time steps
  ! header2(1) = 0  ! dummy
  ! if (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE < 32768) then
  !   header2(2) = int(NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE, kind=2)
  ! else
  !   print *,"!!! BEWARE !!! Two many samples for SU format ! The .su file created won't be usable"
  !   header2(2)=-9999
  ! endif

  ! ! time increment
  ! sampling_deltat = DT*NTSTEP_BETWEEN_OUTPUT_SAMPLE

  ! ! INTEGER(kind=2) values range from -32,768 to 32,767
  ! ! adapts time step info
  ! if (NINT(sampling_deltat*1.0d6) < 32768) then
  !   header3(1) = NINT(sampling_deltat*1.0d6, kind=2)  ! deltat (unit: 10^{-6} second)
  ! else if (NINT(sampling_deltat*1.0d3) < 32768) then
  !   header3(1) = NINT(sampling_deltat*1.0d3, kind=2)  ! deltat (unit: 10^{-3} second)
  ! else
  !   header3(1) = NINT(sampling_deltat, kind=2)  ! deltat (unit: 10^{0} second)
  ! endif
  ! header3(2) = 0  ! dummy

  ! ! first component trace
  ! ! samples trace
  ! do isample = 1,seismo_current
  !   single_precision_seismo(isample) = sngl(buffer_binary(isample,irec,1))
  ! enddo

  ! ! output
  ! if (seismo_offset == 0) then
  !   ioffset = 4 * ((irec-1) * (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60)) + 1
  !   write(12,pos=ioffset) header1,header2,header3,header4
  ! endif
  ! ioffset = 4 * ((irec-1) * (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60) + 60 + seismo_offset) + 1
  ! write(12,pos=ioffset) single_precision_seismo

  ! ! second component trace (not for pressure or membranes)
  ! if (seismotype_l /= 4 .and. seismotype_l /= 6 .and. P_SV) then
  !   ! samples trace
  !   do isample = 1,seismo_current
  !     single_precision_seismo(isample) = sngl(buffer_binary(isample,irec,2))
  !   enddo

  !   ! output
  !   if (seismo_offset == 0) then
  !     ioffset = 4 * ((irec-1) * (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60)) + 1
  !     write(14,pos=ioffset) header1,header2,header3,header4
  !   endif

  !   ioffset = 4 * ((irec-1)*(NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60) + 60 + seismo_offset) + 1
  !   write(14,pos=ioffset) single_precision_seismo
  ! endif

  ! end subroutine write_output_SU


! 4 bytes NSAMP and DT Seismic Unix format
  subroutine write_output_SU(x_source,z_source,irec,buffer_binary,number_of_components,seismo_offset,seismo_current,seismotype_l)

  use specfem_par, only: NSTEP,nrec,DT,st_xval, &
                         P_SV,st_zval,NTSTEP_BETWEEN_OUTPUT_SAMPLE

  implicit none

  double precision,intent(in) :: x_source,z_source
  integer,intent(in) :: irec,number_of_components,seismo_offset,seismo_current,seismotype_l

  ! to write seismograms in single precision SEP and double precision binary
  double precision, dimension(seismo_current,nrec,number_of_components),intent(in) :: buffer_binary

  ! local parameters
  double precision :: sampling_deltat
  integer :: isample,ioffset
  integer, dimension(26) :: header1    ! уменьшено с 28 до 26 → 104 байта
  integer :: ns_int4, dt_int4          ! 4+4 = 8 байт (вместо старых header2+header3 = 8 байт)
  real(kind=4), dimension(30) :: header4  ! 120 байт
  real(kind=4), dimension(seismo_current) :: single_precision_seismo

  ! header
  header1(:) = 0
  ns_int4 = 0
  dt_int4 = 0
  header4(:) = 0

  ! write SU headers (refer to Seismic Unix for details)
  header1(1)  =  irec                          ! receiver ID
  header1(10) = NINT(st_xval(irec)-x_source)  ! offset
  header1(19) = NINT(x_source)                ! source location xs
  header1(20) = NINT(z_source)                ! source location zs
  header1(21) = NINT(st_xval(irec))           ! receiver location xr
  header1(22) = NINT(st_zval(irec))           ! receiver location zr

  if (nrec > 1) header4(18) = SNGL(st_xval(2)-st_xval(1)) ! receiver interval

  ! time steps — теперь 4-байтовое целое
  ns_int4 = NSTEP / NTSTEP_BETWEEN_OUTPUT_SAMPLE
  ! если хочешь предупреждение о "слишком много отсчетов" — можно оставить, но int4 вмещает до ~2e9
  ! if (ns_int4 > huge(0_4)) ... — обычно не нужно

  ! time increment — теперь 4-байтовое целое в МИКРОСЕКУНДАХ (единый масштаб!)
  sampling_deltat = DT * NTSTEP_BETWEEN_OUTPUT_SAMPLE
  dt_int4 = NINT(sampling_deltat * 1.0d6)   ! всегда микросекунды, без адаптивного масштаба

  ! first component trace
  do isample = 1,seismo_current
    single_precision_seismo(isample) = sngl(buffer_binary(isample,irec,1))
  enddo

  ! output
  if (seismo_offset == 0) then
    ioffset = 4 * ((irec-1) * (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60)) + 1
    write(12,pos=ioffset) header1, ns_int4, dt_int4, header4
  endif
  ioffset = 4 * ((irec-1) * (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60) + 60 + seismo_offset) + 1
  write(12,pos=ioffset) single_precision_seismo

  ! second component trace (not for pressure or membranes)
  if (seismotype_l /= 4 .and. seismotype_l /= 6 .and. P_SV) then
    do isample = 1,seismo_current
      single_precision_seismo(isample) = sngl(buffer_binary(isample,irec,2))
    enddo

    if (seismo_offset == 0) then
      ioffset = 4 * ((irec-1) * (NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60)) + 1
      write(14,pos=ioffset) header1, ns_int4, dt_int4, header4
    endif

    ioffset = 4 * ((irec-1)*(NSTEP/NTSTEP_BETWEEN_OUTPUT_SAMPLE + 60) + 60 + seismo_offset) + 1
    write(14,pos=ioffset) single_precision_seismo
  endif

  end subroutine write_output_SU