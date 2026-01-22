/*
!========================================================================
!
!                            S P E C F E M 2 D
!                            -----------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, April 2014
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
*/


__global__ void compute_kernels_hess_el_cudakernel(const int* ispec_is_elastic,
                                                   const int* ibool,
                                                   const realw* accel,
                                                   const realw* b_accel,
                                                   realw* hess_kl1,
                                                   realw* hess_kl2,
                                                   const int NSPEC_AB,
                                                   const realw dt_factor) {

  int ispec = blockIdx.x + blockIdx.y*gridDim.x;
  int ij = threadIdx.x;

  // handles case when there is 1 extra block (due to rectangular grid)
  if (ispec < NSPEC_AB) {

    // elastic elements only
    if (ispec_is_elastic[ispec]) {
      int iglob = ibool[ij + NGLL2_PADDED*ispec] - 1;

      realw acc_x = accel[2 * iglob];
      realw acc_z = accel[2 * iglob + 1];
      realw b_acc_x = b_accel[2 * iglob];
      realw b_acc_z = b_accel[2 * iglob + 1];

        // approximate hessian
      hess_kl1[ij + NGLL2*ispec] +=  b_acc_x * b_acc_x + b_acc_z * b_acc_z;
      hess_kl2[ij + NGLL2*ispec] +=  acc_x * b_acc_x + acc_z * b_acc_z;
      // hess_kl1[ij + NGLL2*ispec] +=  (accel[2*iglob]*b_accel[2*iglob] + accel[2*iglob+1]*b_accel[2*iglob+1]);
      // hess_kl2[ij + NGLL2*ispec] +=  (accel[2*iglob]*b_accel[2*iglob] + accel[2*iglob+1]*b_accel[2*iglob+1]);
    }
  }
}

/* ----------------------------------------------------------------------------------------------- */

__global__ void compute_kernels_hess_ac_cudakernel(const int* ispec_is_acoustic,
                                                   const int* d_ibool,
                                                   const realw* potential_acoustic,
                                                   const realw* b_potential_acoustic,
                                                   const realw* rhostore,
                                                   const realw* d_hprime_xx,
                                                   const realw* d_xix,
                                                   const realw* d_xiz,
                                                   const realw* d_gammax,
                                                   const realw* d_gammaz,
                                                   realw* hess_kl1,
                                                   realw* hess_kl2,
                                                   const int NSPEC_AB,
                                                   const realw dt_factor) {

  int ispec = blockIdx.x + blockIdx.y * gridDim.x;
  int ij = threadIdx.x;

  if (ispec >= NSPEC_AB) return;
  if (!ispec_is_acoustic[ispec]) return;

  int i = ij % NGLLX;
  int j = ij / NGLLX;

  // Вычисляем градиент потенциала (forward)
  realw tempx1l = 0.0f;
  realw tempx2l = 0.0f;
  for (int k = 0; k < NGLLX; k++) {
    int iglob_kj = d_ibool[k + j * NGLLX + NGLL2_PADDED * ispec] - 1; // (k, j)
    int iglob_ik = d_ibool[i + k * NGLLX + NGLL2_PADDED * ispec] - 1; // (i, k)

    tempx1l += potential_acoustic[iglob_kj] * d_hprime_xx[i + k * NGLLX];
    tempx2l += potential_acoustic[iglob_ik] * d_hprime_xx[j + k * NGLLX]; // hprime_zz == hprime_xx
  }

  // То же для adjoint
  realw b_tempx1l = 0.0f;
  realw b_tempx2l = 0.0f;
  for (int k = 0; k < NGLLX; k++) {
    int iglob_kj = d_ibool[k + j * NGLLX + NGLL2_PADDED * ispec] - 1;
    int iglob_ik = d_ibool[i + k * NGLLX + NGLL2_PADDED * ispec] - 1;

    b_tempx1l += b_potential_acoustic[iglob_kj] * d_hprime_xx[i + k * NGLLX];
    b_tempx2l += b_potential_acoustic[iglob_ik] * d_hprime_xx[j + k * NGLLX];
  }

  // Метрика и плотность
  int idx = ij + NGLL2 * ispec;
  realw rhol = rhostore[idx];
  realw xixl = d_xix[idx];
  realw xizl = d_xiz[idx];
  realw gammaxl = d_gammax[idx];
  realw gammazl = d_gammaz[idx];

  // Локальное "ускорение" (на самом деле смещение, но так в CPU)
  realw accel_loc_x = (tempx1l * xixl + tempx2l * gammaxl) / rhol;
  realw accel_loc_z = (tempx1l * xizl + tempx2l * gammazl) / rhol;

  realw b_accel_loc_x = (b_tempx1l * xixl + b_tempx2l * gammaxl) / rhol;
  realw b_accel_loc_z = (b_tempx1l * xizl + b_tempx2l * gammazl) / rhol;

  // Гессиан
  realw dot1 = accel_loc_x * accel_loc_x + accel_loc_z * accel_loc_z;
  realw dot2 = accel_loc_x * b_accel_loc_x + accel_loc_z * b_accel_loc_z;

  hess_kl1[idx] += dot1 * dt_factor;
  hess_kl2[idx] += dot2 * dt_factor;
}
