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


// __global__ void compute_kernels_hess_el_cudakernel(const int* ispec_is_elastic,
//                                                    const int* ibool,
//                                                    const realw* accel,
//                                                    const realw* b_accel,
//                                                    realw* hess_kl1,
//                                                    realw* hess_kl2,
//                                                    realw* hess_kl3,
//                                                    realw* hess_kl4,
//                                                    const int NSPEC_AB,
//                                                    const realw dt_factor) {

//   int ispec = blockIdx.x + blockIdx.y*gridDim.x;
//   int ij = threadIdx.x;

//   // handles case when there is 1 extra block (due to rectangular grid)
//   if (ispec < NSPEC_AB) {

//     // elastic elements only
//     if (ispec_is_elastic[ispec]) {
//       int iglob = ibool[ij + NGLL2_PADDED*ispec] - 1;

//       realw acc_x = accel[2 * iglob];
//       realw acc_z = accel[2 * iglob + 1];
//       realw b_acc_x = b_accel[2 * iglob];
//       realw b_acc_z = b_accel[2 * iglob + 1];

//         // approximate hessian
//       hess_kl1[ij + NGLL2*ispec] +=  b_acc_x * b_acc_x + b_acc_z * b_acc_z;
//       hess_kl2[ij + NGLL2*ispec] +=  fabs(acc_x * b_acc_x + acc_z * b_acc_z);
//       hess_kl3[ij + NGLL2*ispec] +=  (accel[2*iglob]*b_accel[2*iglob] + accel[2*iglob+1]*b_accel[2*iglob+1]);
//       hess_kl4[ij + NGLL2*ispec] +=  (accel[2*iglob]*b_accel[2*iglob] + accel[2*iglob+1]*b_accel[2*iglob+1]);
//     }
//   }
// }

// /* ----------------------------------------------------------------------------------------------- */

// __global__ void compute_kernels_hess_ac_cudakernel(const int* ispec_is_acoustic,
//                                                    const int* d_ibool,
//                                                    const realw* potential_acoustic,
//                                                    const realw* b_potential_acoustic,
//                                                    const realw* rhostore,
//                                                    const realw* d_hprime_xx,
//                                                    const realw* d_xix,
//                                                    const realw* d_xiz,
//                                                    const realw* d_gammax,
//                                                    const realw* d_gammaz,
//                                                    realw* hess_kl1,
//                                                    realw* hess_kl2,
//                                                    realw* hess_kl3,
//                                                    realw* hess_kl4,
//                                                    const int NSPEC_AB,
//                                                    const realw dt_factor) {

//   int ispec = blockIdx.x + blockIdx.y * gridDim.x;
//   int ij = threadIdx.x;

//   if (ispec >= NSPEC_AB) return;
//   if (!ispec_is_acoustic[ispec]) return;

//   int i = ij % NGLLX;
//   int j = ij / NGLLX;

//   // Вычисляем градиент потенциала (forward)
//   realw tempx1l = 0.0f;
//   realw tempx2l = 0.0f;
//   for (int k = 0; k < NGLLX; k++) {
//     int iglob_kj = d_ibool[k + j * NGLLX + NGLL2_PADDED * ispec] - 1; // (k, j)
//     int iglob_ik = d_ibool[i + k * NGLLX + NGLL2_PADDED * ispec] - 1; // (i, k)

//     tempx1l += potential_acoustic[iglob_kj] * d_hprime_xx[i + k * NGLLX];
//     tempx2l += potential_acoustic[iglob_ik] * d_hprime_xx[j + k * NGLLX]; // hprime_zz == hprime_xx
//   }

//   // То же для adjoint
//   realw b_tempx1l = 0.0f;
//   realw b_tempx2l = 0.0f;
//   for (int k = 0; k < NGLLX; k++) {
//     int iglob_kj = d_ibool[k + j * NGLLX + NGLL2_PADDED * ispec] - 1;
//     int iglob_ik = d_ibool[i + k * NGLLX + NGLL2_PADDED * ispec] - 1;

//     b_tempx1l += b_potential_acoustic[iglob_kj] * d_hprime_xx[i + k * NGLLX];
//     b_tempx2l += b_potential_acoustic[iglob_ik] * d_hprime_xx[j + k * NGLLX];
//   }

//   // Метрика и плотность
//   int idx = ij + NGLL2 * ispec;
//   realw rhol = rhostore[idx];
//   realw xixl = d_xix[idx];
//   realw xizl = d_xiz[idx];
//   realw gammaxl = d_gammax[idx];
//   realw gammazl = d_gammaz[idx];

//   // Локальное "ускорение" (на самом деле смещение, но так в CPU)
//   realw accel_loc_x = (tempx1l * xixl + tempx2l * gammaxl) / rhol;
//   realw accel_loc_z = (tempx1l * xizl + tempx2l * gammazl) / rhol;

//   realw b_accel_loc_x = (b_tempx1l * xixl + b_tempx2l * gammaxl) / rhol;
//   realw b_accel_loc_z = (b_tempx1l * xizl + b_tempx2l * gammazl) / rhol;

//   // Гессиан
//   realw dot1 = accel_loc_x * accel_loc_x + accel_loc_z * accel_loc_z;
//   realw dot2 = accel_loc_x * b_accel_loc_x + accel_loc_z * b_accel_loc_z;

//   hess_kl1[idx] += dot1 * dt_factor;
//   hess_kl2[idx] += fabs(dot2) * dt_factor;
//   hess_kl3[idx] += dot2 * dt_factor;
//   hess_kl4[idx] += dot2 * dt_factor;
// }


__global__ void compute_kernels_hess_el_cudakernel(const int* ispec_is_elastic,
                                                   const int* ibool,
                                                   const realw* accel,
                                                   const realw* b_accel,
                                                   const realw* veloc,
                                                   const realw* b_veloc,
                                                   const realw* d_xix,
                                                   const realw* d_xiz,
                                                   const realw* d_gammax,
                                                   const realw* d_gammaz,
                                                   const realw* d_hprime_xx,
                                                   const realw* rhostore,
                                                   const realw* rho_vp,
                                                   const realw* rho_vs,
                                                   realw* hess_kl1,
                                                   realw* hess_kl2,
                                                   realw* hess_kl3,
                                                   realw* hess_kl4,
                                                   const int NSPEC_AB,
                                                   const realw dt_factor) {

  int ispec = blockIdx.x + blockIdx.y*gridDim.x;
  int ij = threadIdx.x;

  // handles case when there is 1 extra block (due to rectangular grid)
  if (ispec < NSPEC_AB) {

    // elastic elements only
    if (ispec_is_elastic[ispec]) {
      int i = ij % NGLLX;
      int j = ij / NGLLX;
      
      // global index
      int iglob = ibool[ij + NGLL2_PADDED*ispec] - 1;
      int idx = ij + NGLL2*ispec;          // индекс для гессиана и параметров
      int idx_padded = ij + NGLL2_PADDED*ispec;  // индекс для метрических коэффициентов

      // P1 и P2 вычисляем как раньше
      realw acc_x = accel[2 * iglob];
      realw acc_z = accel[2 * iglob + 1];
      realw b_acc_x = b_accel[2 * iglob];
      realw b_acc_z = b_accel[2 * iglob + 1];

      // P1: Hρρ - только источник
      hess_kl1[idx] += b_acc_x * b_acc_x + b_acc_z * b_acc_z;
      
      // P2: Ĥρρ - источник и приемник
      realw dot_product = acc_x * b_acc_x + acc_z * b_acc_z;
      hess_kl2[idx] += fabs(dot_product);
      
      // Для P3 и P4 используем shared memory как в примере
      __shared__ realw field_veloc[2*NGLL2];
      
      // copy field values to shared memory (только прямое поле для P3)
      field_veloc[2*ij] = veloc[2 * iglob];
      field_veloc[2*ij+1] = veloc[2 * iglob + 1];
      
      // synchronize threads
      __syncthreads();
      
      // Вычисляем производные как в примере compute_kernels_cudakernel
      // derivative along xi
      realw vx_xi = 0.f;
      realw vz_xi = 0.f;
      
      // derivative along gamma
      realw vx_gamma = 0.f;
      realw vz_gamma = 0.f;
      
      for(int l=0; l<NGLLX; l++) {
        realw hp1 = d_hprime_xx[l*NGLLX + i];
        int offset1 = j*NGLLX + l;
        
        vx_xi += field_veloc[2*offset1] * hp1;
        vz_xi += field_veloc[2*offset1+1] * hp1;
      }
      
      for(int l=0; l<NGLLX; l++) {
        // assumes hprime_xx == hprime_zz
        realw hp3 = d_hprime_xx[l*NGLLX + j];
        int offset3 = l*NGLLX + i;
        
        vx_gamma += field_veloc[2*offset3] * hp3;
        vz_gamma += field_veloc[2*offset3+1] * hp3;
      }
      
      // Получаем метрические коэффициенты
      realw xixl = d_xix[idx_padded];
      realw xizl = d_xiz[idx_padded];
      realw gammaxl = d_gammax[idx_padded];
      realw gammazl = d_gammaz[idx_padded];
      
      // Производные скоростей по физическим координатам
      realw vx_x = xixl * vx_xi + gammaxl * vx_gamma;
      realw vx_z = xizl * vx_xi + gammazl * vx_gamma;
      realw vz_x = xixl * vz_xi + gammaxl * vz_gamma;
      realw vz_z = xizl * vz_xi + gammazl * vz_gamma;
      
      // Получаем параметры среды (используем idx, а не idx_padded!)
      realw rho = rhostore[idx];
      realw rho_alpha = rho_vp[idx];  // ρ * α
      realw rho_beta = rho_vs[idx];   // ρ * β
      
      // Вычисляем дивергенцию и другие величины
      realw div_v = vx_x + vz_z;           // ∂x vx + ∂z vz
      realw term1_beta = vx_x * vx_x + vz_z * vz_z;  // (∂x vx)² + (∂z vz)²
      realw term2_beta = vz_x + vx_z;      // ∂x vz + ∂z vx
      realw term2_beta_sq = term2_beta * term2_beta;
      
      // Отладочная информация: проверяем масштабы величин
      // Для отладки можно временно записать div_v в hess_kl3, чтобы посмотреть его значения
      // hess_kl3[idx] += div_v * div_v;  // без коэффициентов
      
      // P3: Hαα = 8ρ²α² ∫ (∂x vx + ∂z vz)² dt
      // Проблема может быть здесь: rho_alpha уже содержит ρ*α, а не α
      // rho2_alpha2 = (ρ*α)² = ρ²α² - это правильно
      realw rho2_alpha2 = rho_alpha * rho_alpha;
      realw p3_value = 8.0f * rho2_alpha2 * div_v * div_v;
      hess_kl3[idx] += fabs(p3_value);
      
      // P4: Hββ = 16ρ²β² ∫ [(∂x vx)² + (∂z vz)²] dt + 4ρ²β² ∫ (∂x vz + ∂z vx)² dt
      realw rho2_beta2 = rho_beta * rho_beta;
      realw p4_value = 16.0f * rho2_beta2 * term1_beta + 4.0f * rho2_beta2 * term2_beta_sq;
      hess_kl4[idx] += fabs(p4_value);
    }
  }
}

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
                                                   realw* hess_kl1,    // P1 для акустики
                                                   realw* hess_kl2,    // P2 для акустики  
                                                   realw* hess_kl3,    // P3 для акустики
                                                   realw* hess_kl4,    // P4 для акустики
                                                   const int NSPEC_AB,
                                                   const realw dt_factor) {

  int ispec = blockIdx.x + blockIdx.y * gridDim.x;
  int ij = threadIdx.x;

  if (ispec >= NSPEC_AB) return;
  if (!ispec_is_acoustic[ispec]) return;

  int i = ij % NGLLX;
  int j = ij / NGLLX;
  int idx = ij + NGLL2 * ispec;

  // 1. Вычисляем градиент потенциала для прямого поля
  realw tempx1l = 0.0f;
  realw tempx2l = 0.0f;
  for (int k = 0; k < NGLLX; k++) {
    int iglob_kj = d_ibool[k + j * NGLLX + NGLL2_PADDED * ispec] - 1;
    int iglob_ik = d_ibool[i + k * NGLLX + NGLL2_PADDED * ispec] - 1;

    tempx1l += potential_acoustic[iglob_kj] * d_hprime_xx[i + k * NGLLX];
    tempx2l += potential_acoustic[iglob_ik] * d_hprime_xx[j + k * NGLLX];
  }

  // 2. Вычисляем градиент потенциала для сопряженного поля
  realw b_tempx1l = 0.0f;
  realw b_tempx2l = 0.0f;
  for (int k = 0; k < NGLLX; k++) {
    int iglob_kj = d_ibool[k + j * NGLLX + NGLL2_PADDED * ispec] - 1;
    int iglob_ik = d_ibool[i + k * NGLLX + NGLL2_PADDED * ispec] - 1;

    b_tempx1l += b_potential_acoustic[iglob_kj] * d_hprime_xx[i + k * NGLLX];
    b_tempx2l += b_potential_acoustic[iglob_ik] * d_hprime_xx[j + k * NGLLX];
  }

  // 3. Получаем метрические коэффициенты и плотность
  realw rhol = rhostore[idx];
  realw xixl = d_xix[idx];
  realw xizl = d_xiz[idx];
  realw gammaxl = d_gammax[idx];
  realw gammazl = d_gammaz[idx];

  // 4. Вычисляем ускорения (связанные с градиентом потенциала)
  realw accel_loc_x = (tempx1l * xixl + tempx2l * gammaxl) / rhol;
  realw accel_loc_z = (tempx1l * xizl + tempx2l * gammazl) / rhol;

  realw b_accel_loc_x = (b_tempx1l * xixl + b_tempx2l * gammaxl) / rhol;
  realw b_accel_loc_z = (b_tempx1l * xizl + b_tempx2l * gammazl) / rhol;

  // P1: Акустический аналог Hρρ - только источник
  realw p1_value = accel_loc_x * accel_loc_x + accel_loc_z * accel_loc_z;
  hess_kl1[idx] += p1_value * dt_factor;
  
  // P2: Акустический аналог Hρρ - источник и приемник
  realw p2_dot = accel_loc_x * b_accel_loc_x + accel_loc_z * b_accel_loc_z;
  hess_kl2[idx] += fabs(p2_dot) * dt_factor;
  
  // 5. Для акустики вычисляем дивергенцию скорости (через лапласиан потенциала)
  // В акустике скорость v = ∇φ / ρ
  // Дивергенция скорости div v = ∇²φ / ρ
  
  // Вычисляем вторые производные потенциала для лапласиана
  // (это упрощенное приближение, так как точный лапласиан требует вычисления
  // вторых производных, которые в SPECFEM обычно не хранятся)
  
  // Вместо этого, для акустики используем приближение:
  // Hκκ ≈ ∫ (∇φ)² dt, где κ - объемный модуль
  
  // P3: Гессиан для κ - только источник
  realw grad_pot_sq = (tempx1l * tempx1l + tempx2l * tempx2l);
  hess_kl3[idx] += grad_pot_sq * dt_factor;
  
  // P4: Гессиан для κ - источник и приемник
  realw grad_pot_dot = fabs(tempx1l * b_tempx1l + tempx2l * b_tempx2l);
  hess_kl4[idx] += grad_pot_dot * dt_factor;
}