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


__global__ void compute_coupling_acoustic_el_kernel(realw* displ,
                                                    realw* potential_dot_dot_acoustic,
                                                    int num_coupling_ac_el_faces,
                                                    int* coupling_ac_el_ispec,
                                                    int* coupling_ac_el_ijk,
                                                    realw* coupling_ac_el_normal,
                                                    realw* coupling_ac_el_jacobian1Dw,
                                                    int* d_ibool,
                                                    int simulation_type,
                                                    int backward_simulation,
                                                    const int PML,
                                                    const int* d_spec_to_pml,
                                                    const int NSPEC_PML_X,
                                                    const int NSPEC_PML_Z,
                                                    const realw deltat,
                                                    realw_const_p d_displ_elastic_old,
                                                    realw* d_rmemory_fsb_displ_elastic,
                                                    realw_const_p alphax_store,
                                                    realw_const_p alphaz_store,
                                                    realw_const_p betax_store,
                                                    realw_const_p betaz_store) {

  int igll = threadIdx.x;
  int iface = blockIdx.x + gridDim.x*blockIdx.y;

  int i,j,iglob,ispec;
  realw displ_x,displ_z,displ_n;
  realw nx,nz;
  realw jacobianw;

  if (iface < num_coupling_ac_el_faces){
    // gets iglob index
    // note: iglob is the same for displ(..) and potential_dot_dot(..)
    //       we take iglob from the elastic element side, since for PML the indexing requires the elastic element.
    //       however, the normal and jacobian weight are from the acoustic element side as in routine compute_coupling_acoustic_el().

    // "-1" from index values to convert from Fortran-> C indexing
    ispec = coupling_ac_el_ispec[iface*2+1] - 1;                      // ispec from elastic element

    i = coupling_ac_el_ijk[INDEX4(NDIM,2,NGLLX,0,1,igll,iface)] - 1;  // (i,j) from elastic element
    j = coupling_ac_el_ijk[INDEX4(NDIM,2,NGLLX,1,1,igll,iface)] - 1;

    iglob = d_ibool[INDEX3_PADDED(NGLLX,NGLLX,i,j,ispec)] - 1;

    // elastic displacement on global point
    displ_x = displ[iglob*2] ; // (1,iglob)
    displ_z = displ[iglob*2+1] ; // (2,iglob)

    // adjoint wavefield case
    if (simulation_type == 3 && backward_simulation == 0){
      // handles adjoint runs coupling between adjoint potential and adjoint elastic wavefield
      // adjoint definition: \partial_t^2 \bfs^\dagger = - \frac{1}{\rho} \bfnabla\phi^\dagger
      displ_x = - displ_x;
      displ_z = - displ_z;
    }

    // PML
    if (PML) {
      // PML element index
      // takes elastic side since we access d_displ_elastic_old with offset_pml for local PML elements
      int ispec_pml = d_spec_to_pml[ispec] - 1;
      // checks if element is inside the PML
      if (ispec_pml >= 0) {
        realw alpha1,beta1;
        realw coef0_1,coef1_1,coef2_1;
        realw A9;

        // to match offset for local (i,j) index to thread index (tx == i + j * NGLLX from I = (tx-J*NGLLX))
        int tx = i + j * NGLLX;

        // local PML array index
        int offset_pml = ispec_pml * NGLL2 + tx;  // ispec_pml elements in range [0,NSPEC_PML-1]

        // coefficients
        if (ispec_pml < NSPEC_PML_X){
          // in CPML_X_ONLY
          //   alpha1  == alpha_x
          //   alpha_z == 0
          //
          //   beta1  == beta_x  == alpha_x + d_x / K_x
          //   beta_z == 0
          //
          alpha1 = alphax_store[offset_pml];
          beta1 = betax_store[offset_pml];
        } else if (ispec_pml < (NSPEC_PML_X + NSPEC_PML_Z)){
          // in CPML_Z_ONLY region
          //   alpha1  == alpha_z
          //   alpha_x == 0
          //
          //   beta1  == beta_z  == alpha_z + d_z / K_z
          //   beta_x == 0
          alpha1 = alphaz_store[offset_pml];
          beta1  = betaz_store[offset_pml];
        } else {
          // not used, there should be no coupling interface in the CPML_XZ regions at the corners
          alpha1 = 0.f;
          beta1  = 0.f;
        }

        // for all PML regions
        realw c1 = __expf(-0.5f * deltat * alpha1);

        coef0_1 = c1 * c1;
        if (abs(alpha1) > 0.00001f){
          // coef1_zx_1 == (1 - c1)/alpha1
          // coef2_zx_1 == coef1 * c1
          coef1_1 = (1.f - c1) / alpha1;
          coef2_1 = coef1_1 * c1;
        } else {
          // coef1_zx_1 == 1/2 dt
          // coef2_zx_1 == coef1_zx_1
          coef1_1 = 0.5f * deltat;
          coef2_1 = coef1_1;
        }

        // memory variables update
        // see compute_coupling_acoustic_el.f90 (line ~142):
        //   ! Newmark
        //   rmemory_fsb_displ_elastic(1,1,i,j,inum) = coef0_xz_1 * rmemory_fsb_displ_elastic(1,1,i,j,inum) + &
        //                                             coef1_xz_1 * displ_elastic(1,iglob) + coef2_xz_1 * displ_elastic_old(1,iglob)
        //   rmemory_fsb_displ_elastic(1,2,i,j,inum) = coef0_xz_1 * rmemory_fsb_displ_elastic(1,2,i,j,inum) + &
        //                                             coef1_xz_1 * displ_elastic(2,iglob) + coef2_xz_1 * displ_elastic_old(2,iglob)
        //
        // x-comp
        realw r_x = coef0_1 * d_rmemory_fsb_displ_elastic[INDEX3(NDIM,NGLLX,0,igll,iface)] + coef1_1 * displ_x + coef2_1 * d_displ_elastic_old[offset_pml*2];
        // z-comp
        realw r_z = coef0_1 * d_rmemory_fsb_displ_elastic[INDEX3(NDIM,NGLLX,1,igll,iface)] + coef1_1 * displ_z + coef2_1 * d_displ_elastic_old[offset_pml*2+1];

        d_rmemory_fsb_displ_elastic[INDEX3(NDIM,NGLLX,0,igll,iface)] = r_x;  // (1,igll,iface)
        d_rmemory_fsb_displ_elastic[INDEX3(NDIM,NGLLX,1,igll,iface)] = r_z;  // (2,igll,iface)

        // displacement update
        // see compute_coupling_acoustic_el.f90 (line ~166):
        //    displ_x = A8 * displ_elastic(1,iglob) + A9 * rmemory_fsb_displ_elastic(1,1,i,j,inum)
        //    displ_z = A8 * displ_elastic(2,iglob) + A9 * rmemory_fsb_displ_elastic(1,2,i,j,inum)
        //
        // with coefficients A8, A9 from routine lik_parameter_computation(..),
        // note that we require K_x == K_z == 1:
        // for CPML_X_ONLY_TEMP
        //    A8 == A_0 == kappa_x == 1
        //    A9 == - A_0 * (alpha_x - beta_x) == beta_x - alpha_x
        // for CPML_Z_ONLY_TEMP
        //    A8 == A_0 == 1 / kappa_z == 1
        //    A9 == - A_0 * (beta_z - alpha_z) == alpha_z - beta_z
        if (ispec_pml < NSPEC_PML_X){
          // in CPML_X_ONLY region
          //A8 = 1.0f;
          A9 = beta1 - alpha1;
        } else if (ispec_pml < (NSPEC_PML_X + NSPEC_PML_Z)) {
          // in CPML_Z_ONLY region
          //A8 = 1.0f;
          A9 = alpha1 - beta1;
        } else {
          // in CPML_XZ region
          // should not occur
          //A8 = 1.0f; // keeps displ_x and displ_z as is
          A9 = 0.f;
        }
        // overwrites displ_x and displ_z
        displ_x += A9 * r_x;
        displ_z += A9 * r_z;
      }
    } // PML

    // gets associated normal on GLL point
    nx = coupling_ac_el_normal[INDEX3(NDIM,NGLLX,0,igll,iface)]; // (1,igll,iface)
    nz = coupling_ac_el_normal[INDEX3(NDIM,NGLLX,1,igll,iface)]; // (2,igll,iface)

    // calculates displacement component along normal
    // (normal points outwards of acoustic element)
    displ_n = displ_x * nx + displ_z * nz;

    // gets associated, weighted jacobian
    jacobianw = coupling_ac_el_jacobian1Dw[INDEX2(NGLLX,igll,iface)];

    // continuity of pressure and normal displacement on global point

    // note: Newmark time scheme together with definition of scalar potential:
    //          pressure = - chi_dot_dot
    //          requires that this coupling term uses the updated displacement at time step [t+delta_t],
    //          which is done at the very beginning of the time loop
    //          (see e.g. Chaljub & Vilotte, Nissen-Meyer thesis...)
    //          it also means you have to calculate and update this here first before
    //          calculating the coupling on the elastic side for the acceleration...
    atomicAdd(&potential_dot_dot_acoustic[iglob],jacobianw * displ_n);
  }
}

