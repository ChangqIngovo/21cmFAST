MODULE pc

CONTAINS

  SUBROUTINE ionized_regions_photon_conserving(photon_field, Mhyd_field, qi, i_x, i_y, i_z, dgnrcy, ordr)
    IMPLICIT NONE

    REAL, DIMENSION(:,:,:), INTENT(in) :: photon_field, Mhyd_field
    REAL, DIMENSION(:,:,:), INTENT(out) :: qi
    INTEGER, DIMENSION(:), INTENT(in) :: i_x, i_y, i_z, dgnrcy, ordr
    
    INTEGER, DIMENSION(:,:,:), ALLOCATABLE :: overion_once
    INTEGER :: ngrid, n_all, num_overion

    ngrid = SIZE(photon_field, dim=1)
    !PRINT *, 'ngrid = ', ngrid
    n_all = ngrid ** 3
    ALLOCATE(overion_once(ngrid, ngrid, ngrid))

    CALL ionized_bubbles_with_photon_loss(photon_field, Mhyd_field, qi, i_x, i_y, i_z, dgnrcy, ordr, overion_once)
    num_overion = SUM(overion_once)
    !PRINT *, 'called ionized_bubbles_with_photon_loss, num_overion = ', num_overion, SUM(overion_once), SUM(DBLE(qi * Mhyd_field)) / SUM(DBLE(Mhyd_field))

    DO WHILE (num_overion > 0)
       CALL distribute_extra_overlap_photons(Mhyd_field, qi, i_x, i_y, i_z, ordr, dgnrcy, overion_once, num_overion)
       !PRINT *, 'num_overion = ', num_overion, SUM(overion_once), SUM(DBLE(qi * Mhyd_field)) / SUM(DBLE(Mhyd_field))
    END DO
    !PRINT *, SUM(qi) / REAL(ngrid) ** 3, MINVAL(qi), MAXVAL(qi), SUM(photon_field) / SUM(Mhyd_field), MAXLOC(qi), qi(33,42,27)


  END SUBROUTINE ionized_regions_photon_conserving


  SUBROUTINE ionized_bubbles_with_photon_loss(photon_field, Mhyd_field, qi, i_x, i_y, i_z, dgnrcy, ordr, overion_once)
    USE sort_grid_points, ONLY : iperi
    IMPLICIT NONE

    REAL, DIMENSION(:,:,:), INTENT(in) :: photon_field, Mhyd_field
    REAL, DIMENSION(:,:,:), INTENT(out) :: qi
    INTEGER, DIMENSION(:), INTENT(in) :: i_x, i_y, i_z, dgnrcy, ordr
    INTEGER, DIMENSION(:,:,:), INTENT(out) :: overion_once

    INTEGER :: ngrid, n_all, i, j, k, i_grid, i_dgnrcy, ii, jj, kk

    INTEGER, DIMENSION(:), ALLOCATABLE :: ix_p, iy_p, iz_p
    REAL :: photons_avail, Mhyd_this_dist

    INTEGER :: chunk

    !PRINT *, 'ionized_bubbles_with_photon_loss:', SUM(DBLE(photon_field)), SUM(DBLE(Mhyd_field))
    IF (SUM(DBLE(photon_field)) / SUM(DBLE(Mhyd_field)) >= 1.d0) THEN
       qi(:,:,:) = 1.0
       overion_once(:,:,:) = 0
       RETURN
    END IF

    ngrid = SIZE(photon_field, dim=1)
    !PRINT *, 'ngrid = ', ngrid
    n_all = ngrid ** 3


    qi(:,:,:) = 0.0

    IF (ngrid > 25) THEN
       chunk = 1
    ELSE
       chunk = 2
    END IF


    !!$OMP PARALLEL DEFAULT(SHARED) PRIVATE (k,j,i,photons_avail,i_grid,Mhyd_this_dist,i_dgnrcy,ii,jj,kk,ix_p,iy_p,iz_p) reduction(+:qi)

    ALLOCATE(ix_p(n_all), iy_p(n_all), iz_p(n_all))

    !!$OMP DO  schedule(dynamic, chunk)
    DO k = 1, ngrid
       !PRINT *, 'bubbles R1:', k, ngrid
       DO j = 1, ngrid
          DO i = 1, ngrid
             !PRINT *, 'i = ', i
             IF (photon_field(i,j,k) > 0.0) THEN
                photons_avail = photon_field(i,j,k)

                DO i_grid = 1, n_all
                   IF (ordr(i_grid) > 1) CYCLE
                   !PRINT *, 'i_grid, photons_avail, dgnrcy(i_grid)', i_grid, photons_avail, dgnrcy(i_grid)

                   Mhyd_this_dist = 0.0
                   DO i_dgnrcy = i_grid, i_grid + dgnrcy(i_grid) - 1

                      ix_p(i_dgnrcy) = iperi(i_x(i_dgnrcy) - 1 + i, ngrid)
                      iy_p(i_dgnrcy) = iperi(i_y(i_dgnrcy) - 1 + j, ngrid)
                      iz_p(i_dgnrcy) = iperi(i_z(i_dgnrcy) - 1 + k, ngrid)

                      ii = ix_p(i_dgnrcy)
                      jj = iy_p(i_dgnrcy)
                      kk = iz_p(i_dgnrcy)

                      Mhyd_this_dist = Mhyd_this_dist + Mhyd_field(ii, jj, kk)
                   END DO
                   !PRINT *, 'Mhyd_this_dist, photons_avail = ', Mhyd_this_dist, photons_avail
                   DO i_dgnrcy = i_grid, i_grid + dgnrcy(i_grid) - 1

                      ii = ix_p(i_dgnrcy)
                      jj = iy_p(i_dgnrcy)
                      kk = iz_p(i_dgnrcy)
                      IF (photons_avail >=  Mhyd_this_dist) THEN
                         qi(ii, jj, kk) = qi(ii, jj, kk) + 1.0
                      ELSE
                         !qi(ii, jj, kk) = qi(ii, jj, kk) + photons_avail /  Mhyd_this_dist
                         qi(ii, jj, kk) = qi(ii, jj, kk) + (photons_avail / REAL(dgnrcy(i_grid)) ) /  Mhyd_field(ii, jj, kk)
                      END IF
                   END DO
                   photons_avail = photons_avail - Mhyd_this_dist
                   IF (photons_avail <= 0.0) EXIT
                END DO
             END IF
          END DO
       END DO
    END DO
    !!$OMP END DO
    DEALLOCATE (ix_p,iy_p,iz_p)
    !!$OMP end PARALLEL
    overion_once = 0
    WHERE (qi > 1.0) overion_once = 1

  END SUBROUTINE ionized_bubbles_with_photon_loss

  SUBROUTINE distribute_extra_overlap_photons(Mhyd_field, qi, i_x, i_y, i_z, ordr, dgnrcy, overion_once, num_overion)
    USE sort_grid_points, ONLY : iperi
    IMPLICIT NONE

    REAL, DIMENSION(:,:,:), INTENT(in) :: Mhyd_field
    REAL, DIMENSION(:,:,:), INTENT(inout) :: qi
    INTEGER, DIMENSION(:), INTENT(in) :: i_x, i_y, i_z, dgnrcy, ordr
    INTEGER, DIMENSION(:,:,:), INTENT(inout) :: overion_once
    INTEGER, INTENT(out) :: num_overion


    INTEGER :: ngrid, n_all, i, j, k, i_grid, i_dgnrcy, ii, jj, kk

    REAL, DIMENSION(:,:,:), ALLOCATABLE :: qi_extra
    INTEGER, DIMENSION(:), ALLOCATABLE :: ix_p, iy_p, iz_p
    REAL :: photons_avail, Mhyd_this_dist

    INTEGER :: chunk

    IF (MINVAL(qi) >= 1.0) RETURN

    ngrid = SIZE(Mhyd_field, dim=1)
    !PRINT *, 'ngrid = ', ngrid
    n_all = ngrid ** 3

    !PRINT *, 'overlapped points = ', COUNT(qi > 1.0)

    ALLOCATE(qi_extra(ngrid, ngrid, ngrid))

    qi_extra = 0.0

    IF (ngrid > 25) THEN
       chunk = 1
    ELSE
       chunk = 2
    END IF

    !!$OMP PARALLEL DEFAULT(SHARED) PRIVATE (k,j,i,photons_avail,i_grid,Mhyd_this_dist,i_dgnrcy,ii,jj,kk,ix_p,iy_p,iz_p) reduction(+:qi_extra)

    ALLOCATE(ix_p(n_all), iy_p(n_all), iz_p(n_all))

    !!$OMP DO  schedule(dynamic, chunk)
    DO k = 1, ngrid
       DO j = 1, ngrid
          DO i = 1, ngrid

             IF (qi(i,j,k) <= 1.0) CYCLE


             photons_avail = (qi(i,j,k) - 1.0) * Mhyd_field(i,j,k)
             !PRINT *, 'i_indx, i,j,k, photons_avail, qi', i_indx, i,j,k, photons_avail, qi(i,j,k)
             qi_extra(i,j,k) = qi_extra(i,j,k) + 1.0 - qi(i,j,k)

             DO i_grid = 2, n_all
                IF (ordr(i_grid) > 1) CYCLE
                !PRINT *, 'i_grid, photons_avail, dgnrcy(i_grid)', i_grid, photons_avail, dgnrcy(i_grid)

                Mhyd_this_dist = 0.0
                DO i_dgnrcy = i_grid, i_grid + dgnrcy(i_grid) - 1

                   ix_p(i_dgnrcy) = iperi(i_x(i_dgnrcy) - 1 + i, ngrid)
                   iy_p(i_dgnrcy) = iperi(i_y(i_dgnrcy) - 1 + j, ngrid)
                   iz_p(i_dgnrcy) = iperi(i_z(i_dgnrcy) - 1 + k, ngrid)

                   ii = ix_p(i_dgnrcy)
                   jj = iy_p(i_dgnrcy)
                   kk = iz_p(i_dgnrcy)

                   Mhyd_this_dist = Mhyd_this_dist + MAX(0.0, (1.0 - qi(ii, jj, kk)) * Mhyd_field(ii, jj, kk))
                END DO
                !PRINT *, 'Mhyd_this_dist, photons_avail = ', Mhyd_this_dist, photons_avail
                DO i_dgnrcy = i_grid, i_grid + dgnrcy(i_grid) - 1
                   ii = ix_p(i_dgnrcy)
                   jj = iy_p(i_dgnrcy)
                   kk = iz_p(i_dgnrcy)

                   IF (qi(ii, jj, kk) < 1.0) THEN
                      IF (photons_avail >=  Mhyd_this_dist) THEN
                         qi_extra(ii, jj, kk) = qi_extra(ii, jj, kk) + 1.0 - qi(ii, jj, kk)
                      ELSE
                         qi_extra(ii, jj, kk) = qi_extra(ii, jj, kk) + (1.0 - qi(ii, jj, kk)) * photons_avail /  Mhyd_this_dist
                      END IF
                   END IF
                END DO
                photons_avail = photons_avail - Mhyd_this_dist
                IF (photons_avail <= 0.0) THEN
                   !PRINT *, 'i_grid = ', i_grid
                   EXIT
                END IF
             END DO

          END DO
       END DO
    END DO
    !!$OMP END DO
    DEALLOCATE (ix_p,iy_p,iz_p)
    !!$OMP end PARALLEL
    qi = qi + qi_extra
    WHERE (qi > 1.0) overion_once = 1
    num_overion = COUNT(qi > 1.0)



  END SUBROUTINE distribute_extra_overlap_photons

  ! SUBROUTINE distribute_extra_overlap_photons_parallel(Mhyd_field, qi, i_x, i_y, i_z, ordr, dgnrcy)
  !   USE sort_grid_points
  !   IMPLICIT NONE

  !   REAL, DIMENSION(:,:,:), INTENT(in) :: Mhyd_field
  !   REAL, DIMENSION(:,:,:), INTENT(inout) :: qi
  !   INTEGER, DIMENSION(:), INTENT(in) :: i_x, i_y, i_z, dgnrcy, ordr

  !   INTEGER :: ngrid, n_all, i, j, k, i_grid, i_dgnrcy, ii, jj, kk, i_indx, i_sort

  !   !REAL, DIMENSION(:), ALLOCATABLE :: qi_1d
  !   REAL, DIMENSION(:,:,:), ALLOCATABLE :: qi_old
  !   INTEGER, DIMENSION(:), ALLOCATABLE :: ix_p, iy_p, iz_p, indx_qi
  !   REAL :: photons_avail, Mhyd_this_dist

  !   IF (MINVAL(qi) >= 1.0) RETURN

  !   ngrid = SIZE(Mhyd_field, dim=1)
  !   PRINT *, 'ngrid = ', ngrid
  !   n_all = ngrid ** 3

  !   PRINT *, 'overlapped points = ', COUNT(qi > 1.0)

  !   ALLOCATE(qi_1d(n_all), qi_old(ngrid,ngrid,ngrid))

  !   qi_old = qi
  !   !qi_1d = RESHAPE(qi, (/n_all/))
  !   !PRINT *, SUM(qi_1d) / REAL(ngrid) ** 3, MINVAL(qi_1d), MAXVAL(qi_1d), Nion * SUM(Mhalo_field) / SUM(Mhyd_field)
  !   !PRINT *, MAXLOC(qi_1d)

  !   !ALLOCATE(indx_qi(n_all))
  !   !CALL indexx(-qi_1d, indx_qi)

  !   ALLOCATE(ix_p(n_all), iy_p(n_all), iz_p(n_all))
  !   !DO i_indx = 1, n_all
  !   !i_sort = indx_qi(i_indx)
  !   !IF (qi_1d(i_sort) <= 1.0) EXIT

  !   !k = (i_sort - 1) / ngrid ** 2 + 1
  !   !j = (i_sort - 1 - ngrid ** 2 * (k-1)) / ngrid + 1
  !   !i = MOD(i_sort - 1, ngrid) + 1
  !   DO k = 1, ngrid
  !      DO j = 1, ngrid
  !         DO i = 1, ngrid

  !            IF (qi_old(i,j,k) <= 1.0) EXIT
  !            photons_avail = (qi_old(i,j,k) - 1.0) * Mhyd_field(i,j,k)
  !      !PRINT *, 'i_indx, i,j,k, photons_avail, qi', i_indx, i,j,k, photons_avail, qi(i,j,k)
  !            qi(i,j,k) = 1.0

  !            DO i_grid = 1, n_all
  !               ix_p(i_grid) = iperi(i_x(i_grid) - 1 + i, ngrid)
  !               iy_p(i_grid) = iperi(i_y(i_grid) - 1 + j, ngrid)
  !               iz_p(i_grid) = iperi(i_z(i_grid) - 1 + k, ngrid)
  !            END DO

  !            DO i_grid = 2, n_all
  !               IF (ordr(i_grid) > 1) CYCLE
  !               !PRINT *, 'i_grid, photons_avail, dgnrcy(i_grid)', i_grid, photons_avail, dgnrcy(i_grid)

  !               Mhyd_this_dist = 0.0
  !               DO i_dgnrcy = i_grid, i_grid + dgnrcy(i_grid) - 1

  !                  ii = ix_p(i_dgnrcy)
  !                  jj = iy_p(i_dgnrcy)
  !                  kk = iz_p(i_dgnrcy)

  !                  Mhyd_this_dist = Mhyd_this_dist + MAX(0.0, (1.0 - qi_old(ii, jj, kk)) * Mhyd_field(ii, jj, kk))
  !               END DO
  !               !PRINT *, 'Mhyd_this_dist, photons_avail = ', Mhyd_this_dist, photons_avail
  !               DO i_dgnrcy = i_grid, i_grid + dgnrcy(i_grid) - 1
  !                  ii = ix_p(i_dgnrcy)
  !                  jj = iy_p(i_dgnrcy)
  !                  kk = iz_p(i_dgnrcy)

  !                  IF (qi_old(ii, jj, kk) < 1.0) THEN
  !                     IF (photons_avail >=  Mhyd_this_dist) THEN
  !                        qi(ii, jj, kk) = 1.0
  !                     ELSE
  !                        qi(ii, jj, kk) = qi(ii, jj, kk) + (1.0 - qi(ii, jj, kk)) * photons_avail /  Mhyd_this_dist
  !                     END IF
  !                  END IF
  !               END DO
  !               photons_avail = photons_avail - Mhyd_this_dist
  !               IF (photons_avail <= 0.0) THEN
  !                  !PRINT *, 'i_grid = ', i_grid
  !                  EXIT
  !               END IF
  !            END DO

  !         END DO
  !      END DO
  !   END DO

  ! END SUBROUTINE distribute_extra_overlap_photons_parallel

END MODULE pc
