MODULE sort_grid_points
CONTAINS

  SUBROUTINE sort_grid(ngrid, i_x, i_y, i_z, dgnrcy, ordr, read_write_file)
    IMPLICIT NONE

    INTEGER, INTENT(in) :: ngrid
    INTEGER, DIMENSION(:), INTENT(out) :: i_x, i_y, i_z, dgnrcy, ordr
    LOGICAL, INTENT(in), optional :: read_write_file

    INTEGER, DIMENSION(ngrid * ngrid * ngrid) :: indx
    REAL, DIMENSION(ngrid * ngrid * ngrid) :: distsq

    INTEGER :: i, j, k, i_grid, i_begin, i_end, n_all, ref_grid_default(3)
    REAL :: box, distsq_now

    CHARACTER*200 :: file_sorted_points, ngridnumber
    LOGICAL :: file_operation, exist_old_file

    WRITE(ngridnumber,'(I4)') ngrid
    DO i=1,4
       IF (ngridnumber(i:i) .EQ. ' ') THEN
          ngridnumber(i:i)='0'
       END IF
    END DO

    IF (PRESENT(read_write_file)) THEN
       file_operation = read_write_file
    ELSE
       file_operation = .TRUE.
    END IF

    IF (file_operation) THEN
       CALL system('mkdir -p ./RUN_DATA_FILES/')
       file_sorted_points = './RUN_DATA_FILES/sorted_grid_points_N' // TRIM(ngridnumber)
       !file_sorted_points = './sorted_grid_points_N' // TRIM(ngridnumber)
    
       INQUIRE(FILE=TRIM(file_sorted_points), EXIST=exist_old_file)
       IF (exist_old_file) THEN
          OPEN(2, file=TRIM(file_sorted_points), form='unformatted', status='old',action='read')
          READ(2) i_x, i_y, i_z, dgnrcy, ordr
          CLOSE(2)
          RETURN
       END IF
    END IF
       
    box = REAL(ngrid)
    n_all = ngrid * ngrid * ngrid

    i_grid = 0
    ref_grid_default = (/1, 1, 1/)

    DO k = 1, ngrid
       DO j = 1, ngrid
          DO i = 1, ngrid

             i_grid = i_grid + 1
             distsq(i_grid) = distsq_periodic( (/REAL(ref_grid_default(1)), REAL(ref_grid_default(2)), REAL(ref_grid_default(3))/), (/REAL(i), REAL(j), REAL(k)/), box)

             i_x(i_grid) = i
             i_y(i_grid) = j
             i_z(i_grid) = k

          END DO
       END DO
    END DO

    !PRINT *, 'min/max dist = ', SQRT(MINVAL(distsq)), SQRT(MAXVAL(distsq))

    CALL indexx(distsq, indx)
    distsq = distsq(indx)
    i_x = i_x(indx)
    i_y = i_y(indx)
    i_z = i_z(indx)


    dgnrcy(:) = 0
    ordr(:) = 0

    distsq_now = distsq(1)
    i_begin = 1

    DO i = 2, n_all
       IF (distsq(i) > distsq_now) THEN
          i_end = i - 1
          dgnrcy(i_begin:i_end) = i_end - i_begin + 1
          FORALL(k = i_begin:i_end) ordr(k) = k - i_begin + 1

          distsq_now = distsq(i)
          i_begin = i
       ELSEIF (distsq(i) == distsq_now) THEN
          CONTINUE
       ELSE
          STOP ('error in ordering')
       END IF
    END DO
    dgnrcy(i_begin:n_all) = n_all - i_begin + 1
    FORALL(k = i_begin:n_all) ordr(k) = k - i_begin + 1

    IF (file_operation .AND. (.NOT. exist_old_file)) THEN
       OPEN(2, file=TRIM(file_sorted_points), form='unformatted')
       WRITE(2) i_x, i_y, i_z, dgnrcy, ordr
       CLOSE(2)
    END IF

  END SUBROUTINE sort_grid


  FUNCTION iperi(i,n)
    IMPLICIT NONE
    INTEGER, INTENT(in) :: i,n
    INTEGER :: iperi

    iperi = MODULO(i - 1, n) + 1

  END FUNCTION iperi


  FUNCTION distsq_periodic(pt1, pt2, box)
    IMPLICIT NONE
    REAL, DIMENSION(:), INTENT(in) :: pt1, pt2
    REAL, INTENT(in) :: box
    REAL :: distsq_periodic

    REAL, DIMENSION(:), ALLOCATABLE :: dist_i
    INTEGER :: ndim, i

    ndim = SIZE(pt1)
    ALLOCATE(dist_i(ndim))
    distsq_periodic = 0.0

    DO i = 1, ndim
       dist_i(i) = ABS(pt1(i) - pt2(i))
       IF (dist_i(i)  > 0.5 * box) dist_i(i) = box - dist_i(i)
       distsq_periodic = distsq_periodic + dist_i(i) ** 2
    END DO

  END FUNCTION distsq_periodic

  SUBROUTINE indexx(arr,index)
    IMPLICIT NONE
    REAL, DIMENSION(:), INTENT(IN) :: arr
    INTEGER, DIMENSION(:), INTENT(OUT) :: index
    INTEGER, PARAMETER :: NN=15, NSTACK=50
    REAL :: a
    INTEGER :: n,k,i,j,indext,jstack,l,r
    INTEGER, DIMENSION(NSTACK) :: istack
    n=SIZE(index)
    FORALL(i=1:n) INDEX(i)=i
    !index=arth(1,1,n)
    jstack=0
    l=1
    r=n
    DO
       IF (r-l < NN) THEN
          DO j=l+1,r
             indext=INDEX(j)
             a=arr(indext)
             DO i=j-1,l,-1
                IF (arr(INDEX(i)) <= a) EXIT
                INDEX(i+1)=INDEX(i)
             END DO
             INDEX(i+1)=indext
          END DO
          IF (jstack == 0) RETURN
          r=istack(jstack)
          l=istack(jstack-1)
          jstack=jstack-2
       ELSE
          k=(l+r)/2
          CALL swap(INDEX(k),INDEX(l+1))
          CALL icomp_xchg(INDEX(l),INDEX(r))
          CALL icomp_xchg(INDEX(l+1),INDEX(r))
          CALL icomp_xchg(INDEX(l),INDEX(l+1))
          i=l+1
          j=r
          indext=INDEX(l+1)
          a=arr(indext)
          DO
             DO
                i=i+1
                IF (arr(INDEX(i)) >= a) EXIT
             END DO
             DO
                j=j-1
                IF (arr(INDEX(j)) <= a) EXIT
             END DO
             IF (j < i) EXIT
             CALL swap(INDEX(i),INDEX(j))
          END DO
          INDEX(l+1)=INDEX(j)
          INDEX(j)=indext
          jstack=jstack+2
          IF (jstack > NSTACK) STOP ('indexx: NSTACK too small')
          IF (r-i+1 >= j-l) THEN
             istack(jstack)=r
             istack(jstack-1)=i
             r=j-1
          ELSE
             istack(jstack)=j-1
             istack(jstack-1)=l
             l=i
          END IF
       END IF
    END DO
  CONTAINS
    !BL
    SUBROUTINE icomp_xchg(i,j)
      INTEGER, INTENT(INOUT) :: i,j
      INTEGER :: swp
      IF (arr(j) < arr(i)) THEN
         swp=i
         i=j
         j=swp
      END IF
    END SUBROUTINE icomp_xchg
  END SUBROUTINE indexx


  SUBROUTINE swap(a,b)
    INTEGER, INTENT(INOUT) :: a,b
    INTEGER :: dum
    dum=a
    a=b
    b=dum
  END SUBROUTINE swap
END MODULE sort_grid_points
