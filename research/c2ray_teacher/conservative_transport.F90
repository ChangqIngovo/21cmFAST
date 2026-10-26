! Opt-in transport discretization. Native short characteristics stay available.
! Each angular pixel is one continuous ray. Its outgoing column is the next
! cell's incoming column, so primary table differences telescope along it.
module conservative_transport
  use precision, only: dp
  use sizes, only: mesh
  use my_mpi
  use grid, only: dr,vol
  use material, only: ndens
  use abundances, only: abu_he
  use c2ray_parameters, only: fraction_floor=>epsilon,use_LLS,max_subbox
  use evolve_data, only: xh_av,xhe_av,phih_grid,phihe_grid,phiheat,periodic_bc
  use sourceprops, only: srcpos,NormFlux
  use photonstatistics, only: photon_loss
  use radiation_photoionrates, only: photrates,photoion_rates
#ifdef C2_DIAGNOSTICS
  use transport_diagnostics, only: diag_record_cell,diag_subbox_start,diag_record_boundary,diag_source_complete
#endif
  implicit none
  private
  public :: transport_source
  integer, parameter :: angular_order=C2_RAY_ORDER
  real(dp), parameter :: pi=acos(-1.0_dp)
  real(dp), allocatable, save :: directions(:,:),weights(:)
contains
  pure real(dp) function primitive(a,b)
    real(dp), intent(in) :: a,b
    primitive=atan2(a*b,sqrt(1.0_dp+a*a+b*b))
  end function primitive

  subroutine initialize_angles()
    integer :: axis,sign,i,j,p,u,v
    real(dp) :: a0,a1,b0,b1,d(3)
    if(allocated(weights)) return
    allocate(weights(6*angular_order**2),directions(3,6*angular_order**2))
    p=0
    do axis=1,3
      u=mod(axis,3)+1
      v=mod(axis+1,3)+1
      do sign=-1,1,2
        do i=0,angular_order-1
          a0=-1.0_dp+2.0_dp*i/angular_order
          a1=-1.0_dp+2.0_dp*(i+1)/angular_order
          do j=0,angular_order-1
            b0=-1.0_dp+2.0_dp*j/angular_order
            b1=-1.0_dp+2.0_dp*(j+1)/angular_order
            d(axis)=real(sign,dp)
            d(u)=0.5_dp*(a0+a1)
            d(v)=0.5_dp*(b0+b1)
            p=p+1
            directions(:,p)=d/sqrt(sum(d*d))
            weights(p)=(primitive(a1,b1)-primitive(a0,b1)-primitive(a1,b0)+primitive(a0,b0))/(4.0_dp*pi)
          enddo
        enddo
      enddo
    enddo
    if(abs(sum(weights)-1.0_dp)>1e-12_dp) error stop 'Invalid geometric sky quadrature'
  end subroutine initialize_angles

  subroutine transport_source(ns)
    integer,intent(in) :: ns
    integer :: ray,k,cell(3),pos(3),step(3),left(3),right(3)
    real(dp) :: d(3),next_face(3),delta(3),t,t_next,length,w,columns(3),out_columns(3),neutral(3)
    real(dp) :: escape,ray_escape,tolerance
    type(photrates) :: phi
    if(use_LLS) error stop 'Conservative rays currently require LLS disabled'
    if(any(abs(dr-dr(1))>1e-12_dp*dr(1))) error stop 'Conservative rays require cubic cells'
    if(NormFlux(ns)<=0.0_dp) return
    call initialize_angles()
    if(periodic_bc) then
      left=srcpos(:,ns)-min(max_subbox,mesh/2)
      right=srcpos(:,ns)+min(max_subbox,mesh/2-1+mod(mesh,2))
    else
      left=max(srcpos(:,ns)-max_subbox,1)
      right=min(srcpos(:,ns)+max_subbox,mesh)
    endif
#ifdef C2_DIAGNOSTICS
    call diag_subbox_start()
#endif
    escape=0.0_dp
    !$omp parallel do default(shared) schedule(static) reduction(+:escape) &
    !$omp private(ray,k,cell,pos,step,d,next_face,delta,t,t_next,length,w,columns,out_columns,neutral,phi,ray_escape,tolerance)
    do ray=1,size(weights)
      d=directions(:,ray)
      w=weights(ray)
      cell=srcpos(:,ns)
      columns=0.0_dp
      t=0.0_dp
      do k=1,3
        step(k)=0
        delta(k)=huge(1.0_dp)/4.0_dp
        if(abs(d(k))>0.0_dp) then
          step(k)=int(sign(1.0_dp,d(k)))
          delta(k)=1.0_dp/abs(d(k))
        endif
      enddo
      next_face=0.5_dp*delta
      do
        t_next=minval(next_face)
        length=(t_next-t)*dr(1)
        pos=modulo(cell-1,mesh)+1
        neutral(1)=ndens(pos(1),pos(2),pos(3))*(1.0_dp-abu_he)*max(xh_av(pos(1),pos(2),pos(3),0),fraction_floor)
        neutral(2)=ndens(pos(1),pos(2),pos(3))*abu_he*max(xhe_av(pos(1),pos(2),pos(3),0),fraction_floor)
        neutral(3)=ndens(pos(1),pos(2),pos(3))*abu_he*max(xhe_av(pos(1),pos(2),pos(3),1),fraction_floor)
        out_columns=columns+neutral*length
        phi=photoion_rates(columns(1),out_columns(1),columns(2),out_columns(2),columns(3),out_columns(3), &
            vol/w,ns,max(xh_av(pos(1),pos(2),pos(3),1),fraction_floor))
#ifdef C2_DIAGNOSTICS
        call diag_record_cell(phi%photo_cell_HI,phi%photo_cell_HeI,phi%photo_cell_HeII,vol)
#endif
        !$omp atomic update
        phih_grid(pos(1),pos(2),pos(3))=phih_grid(pos(1),pos(2),pos(3))+phi%photo_cell_HI/neutral(1)
        !$omp atomic update
        phihe_grid(pos(1),pos(2),pos(3),0)=phihe_grid(pos(1),pos(2),pos(3),0)+phi%photo_cell_HeI/neutral(2)
        !$omp atomic update
        phihe_grid(pos(1),pos(2),pos(3),1)=phihe_grid(pos(1),pos(2),pos(3),1)+phi%photo_cell_HeII/neutral(3)
        !$omp atomic update
        phiheat(pos(1),pos(2),pos(3))=phiheat(pos(1),pos(2),pos(3))+phi%heat
        columns=out_columns
        t=t_next
        tolerance=32.0_dp*epsilon(1.0_dp)*max(1.0_dp,t_next)
        do k=1,3
          if(next_face(k)<=t_next+tolerance) then
            cell(k)=cell(k)+step(k)
            next_face(k)=next_face(k)+delta(k)
          endif
        enddo
        if(any(cell<left).or.any(cell>right)) exit
      enddo
      ray_escape=phi%photo_out*w
      escape=escape+ray_escape
#ifdef C2_DIAGNOSTICS
      call diag_record_boundary(ray_escape)
#endif
    enddo
    !$omp end parallel do
    photon_loss(1)=photon_loss(1)+escape
#ifdef C2_DIAGNOSTICS
    call diag_source_complete()
#endif
  end subroutine transport_source
end module conservative_transport
