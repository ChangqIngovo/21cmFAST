! Passive counters: none of these values feed back into the native solver.
module transport_diagnostics
  use iso_fortran_env, only: int64
  use precision, only: dp
  use sizes, only: mesh
  use my_mpi
  use abundances, only: abu_he
  use c2ray_parameters, only: epsilon
  implicit none
  private
  public :: diag_capture_primary, diag_record_cell, diag_iteration_start
  public :: diag_transport_complete, diag_write_step
  public :: diag_subbox_start, diag_record_boundary, diag_source_complete
  real(dp), save :: captured_primary(3)=0.0_dp
  !$omp threadprivate(captured_primary)
  real(dp), allocatable, save :: thread_rates(:,:), transport_neutrals(:,:,:,:)
  real(dp), allocatable, save :: thread_boundary(:)
  integer(int64), allocatable, save :: thread_visits(:)
  real(dp), save :: global_rates(6)=0.0_dp, transport_product=0.0_dp
  real(dp), save :: local_boundary=0.0_dp,global_boundary=0.0_dp
  integer(int64), save :: global_visits=0_int64
  integer, save :: last_iteration=0
contains
  subroutine diag_capture_primary(h,he,he2)
    real(dp), intent(in) :: h,he,he2
    captured_primary=[h,he,he2]
  end subroutine diag_capture_primary

  subroutine diag_record_cell(h,he,he2,cell_volume)
    real(dp), intent(in) :: h,he,he2,cell_volume
    integer :: thread_index
    thread_index=1
#ifdef MY_OPENMP
    thread_index=omp_get_thread_num()+1
#endif
    thread_rates(1:3,thread_index)=thread_rates(1:3,thread_index)+captured_primary*cell_volume
    thread_rates(4:6,thread_index)=thread_rates(4:6,thread_index)+([h,he,he2]-captured_primary)*cell_volume
    thread_visits(thread_index)=thread_visits(thread_index)+1_int64
  end subroutine diag_record_cell

  subroutine diag_subbox_start()
    thread_boundary=0.0_dp
  end subroutine diag_subbox_start

  subroutine diag_record_boundary(rate)
    real(dp), intent(in) :: rate
    integer :: thread_index
    thread_index=1
#ifdef MY_OPENMP
    thread_index=omp_get_thread_num()+1
#endif
    thread_boundary(thread_index)=thread_boundary(thread_index)+rate
  end subroutine diag_record_boundary

  subroutine diag_source_complete()
    ! The source can expand through several boxes; only its final box escapes.
    local_boundary=local_boundary+sum(thread_boundary)
  end subroutine diag_source_complete

  subroutine diag_iteration_start(h_av,he_av,iteration)
    real(dp), intent(in) :: h_av(:,:,:,0:),he_av(:,:,:,0:)
    integer, intent(in) :: iteration
    if (.not.allocated(thread_rates)) then
      allocate(thread_rates(6,max(1,nthreads)),thread_visits(max(1,nthreads)))
      allocate(thread_boundary(max(1,nthreads)))
      allocate(transport_neutrals(mesh(1),mesh(2),mesh(3),3))
    endif
    thread_rates=0.0_dp
    thread_visits=0_int64
    global_rates=0.0_dp
    global_visits=0_int64
    thread_boundary=0.0_dp
    local_boundary=0.0_dp
    global_boundary=0.0_dp
    transport_product=0.0_dp
    last_iteration=iteration
    ! Match the native denominator floor, without changing native fractions.
    transport_neutrals(:,:,:,1)=max(h_av(:,:,:,0),epsilon)
    transport_neutrals(:,:,:,2)=max(he_av(:,:,:,0),epsilon)
    transport_neutrals(:,:,:,3)=max(he_av(:,:,:,1),epsilon)
  end subroutine diag_iteration_start

  subroutine diag_transport_complete(density,h_rate,he_rate,cell_volume)
    real(dp), intent(in) :: density(:,:,:),h_rate(:,:,:),he_rate(:,:,:,0:),cell_volume
    real(dp) :: local_rates(6)
    integer(int64) :: local_visits
    integer :: status
    local_rates=sum(thread_rates,dim=2)
    local_visits=sum(thread_visits)
#ifdef MPI
    call MPI_ALLREDUCE(local_rates,global_rates,6,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_NEW,status)
    call MPI_ALLREDUCE(local_visits,global_visits,1,MPI_INTEGER8,MPI_SUM,MPI_COMM_NEW,status)
    call MPI_ALLREDUCE(local_boundary,global_boundary,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_NEW,status)
#else
    global_rates=local_rates
    global_visits=local_visits
    global_boundary=local_boundary
#endif
    transport_product=cell_volume*sum(density*((1.0_dp-abu_he)*transport_neutrals(:,:,:,1)*h_rate &
        +abu_he*transport_neutrals(:,:,:,2)*he_rate(:,:,:,0) &
        +abu_he*transport_neutrals(:,:,:,3)*he_rate(:,:,:,1)))
  end subroutine diag_transport_complete

  subroutine diag_write_step(unit,step,time,dt,nominal,sed_ratio,outflow,lls,density,h_av,he_av,h_rate,he_rate,cell_volume)
    integer, intent(in) :: unit,step
    real(dp), intent(in) :: time,dt,nominal,sed_ratio,outflow,lls,cell_volume
    real(dp), intent(in) :: density(:,:,:),h_av(:,:,:,0:),he_av(:,:,:,0:),h_rate(:,:,:),he_rate(:,:,:,0:)
    real(dp) :: updated_product,max_abs,max_relative
    real(dp), allocatable :: difference(:,:,:)
    allocate(difference(mesh(1),mesh(2),mesh(3)))
    updated_product=cell_volume*sum(density*((1.0_dp-abu_he)*h_av(:,:,:,0)*h_rate &
        +abu_he*he_av(:,:,:,0)*he_rate(:,:,:,0)+abu_he*he_av(:,:,:,1)*he_rate(:,:,:,1)))
    difference=abs(h_av(:,:,:,0)-transport_neutrals(:,:,:,1))
    max_abs=maxval(difference)
    max_relative=maxval(difference/transport_neutrals(:,:,:,1))
    difference=abs(he_av(:,:,:,0)-transport_neutrals(:,:,:,2))
    max_abs=max(max_abs,maxval(difference))
    max_relative=max(max_relative,maxval(difference/transport_neutrals(:,:,:,2)))
    difference=abs(he_av(:,:,:,1)-transport_neutrals(:,:,:,3))
    max_abs=max(max_abs,maxval(difference))
    max_relative=max(max_relative,maxval(difference/transport_neutrals(:,:,:,3)))
    deallocate(difference)
    write(unit,'(A,I0,A)',advance='no') '{"schema_version":1,"step":',step,','
    write(unit,'(A,I0,A)',advance='no') '"final_transport_iteration":',last_iteration,','
    write(unit,'(A,I0,A)',advance='no') '"recorded_cell_visits":',global_visits,','
    call scalar(unit,'time_seconds',time)
    call scalar(unit,'dt_seconds',dt)
    call scalar(unit,'nominal_emitted_photons',nominal)
    call scalar(unit,'table_emitted_photons',nominal*sed_ratio)
    call species(unit,'primary_absorptions_by_species',global_rates(1:3)*dt)
    call species(unit,'secondary_ionizations_by_species',global_rates(4:6)*dt)
    call scalar(unit,'primary_absorptions',sum(global_rates(1:3))*dt)
    call scalar(unit,'secondary_ionizations',sum(global_rates(4:6))*dt)
    call scalar(unit,'boundary_outflow_photons',global_boundary*dt)
    call scalar(unit,'native_boundary_outflow_photons',outflow)
    call scalar(unit,'lls_loss_photons',lls)
    call scalar(unit,'transport_rate_product_ionizations',transport_product*dt)
    call scalar(unit,'postchemistry_rate_product_ionizations',updated_product*dt)
    call scalar(unit,'transport_fraction_max_abs_change',max_abs)
    write(unit,'(A,ES25.16E3,A)') '"transport_fraction_max_relative_change":',max_relative,'}'
    flush(unit)
  end subroutine diag_write_step

  subroutine scalar(unit,name,value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: name
    real(dp), intent(in) :: value
    write(unit,'(A,ES25.16E3,A)',advance='no') '"'//name//'":',value,','
  end subroutine scalar

  subroutine species(unit,name,value)
    integer, intent(in) :: unit
    character(len=*), intent(in) :: name
    real(dp), intent(in) :: value(3)
    write(unit,'(A,ES25.16E3,A,ES25.16E3,A,ES25.16E3,A)',advance='no') &
        '"'//name//'": [',value(1),',',value(2),',',value(3),'],'
  end subroutine species
end module transport_diagnostics
