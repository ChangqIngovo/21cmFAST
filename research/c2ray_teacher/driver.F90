! External input/output driver. Upstream transport/chemistry sources are unchanged.
program c2ray_teacher_driver
  use iso_fortran_env, only: int32, real64, error_unit
  use ieee_arithmetic, only: ieee_is_finite
  use precision, only: dp
  use sizes, only: mesh
  use my_mpi
  use clocks, only: setup_clocks
  use file_admin, only: logf, timefile
  use grid, only: x,y,z,dr,vol,sim_volume
  use material, only: ndens,xh,xhe,temperature_grid,temper_val,isothermal,set_clumping
  use cgsconstants, only: ini_rec_colion_factors
  use cosmology_parameters, only: H0,Omega0
  use cosmology, only: zred,Hz,zred_t0,t0
  use radiation_tables, only: rad_ini,bb_photo_thick_table
  use radiation_sed_parameters, only: S_star
  use radiative_cooling, only: setup_cool
  use sourceprops, only: NumSrc,srcpos,NormFlux,srcSeries
  use evolve_data, only: evolve_ini,phih_grid,phihe_grid,phiheat,xh_av,xhe_av,xh_intermed,xhe_intermed
  use evolve, only: evolve3D
  use photonstatistics, only: initialize_photonstatistics,total_ion,totrec,totcollisions,recomions, &
       photon_loss,LLS_loss,dh0,dhe0,dhe2
  implicit none
  character(len=4096) :: input_path,output_path,line
  character(len=8) :: magic
  integer(int32) :: header(4)
  integer :: input_unit,output_unit,step,i,status,log_read_unit,converged_count,error_flag
  integer(kind=8) :: log_position=1
  real(dp) :: initial(6),step_info(3),dt,sim_time,box_cm,metadata(14),sed_ratio
  real(dp),allocatable :: temperature_output(:,:,:)

  call setup_clocks()
  call mpi_setup()
  if (command_argument_count() /= 2) call fail('Expected input.bin output.bin arguments')
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  ! Input files are immutable and identical on every rank; reads avoid MPI count limits.
  open(newunit=input_unit,file=trim(input_path),status='old',access='stream', &
       form='unformatted',action='read',convert='little_endian',iostat=status)
  if(status/=0) call fail('Cannot open binary input')
  read(input_unit,iostat=status) magic,header,initial
  if(status/=0 .or. magic/='C2RTHHE1') call fail('Bad input header')
  if(header(1)/=mesh(1) .or. header(1)<16) call fail('Grid does not match compiled dimension >=16')
  if(header(2)<1 .or. header(3)<1) call fail('At least one step and source slot required')
  if(header(4)/=1) call fail('This unchanged upstream requires thermal evolution; isothermal table logging is invalid')
  if(any(.not.ieee_is_finite(initial))) call fail('Non-finite initial state')
  if(initial(1)<0 .or. initial(2)<=0 .or. initial(3)<=0) call fail('Invalid initial physical state')
  if(any(initial(4:6)<=0) .or. initial(4)>=1 .or. sum(initial(5:6))>=1) call fail('Invalid initial fractions')
  NumSrc=header(3)
  isothermal=header(4)==0
  temper_val=initial(3)
  allocate(srcpos(3,NumSrc),NormFlux(0:NumSrc),srcSeries(NumSrc))
  read(input_unit,iostat=status) srcpos
  if(status/=0 .or. any(srcpos<1) .or. any(srcpos>mesh(1))) call fail('Invalid source positions')
  do i=1,NumSrc
    srcSeries(i)=i
  enddo
  NormFlux=0.0_dp
  allocate(ndens(mesh(1),mesh(2),mesh(3)))
  allocate(xh(mesh(1),mesh(2),mesh(3),0:1),xhe(mesh(1),mesh(2),mesh(3),0:2))
  allocate(temperature_grid(mesh(1),mesh(2),mesh(3),0:2))
  allocate(temperature_output(mesh(1),mesh(2),mesh(3)))
  allocate(x(mesh(1)),y(mesh(2)),z(mesh(3)))
  read(input_unit,iostat=status) ndens
  if(status/=0 .or. any(.not.ieee_is_finite(ndens)) .or. any(ndens<=0)) call fail('Invalid initial density')
  xh(:,:,:,0)=1.0_dp-initial(4)
  xh(:,:,:,1)=initial(4)
  xhe(:,:,:,0)=1.0_dp-initial(5)-initial(6)
  xhe(:,:,:,1)=initial(5)
  xhe(:,:,:,2)=initial(6)
  temperature_grid=real(temper_val)
  zred_t0=initial(1)
  t0=0.0_dp
  call set_environment(initial(1),initial(2))
  call ini_rec_colion_factors(temper_val)
  if(rank==0) then
    open(unit=timefile,file='results/timing.txt',status='replace',action='write')
    open(unit=90,file='results/native_photon_statistics.txt',status='replace',action='write')
    write(logf,*) 'External adapter: physical total H+He nuclei density; native source rates photons/s'
    flush(logf)
  endif
  call rad_ini()
  if(.not.isothermal) call setup_cool()
  call initialize_photonstatistics()
  call evolve_ini()
  if(rank==0) then
    ! A diagnostic only: integration consistency of untouched upstream SED tables.
    sed_ratio=sum(bb_photo_thick_table(0,:))/S_star
    open(newunit=output_unit,file=trim(output_path),status='new',access='stream', &
         form='unformatted',action='write',convert='little_endian',iostat=status)
    if(status/=0) call fail('Output must be a fresh writable path')
    write(output_unit) 'C2RTOUT1',int([mesh(1),header(2),13,14],int32)
    write(*,'(A,ES24.16,A)') '{"stage":"native_initialized","sed_table_photons_over_nominal":',sed_ratio,'}'
    flush(6)
  endif
  sim_time=0.0_dp
  do step=1,header(2)
    read(input_unit,iostat=status) step_info,ndens,NormFlux(1:NumSrc)
    if(status/=0) call fail('Truncated step input')
    if(any(.not.ieee_is_finite(step_info)) .or. step_info(1)<=0 .or. step_info(2)<0 .or. step_info(3)<=0) &
         call fail('Invalid step timing/box')
    if(any(.not.ieee_is_finite(ndens)) .or. any(ndens<=0)) call fail('Invalid step density')
    if(any(.not.ieee_is_finite(NormFlux)) .or. any(NormFlux<0)) call fail('Invalid emission rate')
    dt=step_info(1)
    box_cm=step_info(3)
    call set_environment(step_info(2),box_cm)
    NormFlux(0)=0.0_dp
    NormFlux(1:NumSrc)=NormFlux(1:NumSrc)/S_star
    call evolve3D(sim_time,dt,0)
    ! The unchanged upstream loop logs failure but does not return an error.
    ! Check its explicit convergence marker before accepting any training state.
    error_flag=0
    if(rank==0) then
      flush(logf)
      open(newunit=log_read_unit,file='results/C2Ray.log',status='old',action='read',access='stream',form='formatted')
      converged_count=0
      do
        read(log_read_unit,'(A)',pos=log_position,iostat=status) line
        if(status/=0) exit
        inquire(unit=log_read_unit,pos=log_position)
        if(index(line,'Multiple sources not converging')>0) error_flag=1
        if(index(line,'Multiple sources convergence reached')>0) converged_count=converged_count+1
      enddo
      close(log_read_unit)
      if(converged_count/=1) error_flag=1
    endif
#ifdef MPI
    call MPI_BCAST(error_flag,1,MPI_INTEGER,0,MPI_COMM_NEW,status)
#endif
    if(error_flag/=0) call fail('Native solver did not converge; output cannot be used for training')
    if(any(.not.ieee_is_finite(xh)) .or. any(.not.ieee_is_finite(xhe))) call fail('Non-finite ionization fractions')
    if(any(xh<0) .or. any(xh>1) .or. any(xhe<0) .or. any(xhe>1)) call fail('Unphysical ionization fractions')
    if(maxval(abs(sum(xh,dim=4)-1.0_dp))>1e-10_dp .or. &
       maxval(abs(sum(xhe,dim=4)-1.0_dp))>1e-10_dp) call fail('Species fractions do not sum to one')
    if(any(.not.ieee_is_finite(phih_grid)) .or. any(.not.ieee_is_finite(phihe_grid)) .or. &
       any(.not.ieee_is_finite(phiheat))) call fail('Non-finite radiation fields')
    if(any(.not.ieee_is_finite(xh_av)) .or. any(.not.ieee_is_finite(xhe_av))) &
       call fail('Non-finite time-averaged fractions')
    if(any(xh/=xh_intermed) .or. any(xhe/=xhe_intermed)) call fail('Native final state not committed')
    temperature_output=real(temperature_grid(:,:,:,2),dp)
    if(isothermal) temperature_output=temper_val
    if(any(.not.ieee_is_finite(temperature_output)) .or. any(temperature_output<=0)) call fail('Invalid temperature')
    sim_time=sim_time+dt
    if(rank==0) then
      ! These are upstream counters, not an independent exact closure assertion.
      metadata=[sim_time,dt,zred,box_cm,sum(NormFlux(1:NumSrc))*S_star*dt, &
                total_ion,totrec,totcollisions,recomions, &
                sum(photon_loss)*dt*real(product(mesh),dp),LLS_loss*dt,dh0,dhe0,dhe2]
      write(output_unit) metadata,ndens,xh(:,:,:,1),xhe(:,:,:,1),xhe(:,:,:,2),temperature_output, &
           xh_av(:,:,:,0),xhe_av(:,:,:,0),xhe_av(:,:,:,1),phih_grid,phihe_grid(:,:,:,0), &
           phihe_grid(:,:,:,1),phiheat,xhe_av(:,:,:,2)
      flush(output_unit)
      flush(logf)
      write(*,'(A,I0,A,ES24.16,A,ES24.16,A)') '{"stage":"native_step_complete","step":',step, &
           ',"redshift_mid":',zred,',"mean_xHII":',sum(xh(:,:,:,1))/real(product(mesh),dp),'}'
      flush(6)
    endif
  enddo
  read(input_unit,iostat=status) magic
  if(status==0) call fail('Unexpected trailing input records')
  close(input_unit)
  if(rank==0) then
    close(output_unit)
    close(timefile)
    close(90)
  endif
  call mpi_end()
contains
  subroutine set_environment(redshift,proper_box)
    real(dp),intent(in) :: redshift,proper_box
    integer :: j
    zred=redshift
    ! Preserve upstream matter-dominated H(z) cooling prescription.
    Hz=H0*sqrt(Omega0)*(1.0_dp+zred)**1.5_dp
    dr=proper_box/real(mesh,dp)
    vol=product(dr)
    sim_volume=proper_box**3
    do j=1,mesh(1)
      x(j)=(real(j,dp)-0.5_dp)*dr(1)
      y(j)=(real(j,dp)-0.5_dp)*dr(2)
      z(j)=(real(j,dp)-0.5_dp)*dr(3)
    enddo
    call set_clumping(zred)
  end subroutine set_environment

  subroutine fail(message)
    character(len=*),intent(in) :: message
    integer :: failure_status
    write(error_unit,'(A)') trim(message)
#ifdef MPI
    call MPI_ABORT(MPI_COMM_NEW,2,failure_status)
#endif
    error stop 2
  end subroutine fail
end program c2ray_teacher_driver
