! External adapter for the pinned pyC2Ray hydrogen CPU kernels.
! Upstream raytracing.f90, photorates.f90 and chemistry.f90 are unmodified.
! Compile with GREY_NOTABLES and USE_SUBBOX. No recombination or collisions.
program pc_c2ray_reference
  use iso_fortran_env, only: int32, real64, error_unit
  use ieee_arithmetic, only: ieee_is_finite
  use raytracing, only: do_all_sources
  use photorates, only: S_star
  use chemistry, only: doric
  implicit none
  integer :: n, steps, nsrc, maxiter, input, output, stat, step, iter, src, i, j, k
  integer :: nbox, total_iter, subbox_size
  integer(int32) :: header(4)
  integer, allocatable :: positions(:,:)
  character(len=8) :: magic
  character(len=4096) :: input_path, output_path
  real(real64) :: dr, sigma, duration, tolerance, dt, loss, loss_step, total_loss
  real(real64) :: err, rate_abs, emitted, actual_ions, initial_ions, primary, floor_x
  real(real64) :: controls(4), stats(8), dummy(2), norm(1), loss_fraction
  real(real64), allocatable :: density(:,:,:), rates(:), x(:,:,:), av(:,:,:)
  real(real64), allocatable :: intermediate(:,:,:), next_average(:,:,:), gamma(:,:,:)
  real(real64), allocatable :: one_gamma(:,:,:), heat(:,:,:), columns(:,:,:), previous(:,:,:)
  logical :: converged

  if (command_argument_count() /= 2) error stop 'Expected input.bin output.bin'
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  open(newunit=input,file=trim(input_path),status='old',access='stream', &
       form='unformatted',action='read',convert='little_endian',iostat=stat)
  if (stat /= 0) error stop 'Cannot open input'
  read(input,iostat=stat) magic,header,controls
  if (stat /= 0 .or. (magic /= 'PC2REF01' .and. magic /= 'PC2REF02')) error stop 'Invalid reference input'
  n=header(1); steps=header(2); nsrc=header(3); maxiter=header(4)
  dr=controls(1); sigma=controls(2); duration=controls(3); tolerance=controls(4)
  if (n<3 .or. steps<1 .or. nsrc<1 .or. maxiter<1) error stop 'Invalid sizes'
  if (any(.not.ieee_is_finite(controls)) .or. any(controls<=0)) error stop 'Invalid controls'
  subbox_size=n
  loss_fraction=0.0_real64
  if (magic=='PC2REF02') then
    read(input,iostat=stat) subbox_size,loss_fraction
    if (stat/=0) error stop 'Truncated subbox controls'
  endif
  if (subbox_size<1 .or. subbox_size>n) error stop 'Invalid subbox size'
  if (.not.ieee_is_finite(loss_fraction) .or. loss_fraction<0 .or. loss_fraction>=1) &
       error stop 'Invalid subbox loss fraction'
  allocate(positions(3,nsrc),rates(nsrc),density(n,n,n),x(n,n,n),av(n,n,n))
  allocate(intermediate(n,n,n),next_average(n,n,n),gamma(n,n,n),one_gamma(n,n,n))
  allocate(heat(n,n,n),columns(n,n,n),previous(n,n,n))
  read(input,iostat=stat) density,positions,rates
  close(input)
  if (stat/=0 .or. any(.not.ieee_is_finite(density)) .or. any(density<=0)) error stop 'Invalid density'
  if (any(.not.ieee_is_finite(rates)) .or. any(rates<0)) error stop 'Invalid rates'
  if (any(positions<1) .or. any(positions>n)) error stop 'Invalid source positions'
  x=0.0_real64
  dt=duration/real(steps,real64)
  dummy=0.0_real64
  floor_x=1.0e-14_real64
  initial_ions=0.0_real64
  total_loss=0.0_real64
  primary=0.0_real64
  total_iter=0
  err=0.0_real64

  do step=1,steps
    av=x
    previous=x
    converged=.false.
    do iter=1,maxiter
      ! The division by neutral density in upstream requires an interior
      ! time-average. This is a numerical opacity floor, not extra sources.
      av=min(av,1.0_real64-floor_x)
      gamma=0.0_real64
      loss_step=0.0_real64
      do src=1,nsrc
        if (rates(src)==0.0_real64) cycle
        norm(1)=rates(src)/S_star
        one_gamma=0.0_real64
        heat=0.0_real64
        columns=0.0_real64
        ! Call with one source at a time and sum rates before chemistry.
        ! This avoids upstream's normflux(NumSrc) indexing for unequal sources
        ! without editing the author's transport expressions or source list.
        call do_all_sources(norm,positions(:,src:src),n,subbox_size,columns,sigma,dr,density,av, &
             one_gamma,heat,nbox,loss,real(loss_fraction),dummy,dummy,dummy,dummy, &
             -20.0_real64,1.0_real64,huge(1.0_real64)**0.25_real64,2,1,n,n,n)
        gamma=gamma+one_gamma
        loss_step=loss_step+loss
      enddo
      if (any(.not.ieee_is_finite(gamma)) .or. any(gamma<0)) error stop 'Invalid native rates'
      do k=1,n
        do j=1,n
          do i=1,n
            if (gamma(i,j,k)==0.0_real64) then
              ! Native doric forms 0/0 when every physical rate is disabled.
              ! The exact zero-rate branch is unchanged state.
              intermediate(i,j,k)=x(i,j,k)
              next_average(i,j,k)=x(i,j,k)
            else
              call doric(x(i,j,k),dt,1.0e4_real64,0.0_real64,gamma(i,j,k), &
                   0.0_real64,0.0_real64,0.0_real64,0.0_real64,1.0_real64, &
                   intermediate(i,j,k),next_average(i,j,k))
            endif
          enddo
        enddo
      enddo
      if (any(.not.ieee_is_finite(intermediate)) .or. &
          any(.not.ieee_is_finite(next_average))) error stop 'Invalid native chemistry'
      err=max(maxval(abs(next_average-av)),maxval(abs(intermediate-previous)))
      if (err<=tolerance) then
        converged=.true.
        exit
      endif
      ! Damping changes the fixed-point iteration, not its converged equations.
      previous=intermediate
      av=0.5_real64*(av+next_average)
    enddo
    if (.not.converged) then
      write(error_unit,*) 'C2-Ray iteration failed',step,err
      error stop 'C2-Ray reference failed convergence'
    endif
    total_iter=total_iter+iter
    rate_abs=sum(gamma*density*(1.0_real64-av))*dr**3*dt
    primary=primary+rate_abs
    total_loss=total_loss+loss_step*dt
    x=intermediate
    if (any(x<0.0_real64) .or. any(x>1.0_real64)) error stop 'Unphysical native state'
    write(*,'(A,I0,A,I0,A,ES13.5,A,ES13.5)') 'step=',step,' iterations=',iter, &
         ' mean_xHII=',sum(x)/real(n**3,real64),' iteration_change=',err
    flush(6)
  enddo
  emitted=sum(rates)*duration
  actual_ions=sum(density*x)*dr**3
  stats=[emitted,initial_ions,actual_ions,total_loss,primary,err, &
         minval(x),maxval(x)]
  open(newunit=output,file=trim(output_path),status='new',access='stream', &
       form='unformatted',action='write',convert='little_endian',iostat=stat)
  if (stat/=0) error stop 'Cannot create reference output'
  write(output) 'PC2OUT01',int([n,steps,total_iter,1],int32),stats,x,gamma
  close(output)
end program pc_c2ray_reference
