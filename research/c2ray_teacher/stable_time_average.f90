! Stable evaluation of the same analytic eigenmode time average used by DORIC.
module stable_time_average
  use precision, only: dp
  implicit none
contains
  pure real(dp) function average_exponential(x) result(value)
    real(dp), intent(in) :: x
    real(dp) :: term
    integer :: k
    if(abs(x)<0.1_dp) then
      value=1.0_dp
      term=1.0_dp
      do k=1,12
        term=term*x/real(k+1,dp)
        value=value+term
      enddo
    else
      value=(exp(x)-1.0_dp)/x
    endif
  end function average_exponential
end module stable_time_average
