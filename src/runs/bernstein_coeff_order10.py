'''
implements a function which computes Bernstein polynomials 
and their derivatives up to the second order. 
This is often used in Bezier curve generation for smooth trajectory planning in robotics.
'''

from numpy import *
import scipy.special

def generate_order_10_bernstein_coefficients(n, tmin, tmax, t_actual):
		'''
		n: Degree of the Bernstein polynomial (in this case, 10).
		tmin: Minimum time or parameter value (start point).
		tmax: Maximum time or parameter value (end point).
		t_actual: Current time or parameter value (where we want to evaluate the polynomial).
		'''

		# Total range of the parameter
		l = tmax-tmin

		# Normalized time variable in [0, 1]
		t = (t_actual-tmin)/l

        # This part evaluates the Bernstein basis polynomials
		# scipy.special.binom(n, i): Computes the binomial coefficient
		# This formulation is useful for generating Bezier curves of degree n using control points.
		P0 = scipy.special.binom(n,0)*((1-t)**(n-0))*t**0

		P1 = scipy.special.binom(n,1)*((1-t)**(n-1))*t**1
                          
		P2 = scipy.special.binom(n,2)*((1-t)**(n-2))*t**2
                          
		P3 = scipy.special.binom(n,3)*((1-t)**(n-3))*t**3
                          
		P4 = scipy.special.binom(n,4)*((1-t)**(n-4))*t**4
                          
		P5 = scipy.special.binom(n,5)*((1-t)**(n-5))*t**5
                          
		P6 = scipy.special.binom(n,6)*((1-t)**(n-6))*t**6
                          
		P7 = scipy.special.binom(n,7)*((1-t)**(n-7))*t**7
                          
		P8 = scipy.special.binom(n,8)*((1-t)**(n-8))*t**8
                          
		P9 = scipy.special.binom(n,9)*((1-t)**(n-9))*t**9
                          
		P10 = scipy.special.binom(n,10)*((1-t)**(n-10))*t**10

		# This section manually computes the first derivatives of the Bernstein polynomials
		# These derivatives indicate how the curve’s position changes over time. 
		# This is essential for calculating velocity in motion planning
		P0dot = -10.0*(-t + 1)**9

		P1dot = -90.0*t*(-t + 1)**8 + 10.0*(-t + 1)**9

		P2dot = -360.0*t**2*(-t + 1)**7 + 90.0*t*(-t + 1)**8

		P3dot = -840.0*t**3*(-t + 1)**6 + 360.0*t**2*(-t + 1)**7

		P4dot = -1260.0*t**4*(-t + 1)**5 + 840.0*t**3*(-t + 1)**6

		P5dot = -1260.0*t**5*(-t + 1)**4 + 1260.0*t**4*(-t + 1)**5

		P6dot = -840.0*t**6*(-t + 1)**3 + 1260.0*t**5*(-t + 1)**4

		P7dot = -360.0*t**7*(-t + 1)**2 + 840.0*t**6*(-t + 1)**3

		P8dot = 45.0*t**8*(2*t - 2) + 360.0*t**7*(-t + 1)**2

		P9dot = -10.0*t**9 + 9*t**8*(-10.0*t + 10.0)

		P10dot = 10.0*t**9

		# This section manually computes the second derivatives
		# These derivatives are used to determine the acceleration along the path, 
		# which helps in smooth trajectory generation, reducing jerks.
		P0ddot = 90.0*(-t + 1)**8

		P1ddot = 720.0*t*(-t + 1)**7 - 180.0*(-t + 1)**8

		P2ddot = 2520.0*t**2*(-t + 1)**6 - 1440.0*t*(-t + 1)**7 + 90.0*(-t + 1)**8

		P3ddot = 5040.0*t**3*(-t + 1)**5 - 5040.0*t**2*(-t + 1)**6 + 720.0*t*(-t + 1)**7

		P4ddot = 6300.0*t**4*(-t + 1)**4 - 10080.0*t**3*(-t + 1)**5 + 2520.0*t**2*(-t + 1)**6

		P5ddot = 5040.0*t**5*(-t + 1)**3 - 12600.0*t**4*(-t + 1)**4 + 5040.0*t**3*(-t + 1)**5

		P6ddot = 2520.0*t**6*(-t + 1)**2 - 10080.0*t**5*(-t + 1)**3 + 6300.0*t**4*(-t + 1)**4

		P7ddot = -360.0*t**7*(2*t - 2) - 5040.0*t**6*(-t + 1)**2 + 5040.0*t**5*(-t + 1)**3

		P8ddot = 90.0*t**8 + 720.0*t**7*(2*t - 2) + 2520.0*t**6*(-t + 1)**2

		P9ddot = -180.0*t**8 + 72*t**7*(-10.0*t + 10.0)

		P10ddot = 90.0*t**8

		# P Contains the polynomial values at t_actual
		P = hstack((P0, 
			  P1, 
			  P2, 
			  P3, 
			  P4, 
			  P5, 
			  P6, 
			  P7, 
			  P8, 
			  P9, 
			  P10 ))
		
		# Pdot Contains the velocity components (first derivatives) scaled 
		# by 1/l to adjust for the normalized range.
		Pdot = hstack((P0dot, 
				 P1dot, 
				 P2dot, 
				 P3dot, 
				 P4dot, 
				 P5dot, 
				 P6dot, 
				 P7dot, 
				 P8dot, 
				 P9dot, 
				 P10dot 
				 ))/l

		# Pddot: Contains the acceleration components (second derivatives) scaled 
		# by 1/l^2 for proper scaling.
		Pddot = hstack((P0ddot, 
				  P1ddot, 
				  P2ddot, 
				  P3ddot, 
				  P4ddot, 
				  P5ddot, 
				  P6ddot, 
				  P7ddot, 
				  P8ddot, 
				  P9ddot, 
				  P10ddot 
				  ))/(l**2)

		return P, Pdot, Pddot
