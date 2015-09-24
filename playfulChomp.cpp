/* Trials with CHOMP.
 *
 * Copyright (C) 2014 Roland Philippsen. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


/**
   \file playfulChomp.cpp 
   (modified from pp2d.cpp on Sep., 2015 by Martin Cooney to generate 
   "playful" motions)

   You can play with the parameters to generate various kinds of trajectories
   which have a functional part (the robot goes from start to end) and
   a playful part (like following a snaking sine curve)
   Compile with "cmake .." and "make", run with "./playfulChomp",
   press "jumble" to initialize waypoints, then "run" to find a trajectory

   original description for pp2d.cpp: 
   Interactive trials with CHOMP for point vehicles moving
   holonomously in the plane.  There is a fixed start and goal
   configuration, and you can drag a circular obstacle around to see
   how the CHOMP algorithm reacts to that.  Some of the computations
   involve guesswork, for instance how best to compute velocities, so
   a simple first-order scheme has been used.  This appears to produce
   some unwanted drift of waypoints from the start configuration to
   the end configuration.  Parameters could also be tuned a bit
   better.  Other than that, it works pretty nicely.
*/


#include "gfx.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include <sys/time.h>
#include <err.h>

#define  PI  3.14159265

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Isometry3d Transform;

using namespace std;


//////////////////////////////////////////////////
// trajectory etc

Vector xi;					                        // the trajectory (q_1, q_2, ...q_n)
Vector qs;					                        // the start config a.k.a. q_0
Vector qe;					                        // the end config a.k.a. q_(n+1)
static size_t const nq (20);			          // number of q stacked into xi
static size_t const cdim (2);			          // dimension of config space
static size_t const xidim (nq * cdim); 		  // dimension of trajectory, xidim = nq * cdim
static double const dt (1.0);	       		    // time step
static double const eta (100.0); 		        // >= 1, regularization factor for gradient descent
static double const lambda (1.0); 		      // weight of smoothness objective

						// variables for playfulness:
static double const lambda_shape (0.05); 	  // weight of shape objective
static double const lambda_autotely (0.05); // weight of autotely objective
double AMPLITUDE_PARAMETER = 2.0;		        // used to control the amplitude and frequency of the sinusoid
double FREQUENCY_PARAMETER = 1.0;
double AUTOTELY_STRENGTH = 1.0; 		        // used to stay away from a distractor object 
double AUTOTELY_STRENGTH2 = 0.0; 

//////////////////////////////////////////////////
// gradient descent etc

Matrix AA;			// metric 
Vector bb;			// acceleration bias for start and end config
Matrix Ainv;			// inverse of AA 

//////////////////////////////////////////////////
// gui stuff

enum { PAUSE, STEP, RUN } state;

struct handle_s {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  handle_s (double radius, double red, double green, double blue, double alpha)
    : point_(2),
      radius_(radius),
      red_(red),
      green_(green),
      blue_(blue),
      alpha_(alpha)
  {
  }
  
  Vector point_;
  double radius_, red_, green_, blue_, alpha_;
};

				//radius, RGBA
static handle_s repulsor (0.75, 0.0, 0.0, 1.0, 0.5); 	    //blue
static handle_s distractor (0.75, 1.0, 0.0, 0.0, 0.5); 	  //red
static handle_s goal (0.75, 0.0, 1.0, 0.0, 0.5); 	        //green
static handle_s start (0.75, 1.0, 1.0, 0.0, 0.7); 	      //yellow

static handle_s * handle[] = { &repulsor, &goal, &distractor, &start, 0 }; 
static handle_s * grabbed (0);
static Vector grab_offset (3);

double GOAL_X =10.0;
double GOAL_Y =0.0;
double START_X =0.0;
double START_Y =0.0;
double DISTRACTOR_X =10.0;
double DISTRACTOR_Y =-4.0;
double REPULSOR_X =10.0;
double REPULSOR_Y =-8.0;


//////////////////////////////////////////////////
// robot (one per waypoint)

class Robot
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  Robot ()
    : position_ (Vector::Zero(2))
  {
  }
  
  
  void update (Vector const & position)
  {
    if (position.size() != 2) {
      errx (EXIT_FAILURE, "Robot::update(): position has %zu DOF (but needs 2)",
	    (size_t) position.size());
    }
    position_ = position;
  }
  
  
  void draw () const
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.7, 0.7, 0.7, 0.5);
    gfx::fill_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
    
    // thick circle outline for base
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
  }
  
  static double const radius_;
  
  Vector position_;
};

double const Robot::radius_ (0.5);

Robot rstart;
Robot rend;
vector <Robot> robots;


//////////////////////////////////////////////////

static void update_robots ()
{
  rstart.update (qs);
  rend.update (qe);
  if (nq != robots.size()) {
    robots.resize (nq);
  }
  for (size_t ii (0); ii < nq; ++ii) {
    robots[ii].update (xi.block (ii * cdim, 0, cdim, 1));
  }
}


static void init_chomp ()
{
  std::cout << "init chomp\n";

  qs.resize (cdim);
  qs << 1.0, 0.0; 
  xi = Vector::Zero (xidim);
  qe.resize (cdim);
  qe << 9.0, 0.0;
  
  repulsor.point_ << REPULSOR_X, REPULSOR_Y; 
  distractor.point_ << DISTRACTOR_X, DISTRACTOR_Y;
  goal.point_ << GOAL_X, GOAL_Y;
  start.point_ << START_X, START_X;

  AA = Matrix::Zero (xidim, xidim);
  for (size_t ii(0); ii < nq; ++ii) {
    AA.block (cdim * ii, cdim * ii, cdim , cdim) = 2.0 * Matrix::Identity (cdim, cdim);
    if (ii > 0) {
      AA.block (cdim * (ii-1), cdim * ii, cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
      AA.block (cdim * ii, cdim * (ii-1), cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
    }
  }
  AA /= dt * dt * (nq + 1);
  
  bb = Vector::Zero (xidim);
  bb.block (0,            0, cdim, 1) = qs;
  bb.block (xidim - cdim, 0, cdim, 1) = qe;
  bb /= - dt * dt * (nq + 1);
  
  // not needed anyhow
  // double cc (double (qs.transpose() * qs) + double (qe.transpose() * qe));
  // cc /= dt * dt * (nq + 1);
  
  Ainv = AA.inverse();
  
  // cout << "AA\n" << AA
  //      << "\nAinv\n" << Ainv
  //      << "\nbb\n" << bb << "\n\n";
}


static void cb_step ()
{
  state = STEP;
}


static void cb_run ()
{
  if (RUN == state) {
    state = PAUSE;
  }
  else {
    state = RUN;
  }
}


static void cb_jumble ()
{
  for (size_t ii (0); ii < xidim; ++ii) {
    xi[ii] = double (rand()) / (0.1 * numeric_limits<int>::max()) - 5.0;
  }
  update_robots();
}


static void cb_idle ()
{
  if (PAUSE == state) {
    return;
  }
  if (STEP == state) {
    state = PAUSE;
  }
  
  //////////////////////////////////////////////////
  // beginning of "the" CHOMP iteration
  
  Vector nabla_smooth (AA * xi + bb);
  Vector const & xidd (nabla_smooth); 			  // indeed, it is the same in this formulation... 
  
  Vector nabla_obs (Vector::Zero (xidim));		// xidim is cdim * nq, so 40 by default...
  for (size_t iq (0); iq < nq; ++iq) { 			  // nq is number of waypoints, 20 by default....
							

    Vector const qq (xi.block (iq * cdim, 0, cdim, 1)); // grab a point vector from the large vector with all points for each waypoint
							                                          // cdim is 2, for x and y
							                                          // block(i, j, p, q): i, j is start location, p,q is size

    Vector qd; 
    if (iq == nq - 1) {
      qd = qe - xi.block (iq * cdim, 0, cdim, 1);
    }
    else {
      qd = xi.block ((iq+1) * cdim, 0, cdim, 1) - xi.block (iq * cdim, 0, cdim, 1);
    }


    
    // In this case, C and W are the same, Jacobian is identity.  We
    // still write more or less the full-fledged CHOMP expressions
    // (but we only use one body point) to make subsequent extension
    // easier.
    //
    Vector const & xx (qq);			
    Vector const & xd (qd);
    Matrix const JJ (Matrix::Identity (2, 2)); 				// a little silly here, as noted above. 
    double const vel (xd.norm()); 		
    if (vel < 1.0e-3) {							                  // avoid div by zero further down
      continue;
    }
    Vector const xdn (xd / vel); 					           
    Vector const xdd (JJ * xidd.block (iq * cdim, 0, cdim , 1));
								
    Matrix const prj (Matrix::Identity (2, 2) - xdn * xdn.transpose()); // hardcoded planar case
    Vector const kappa (prj * xdd / pow (vel, 2.0));
    Vector delta (xx - repulsor.point_);
    double const dist (delta.norm());
    static double const maxdist (4.0); 					      // hardcoded parameter
    if ((dist >= maxdist) || (dist < 1e-9)) { 				// if distance is very far go to next point
      continue;
    }
    static double const gain (10.0); 						                                // hardcoded parameter
    double const cost (gain * maxdist * pow (1.0 - dist / maxdist, 3.0) / 3.0); // hardcoded parameter
    delta *= - gain * pow (1.0 - dist / maxdist, 2.0) / dist; 			            // hardcoded parameter
    nabla_obs.block (iq * cdim, 0, cdim, 1) += JJ.transpose() * vel * (prj * delta - cost * kappa);
  }

 
  //calculations to make motions appear "playful"
 
  //1) alter the shape of the motion (we use a sine wave)

  Vector nabla_shape (Vector::Zero (xidim));
  double sine_xi, sine_yi;

  for (size_t iq (0); iq < nq; ++iq) { 	//for each waypoint, calculate a corresponding point on a sine wave, then find the difference

	sine_xi= qs(0) + (((double)iq*(qe(0) - qs(0)))/nq);
        sine_yi= AMPLITUDE_PARAMETER* (sin(FREQUENCY_PARAMETER * sine_xi * (PI/2.0))); 

	nabla_shape(iq*cdim) = xi(iq * cdim) - sine_xi;
	nabla_shape(iq*cdim+1) = xi(iq * cdim+1) - sine_yi;

	//std::cout << "xi x "<< xi(iq * cdim) << " xi y "<< xi(iq * cdim+1) << "\n"; //for debugging
	//std::cout << "nabla_shape x "<< nabla_shape(iq*cdim) << " nabla_shape y "<< nabla_shape(iq*cdim+1) << "\n";
	//std::cout << "iq "<< iq << " sine_xi "<< sine_xi <<" sine_yi "<< sine_yi << "\n";

  }

  //2) avoid a distractor object
  //vg is goal, vd is distractor, vd_prime is reflection of distractor to the other side of the goal
  Vector nabla_autotely (Vector::Zero (xidim));
  double vg_x, vg_y;
  double vd_x, vd_y;
  double vg_dot_vd;
  double vd_prime_x, vd_prime_y;
  double vg_length;

  vg_x = qe(0) - qs(0);
  vg_y = qe(1) - qs(1);
  vg_length = sqrt(vg_x*vg_x + vg_y*vg_y);
  vg_x /= vg_length;
  vg_y /= vg_length;

  vd_x= DISTRACTOR_X - qs(0); 
  vd_y= DISTRACTOR_Y - qs(1);

  vg_dot_vd= vg_x * vd_x + vg_y * vd_y;

  vd_prime_x =  2.0 * vg_x * vg_dot_vd - AUTOTELY_STRENGTH * vd_x; 
  vd_prime_y =  2.0 * vg_y * vg_dot_vd - AUTOTELY_STRENGTH * vd_y; 
  vd_prime_y += AUTOTELY_STRENGTH2;

  //std::cout << "vg_x "<< vg_x << " vg_y "<< vg_y << "\n"; //for debugging
  //std::cout << "vd_x "<< vd_x << " vd_y "<< vd_y << "\n";
  //std::cout << "vd_prime_x "<< vd_prime_x << " vd_prime_y "<< vd_prime_y << "\n";

  //next use a B spline to model a nice curve through start, vd_prime, and end
  //create 5 control points
  Vector cp1 (qs);
  cp1(0) -= 2.0; //stub
  Vector cp2 (qs);
  Vector cp3 (qs);
  cp3(0) = vd_prime_x;
  cp3(1) = vd_prime_y;
  Vector cp4 (qe);
  Vector cp5 (qe);
  cp5(0) += 2.0; //stub

  //calculate number of divisions in first interval, and in second

  double ratioOfLength = (vd_prime_x - qs(0))/(qe(0) - qs(0));
  int numberInFirstInterval = ((int)((ratioOfLength* (double)nq)));
  int numberInSecondInterval = nq - numberInFirstInterval;

  Vector splinePoints (Vector::Zero (xidim));
  double t;

  Vector a(Vector::Zero (4));
  Vector b(Vector::Zero (4));

  a(0) = (-1.0*cp1(0) +  3.0*cp2(0) + -3.0*cp3(0) + 1.0*cp4(0)) / 6.0;
  a(1) = ( 3.0*cp1(0) + -6.0*cp2(0) +  3.0*cp3(0) + 0.0*cp4(0)) / 6.0;
  a(2) = (-3.0*cp1(0) +  0.0*cp2(0) +  3.0*cp3(0) + 0.0*cp4(0)) / 6.0;
  a(3) = ( 1.0*cp1(0) +  4.0*cp2(0) +  1.0*cp3(0) + 0.0*cp4(0)) / 6.0;

  b(0) = (-1.0*cp1(1) +  3.0*cp2(1) + -3.0*cp3(1) + 1.0*cp4(1)) / 6.0;
  b(1) = ( 3.0*cp1(1) + -6.0*cp2(1) +  3.0*cp3(1) + 0.0*cp4(1)) / 6.0;
  b(2) = (-3.0*cp1(1) +  0.0*cp2(1) +  3.0*cp3(1) + 0.0*cp4(1)) / 6.0;
  b(3) = ( 1.0*cp1(1) +  4.0*cp2(1) +  1.0*cp3(1) + 0.0*cp4(1)) / 6.0;

  for (size_t iq (0); iq < (numberInFirstInterval); ++iq) { 
    t= (((double)iq+1.0) /((double)numberInFirstInterval));
    splinePoints(iq*cdim) = ((a[0]*t + a[1])*t + a[2])*t + a[3];
    splinePoints(iq*cdim+1) = ((b[0]*t + b[1])*t + b[2])*t + b[3];
  }

  a(0) = (-1.0*cp2(0) +  3.0*cp3(0) + -3.0*cp4(0) + 1.0*cp5(0)) / 6.0;
  a(1) = ( 3.0*cp2(0) + -6.0*cp3(0) +  3.0*cp4(0) + 0.0*cp5(0)) / 6.0;
  a(2) = (-3.0*cp2(0) +  0.0*cp3(0) +  3.0*cp4(0) + 0.0*cp5(0)) / 6.0;
  a(3) = ( 1.0*cp2(0) +  4.0*cp3(0) +  1.0*cp4(0) + 0.0*cp5(0)) / 6.0;

  b(0) = (-1.0*cp2(1) +  3.0*cp3(1) + -3.0*cp4(1) + 1.0*cp5(1)) / 6.0;
  b(1) = ( 3.0*cp2(1) + -6.0*cp3(1) +  3.0*cp4(1) + 0.0*cp5(1)) / 6.0;
  b(2) = (-3.0*cp2(1) +  0.0*cp3(1) +  3.0*cp4(1) + 0.0*cp5(1)) / 6.0;
  b(3) = ( 1.0*cp2(1) +  4.0*cp3(1) +  1.0*cp4(1) + 0.0*cp5(1)) / 6.0;

  for (size_t iq (numberInFirstInterval); iq < (nq); ++iq) { 	
    t= (((double)iq) /((double)numberInSecondInterval)); 
    splinePoints(iq*cdim) = ((a[0]*t + a[1])*t + a[2])*t + a[3];
    splinePoints(iq*cdim+1) = ((b[0]*t + b[1])*t + b[2])*t + b[3];
  }

  for (size_t iq (0); iq < nq; ++iq) { 	//now calculate distance of waypoints to points on spline

    nabla_autotely(iq*cdim) = xi(iq * cdim) - splinePoints(iq * cdim);
    nabla_autotely(iq*cdim+1) = xi(iq * cdim+1) - splinePoints(iq * cdim+1);

    //std::cout << "splinePoints x "<< splinePoints(iq * cdim) << " splinePoints y "<< splinePoints(iq * cdim+1) << "\n"; //for debugging
    //std::cout << "nabla_autotely x "<< nabla_autotely(iq*cdim) << " nabla_autotely y "<< nabla_autotely(iq*cdim+1) << "\n";

  }

  //Vector dxi (Ainv * (nabla_obs + lambda * nabla_smooth)); 						                          //standard CHOMP
  //Vector dxi (Ainv * (nabla_obs + lambda * nabla_smooth + lambda_shape * nabla_shape)); 		    //standard CHOMP + sinusoid
  //Vector dxi (Ainv * (nabla_obs + lambda * nabla_smooth + lambda_autotely * nabla_autotely)); 	//standard CHOMP + avoiding distractor
  Vector dxi (Ainv * (nabla_obs + lambda * nabla_smooth + lambda_shape * nabla_shape + lambda_autotely * nabla_autotely)); //all together

  xi -= dxi / eta;
  
  // end of "the" CHOMP iteration
  //////////////////////////////////////////////////
  
  update_robots ();
}


static void cb_draw ()
{
  //////////////////////////////////////////////////
  // set bounds
  
  Vector bmin (qs);
  Vector bmax (qs);
  for (size_t ii (0); ii < 2; ++ii) {
    if (qe[ii] < bmin[ii]) {
      bmin[ii] = qe[ii];
    }
    if (qe[ii] > bmax[ii]) {
      bmax[ii] = qe[ii];
    }
    for (size_t jj (0); jj < nq; ++jj) {
      if (xi[ii + cdim * jj] < bmin[ii]) {
	bmin[ii] = xi[ii + cdim * jj];
      }
      if (xi[ii + cdim * jj] > bmax[ii]) {
	bmax[ii] = xi[ii + cdim * jj];
      }
    }
  }
  
  //gfx::set_view (bmin[0] - 2.0, bmin[1] - 2.0, bmax[0] + 2.0, bmax[1] + 2.0); //this can be used to resize the view dynamically
  gfx::set_view (- 2.0, -8.0 - 2.0, 10.0 + 2.0, 4.0 + 2.0);			                //static view

  //////////////////////////////////////////////////
  // robots
  
  rstart.draw();
  for (size_t ii (0); ii < robots.size(); ++ii) {
    robots[ii].draw();
  }
  rend.draw();
  
  //////////////////////////////////////////////////
  // trj trajectory
  
  gfx::set_pen (1.0, 0.2, 0.2, 0.2, 1.0);
  gfx::draw_line (qs[0], qs[1], xi[0], xi[1]);
  for (size_t ii (1); ii < nq; ++ii) {
    gfx::draw_line (xi[(ii-1) * cdim], xi[(ii-1) * cdim + 1], xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::draw_line (xi[(nq-1) * cdim], xi[(nq-1) * cdim + 1], qe[0], qe[1]);
  
  gfx::set_pen (5.0, 0.8, 0.2, 0.2, 1.0);
  gfx::draw_point (qs[0], qs[1]);
  gfx::set_pen (5.0, 0.5, 0.5, 0.5, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {
    gfx::draw_point (xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::set_pen (5.0, 0.2, 0.8, 0.2, 1.0);
  gfx::draw_point (qe[0], qe[1]);
  
  //////////////////////////////////////////////////
  // handles
  
  for (handle_s ** hh (handle); *hh != 0; ++hh) {
    gfx::set_pen (1.0, (*hh)->red_, (*hh)->green_, (*hh)->blue_, (*hh)->alpha_);
    gfx::fill_arc ((*hh)->point_[0], (*hh)->point_[1], (*hh)->radius_, 0.0, 2.0 * M_PI);
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc ((*hh)->point_[0], (*hh)->point_[1], (*hh)->radius_, 0.0, 2.0 * M_PI);
  }
}


static void cb_mouse (double px, double py, int flags)
{
  if (flags & gfx::MOUSE_PRESS) {
    for (handle_s ** hh (handle); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
    	grab_offset = offset;
    	grabbed = *hh;
    	break;
      }
    }
  }
  else if (flags & gfx::MOUSE_DRAG) {
    if (0 != grabbed) {
      grabbed->point_[0] = px;
      grabbed->point_[1] = py;
      grabbed->point_ += grab_offset;
    }
  }
  else if (flags & gfx::MOUSE_RELEASE) {
    grabbed = 0;
  }
}



int main()
{
  struct timeval tt;
  gettimeofday (&tt, NULL);
  srand (tt.tv_usec);
  
  init_chomp();
  update_robots();  
  state = PAUSE;
  
  gfx::add_button ("jumble", cb_jumble);
  gfx::add_button ("step", cb_step);
  gfx::add_button ("run", cb_run);
  gfx::main ("playful chomp", cb_idle, cb_draw, cb_mouse);
}
