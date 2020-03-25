/*

  C version of the inner loop of THEODOR 3
  translated from IDL version of THEODOR code, written by Albrecht Herrmann
  by Laurent Sartran <lsartran@gmail.com> - Ecole Centrale Paris - July 2011

  build with:
  $ gcc -O3 -Wall -shared -fPIC -o libtheofast.so -lm `python-config --includes` -I/usr/lib/python2.5/site-packages/numpy/core/include/numpy/ theofast.c

*/

#include <math.h>
#include <omp.h>
#include <stdio.h>

/* MUST be used with single precision floats, NOT double */
void heat_potential_time_step(int aux, int auy, float factor, float * restrict h_pot, float cc, int plusac, float tratio, float ad, float bd, float * restrict result) 
{
  int i;
  int j;
  float hpc;
  float ww;
  float d_star;
  printf("\n\n\nIn theofast.c\n");
  printf("aux: %d\n", aux);
  /* diffstar */
  if (plusac>0) 
  {
    #pragma omp parallel for private(j, hpc, ww, d_star)
    printf("Start loop 1\n");
    for (i=0; i<=auy-1; ++i)
    {
      for (j=1; j<=aux-2; ++j) 
      {
        printf("i=%d, j=%d   \n", i, j);
        hpc = h_pot[i*aux+j]-cc;
        ww = sqrt((hpc*hpc)+2.*h_pot[i*aux+j]);
        d_star = 1.0+tratio*(hpc+ww);
        result[i*aux+j] = ad+bd/(d_star*d_star);
      }
    }
    printf("End loop 1\n");
  }
  else 
  {
    #pragma omp parallel for private(j, hpc, ww, d_star)
    for (i=0; i<=auy-1; ++i)
    {
      for (j=1; j<=aux-2; ++j) 
      {
        hpc = h_pot[i*aux+j]-cc;
        ww = sqrt((hpc*hpc)+2.*h_pot[i*aux+j]);
        d_star = 1.0+tratio*(hpc-ww);
        result[i*aux+j] = ad+bd/(d_star*d_star);
      }
    }
  }

  /* bulk */
  #pragma omp parallel for private(j)
  for (i=1; i<=auy-2; ++i)
  {
    for (j=1; j<=aux-2; ++j)
    {
      result[i*aux+j] *= (h_pot[i*aux+j-1]-2.*h_pot[i*aux+j]+h_pot[i*aux+j+1]
        + factor*(h_pot[(i-1)*aux+j]-2.*h_pot[i*aux+j]+h_pot[(i+1)*aux+j]));
    }
  }
  /* edges */
  #pragma omp parallel for
  for (j=1; j<=aux-2; ++j)
  {
    result[j] *= (h_pot[j-1]-2.0*h_pot[j]+h_pot[j+1]
        + 2.0*factor*(h_pot[aux+j]-h_pot[j]));
  }
  i = auy-1;
  #pragma omp parallel for
  for (j=1; j<=aux-2; ++j)
  {
    result[i*aux+j] *= (h_pot[i*aux+j-1]-2.*h_pot[i*aux+j]+h_pot[i*aux+j+1]
        + 2.*factor*(h_pot[(i-1)*aux+j]-h_pot[i*aux+j]));
  }
  printf("Leaving theofast.c\n");

}

