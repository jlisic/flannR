#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <stdio.h> 
#include <stdlib.h>

#include "R.h"
#include "Rmath.h"
#include "omp.h"


/* kernel methods */




/***************************** NO KERN *********************************/
void noneKernel( 
    double * x, 
    double * train, 
    int * neighbors,
    size_t xRow, 
    size_t xCol, 
    size_t nNeighbors 
    ) {

  size_t i, j, k;

  for( i = 0; i < xRow; i++) {              // rows
    for( k = 0; k < xCol; k++) {          // dimensions
      x[ i * xCol + k] = 0;         
      for( j = 0; j < nNeighbors; j++) {            // neighbors 
        x[ i * xCol + k ] += train[ neighbors[i* nNeighbors + j] * xCol + k ]/nNeighbors;
      }
    }
  }

  return;
} 




/************************** NORMAL KERN *********************************/
void normalKernel(
  double * query,        /* query matrix */
  double * build,        /* train matrix */
  int * neighbors,      /* neighbors matrix */
  size_t queryRow,
  size_t queryCol,
  size_t nNeighbors,       /* number of neighbors */
  double * bandwidth     /* bandwidth parameters */
  ) {

  size_t i; /* query data index */
  size_t j; /* build data index */ 
  size_t k; /* neighbor index */
  size_t d; /* column index */

  double * y = (double * ) malloc( sizeof(double) * queryCol);
  double w;
  double denominator;
  double * numerator = (double * ) malloc( sizeof(double) * queryCol);
  int currentNeighbor;
  int useWeights = 1; 

  
  /* iterate over each query point */ 
  for( i = 0; i < queryRow; i++)  {

applyWeights:

    /* total weight to divide by for the denominator */
    for( d = 0; d < queryCol; d++)  numerator[d] = 0; 
    denominator = 0; 

    /* iterate over each neighbor */
    for( k = 0; k < nNeighbors; k++)  { 

      /* d'th element of the k'th neighbor of the i'th query point */ 
      currentNeighbor = neighbors[i*nNeighbors + k];
  
      /* set weight to 1 */ 
      w = 1;

      // for each point query over each dimension 
      for( d = 0; d < queryCol; d++)  { 

        // save a copy of the point 
        y[d] = build[currentNeighbor * queryCol + d]; 
        
        // double dnorm( double x, double mu, double sigma, int give_log)   
        if( useWeights == 1 )  w *=  dnorm( ( y[d] - query[i * queryCol + d]) , 0, bandwidth[d], 0 );

      }
     
      // update denominator and numerator 
      for( d = 0; d < queryCol; d++)  numerator[d] = numerator[d] + (y[d] * w); 
      denominator += w; 
    
    }
    
    
    /* run it all over again without any weights if you get a 0 denom */
    if( denominator <= 0 ) {
      useWeights = 0;
      goto applyWeights;
    }
    useWeights = 1;
 
    // update in place the query set 
    for( d = 0; d < queryCol; d++) query[i* queryCol + d] = numerator[d] / denominator;


  
  }

  free(y);
  free(numerator);

  return;
}




// extern start for .C interface
extern "C" {


/* mean shfit nearest neighbors */
void R_meanShiftNN(
  double * x,                /* data to query for */
  double * train, 
  int * neighbors,
  double * distances,
  int * xRowPtr,             /* number of rows of data to query */
  int * trainRowPtr,         /* number of rows of data to train on */
  int * xColPtr,             /* number of columns of data to form train */
  int * nNeighborsPtr,               /* number of Neighbors */
  int * intSearchParameters,         /* integer parameters for search*/
  double * doubleSearchParameters,   /* double parameters for search */ 
  int * intAlgorithmParameters,      /* integer parameters for algorithm*/
  double * doubleAlgorithmParameters,/* double parameters for algorithm*/
  int * methodPtr,                   /* enumerated algorithm type */
  int * kernelMethodPtr,             /* kernel methods */ 
  double * bandwidth                 /* bandwidth for kernel methods */
) {


  /* simplify the interface by dereferencing the pointers */
  size_t xRow =     (size_t) * xRowPtr; 
  size_t trainRow = (size_t) * trainRowPtr; 
  size_t xCol =     (size_t) * xColPtr;
  size_t method =   (size_t) * methodPtr;
  size_t kernelMethod =   (size_t) * kernelMethodPtr;

  size_t nNeighbors = (size_t) * nNeighborsPtr;

  size_t i,j,k; /* index */

  /* create Matrix ADT for input*/
  flann::Matrix<double> xMatrix( x, xRow, xCol ); 
  flann::Matrix<double> trainMatrix( train, trainRow, xCol ); 
  
  /* create Matrix ADT for output */
  flann::Matrix<double> distancesMatrix( distances, xRow, nNeighbors ); 
  flann::Matrix<int> neighborsMatrix( neighbors, xRow, nNeighbors ); 
  
  
  /* create index and set up parameters for Index */ 

  /* if KDTreeIndex */ 
  //if( method == 1 ) {
    flann::Index< flann::L2<double> > index( trainMatrix, flann::KDTreeIndexParams( intAlgorithmParameters[0] ) ); 
  //}

  /* build the index */ 
  index.buildIndex();

  /* search */
  flann::SearchParams mpSearchParameters;
  mpSearchParameters.cores = 0;
  mpSearchParameters.checks = intSearchParameters[0];

  index.knnSearch(xMatrix, neighborsMatrix, distancesMatrix, nNeighbors, mpSearchParameters); 


  /* run kernel methods */
  if( kernelMethod == 0) {
    noneKernel( x, train, neighbors,  xRow, xCol,  nNeighbors ); 
  }
  
  if( kernelMethod == 1) {
    noneKernel( x, train, neighbors,  xRow, xCol,  nNeighbors ); 
  }

  if( kernelMethod == 2) {
    normalKernel(
      x,        /* query matrix */
      train,        /* train matrix */
      neighbors,      /* neighbors matrix */
      xRow,
      xCol,
      nNeighbors,       /* number of neighbors */
      bandwidth     /* bandwidth parameters */
      ); 
  }

  return; 
}


// extern end for .C interface
}
