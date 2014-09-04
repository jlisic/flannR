#include <flann/flann.hpp>
#include <flann/io/hdf5.h>
#include <stdio.h> 
#include <stdlib.h>

#include "R.h"
#include "Rmath.h"
#include "omp.h"




// extern start for .C interface
extern "C" {


/* mean shfit nearest neighbors */
void R_meanShiftNN(
  double * x,                   /* data to query for */
  double * train, 
  int * neighbors,
  double * distances,
  int * xRowPtr,                /* number of rows of data to query */
  int * trainRowPtr,            /* number of rows of data to train on */
  int * xColPtr,                /* number of columns of data to form train */
  int * nNeighborsPtr,          /* number of Neighbors */
  int * intSearchParameters,           /* integer parameters for search*/
  double * doubleSearchParameters,     /* double parameters for search */ 
  int * intAlgorithmParameters,           /* integer parameters for algorithm*/
  double * doubleAlgorithmParameters,     /* double parameters for algorithm*/ 
  int * methodPtr               /* enumerated algorithm type */
) {


  /* simplify the interface by dereferencing the pointers */
  size_t xRow = (size_t) * xRowPtr; 
  size_t trainRow = (size_t) * trainRowPtr; 
  size_t xCol = (size_t) * xColPtr;
  size_t method = (size_t) * methodPtr;

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


// extern end for .C interface
}
