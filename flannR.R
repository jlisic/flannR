.flannAlgorithmEnum <- function( param ) {
  
  if( param == "FLANN_INDEX_LINEAR" ) return( 0 );
  if( param == "FLANN_INDEX_KDTREE" ) return( 1 );
  if( param == "FLANN_INDEX_KMEANS" ) return( 2 );
  if( param == "FLANN_INDEX_COMPOSITE" ) return( 3 );
  if( param == "FLANN_INDEX_KDTREE_SINGLE" ) return( 3 );  # I think this is an error in the documentation
  if( param == "FLANN_INDEX_SAVED" ) return( 254 );
  if( param == "FLANN_INDEX_AUTOTUNED" ) return( 255 );

  stop( sprintf("Invalid Algorithm = %s",param) )
}

.returnKernelEnum <- function( param ) {
  if( param == "NONE" )    return( 0 );
  if( param == "UNIFORM" ) return( 1 );
  if( param == "NORMAL" )  return( 2 );
}


############################################################
# x data set
# m max obs per node
############################################################
getNN <- function(
  queryData,
  trainData,            # subset of query data
  nNeighbors,       # number of neighbors
  indexParameters = NULL,   # index parameters  
  searchParameters = NULL,  # search parameters 
  algorithm = "FLANN_INDEX_KDTREE",
  kernelMethod = "NONE",
  bandwidth = NULL
  ) {
  
  # default search parameters
  integerSearchParameters <- 0
  doubleSearchParameters <- 0


  # default index parameters defined by algorithm
  if ( algorithm == "FLANN_INDEX_KDTREE" ) {
    integerAlgorithmParameters <- 4 
    doubleAlgorithmParameters <- 0
  }

  # get data size
  trainRow <- nrow(trainData)
  queryRow <- nrow(queryData)
  queryCol <- ncol(queryData)

  # allocate space for the return vector for number of neighbors
  neighbors = rep(0,queryRow*nNeighbors)
  distances = neighbors

  # get an algorithm
  paramAlgorithm <- .flannAlgorithmEnum( algorithm )

  # get kernel method
  paramKernelMethod <- .returnKernelEnum( kernelMethod )

  if( kernelMethod == "NORMAL" ) {
    if( is.null(bandwidth) ) stop("bandwidth NULL for kernel method")
    if( length(bandwidth) != queryCol ) stop("bandwidth not equal to columns x")
  }

  # load shared libarary 
  dyn.load('~/src/dissertation/bin/flannR.so')
  #dyn.load('~/src/dissertation/src/flannR/flannR.so')
  
  Cprog <- proc.time()


  # send our data to the C program
  r.result <- .C("R_meanShiftNN",
    as.double( t(queryData) ),                 # 1 data we query
    as.double( t(trainData) ),                 # 2 sample we use to build index
    as.integer( neighbors ),                   # 3
    as.double(  distances ),                   # 4
    as.integer( queryRow ),                    # 5 number of rows of data to query
    as.integer( trainRow ),                    # 5 number of rows of training data 
    as.integer( queryCol ),                    # 6 number of columns of data to query
    as.integer( nNeighbors ),                  # 7 number of Neighbors 
    as.integer( integerSearchParameters ),     # 8 integer Parameters for searching
    as.double( doubleSearchParameters ),       # 9 double Parameters  for searching
    as.integer( integerAlgorithmParameters ),  # 10 integer Parameters for algorithm 
    as.double( doubleAlgorithmParameters ),    # 11 double Parameters for algorithm
    as.integer( paramAlgorithm ),              # 12 method to form tree
    as.integer( paramKernelMethod ),           # 13 method to form tree
    as.double( bandwidth )                     # 14 method to form tree
  )
  
  print("C running time")
  print(proc.time() - Cprog) 

#  print( sprintf("Speed up over linear %f \n", r.result[[10]] ))

  neighbors <- matrix(r.result[[3]],byrow=TRUE,ncol=nNeighbors)
  query <- matrix(r.result[[1]],byrow=TRUE,ncol=queryCol)


  return( list( neighbors=neighbors, query=query ) )
}



testShiftNN <- T

if( "testShiftNN" %in% ls() ) {
  if( testShiftNN ) {

    # example data
    x <- matrix( c( 
               1, 1,
               1, 2,
               2, 1, 
               5, 3,
               5, 4,
               4, 5 ), byrow=T, ncol=2)


#    x1 <- matrix( rnorm( 10000 ),ncol=2)
#    x2 <- matrix( rnorm( 10000 ),ncol=2) + 100 

    x1 <- x
    x2 <- x + 30
    
    x <- rbind( x1, x2 ) 

    nn <- getNN(x, x[4:9,], 
                3,
                kernelMethod = "NORMAL", bandwidth=c(3,3) 
               ) 
     print(nn)

    library(raster)
    r <- raster("~/src/dissertation/src/rSegmentation/test.gri")

    Ext <- c( 
    755774,
    755960,
    2064572,
    2064731
    )

    #r <- crop(r, extent(Ext) )  

    source('~/src/dissertation/src/cSmooth/smooth.R')
    h <- 17 
    WMean   <- matrix(1/h^2,nrow=h,ncol=h)
    WVar    <- matrix(1/h^2,nrow=h,ncol=h)
    r <- rasterSmoothMoments(r, WMean, WVar)


    r.values <- cbind(
#      rep(1:ncol(r$mu),nrow(r$mu)),
#      rep(1:nrow(r$mu),each=ncol(r$mu)),
      values(r$mu),
      values(log(r$var))
      )

     r.values.finite <- is.finite(rowSums(r.values)) 
 
     x <- r.values[ r.values.finite, ]
    
     sampleRate <- 5000 
     #bandwidth <- c(100,100,8,2)
     bandwidth <- c(8,2)
 
     index <- seq.int(from=1,to=ncell(r$mu),by=sampleRate)
 
     #index <- sample( 1:nrow(x), sampleSize ) 
     train <- x[index,]
    
     nn <- getNN(train, train, 
                 #length(index), 
                 100,
                 kernelMethod = "NORMAL" , 
                 bandwidth=bandwidth)
 
     for( i in 1:20) {
       nn <- getNN(nn$query, train, 
                   #length(index), 
                   100,
                   kernelMethod="NORMAL", 
                   bandwidth=bandwidth)
     }
    
     # classify   
     y <- getNN(x, nn$query, 1)
     
 
 #    # copy the results back
     r.new <-r$mu
     r.new.values <- values(r.new) 
     r.new.values[r.values.finite] <- y$query[,1]
     values(r.new) <- r.new.values
     
     r.new.var <-r$var
     r.new.values <- values(r.new) 
     r.new.values[r.values.finite] <- y$query[,2]
     values(r.new.var) <- r.new.values
 
     par( mfrow=c(2,2) )
     plot(r$mu)
     plot(r.new)
     plot(log(r$var))
     plot(r.new.var)
#
#    nCheck <- 2:4
#    
#    y <- x[nCheck,]
#    
#    check <- 3 
#    
#    y.check <-  t(y) - x[check,]
#    y.check.w <- dnorm(y.check)
#    y.check.w <- y.check.w[1,] * y.check.w[2,]
#    
#    output <- colSums(y * y.check.w) / sum(y.check.w)
#    print(output)
  
  }
}


