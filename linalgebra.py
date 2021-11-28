import numpy as np
import random


def rand_matrix(n_l,n_h,n_dim):
    """
    Function that generates a random square matrix of your required size from a range of values

    Argument:
    n_l -- lowest random number to be drawn
    n_h -- highest random number - 1 
    n_dim -- dimension of matrix
    
    Returns:
    B -- An n by n random matrix where B_ij are drawn randomly from range[n_l, n_h+1]
    """
    B = np.random.randint(n_l, n_h+1, (n_dim,n_dim))
    return B


def LU_factor(matrix_B):
    """
    Function that generates lower and upper diagonal matrices 
    Argument:
    matrix_B -- an nxn matrix 
    
    Returns:
    matrix_L -- Lower diagonal matrix
    matrix_U -- Upper diagonal matrix
    matrix_dB -- matrix_B with diagonals removed
    """
    matrix_L = np.zeros((len(matrix_B[0]),len(matrix_B[:,0])))
    matrix_U = np.zeros((len(matrix_B[0]),len(matrix_B[:,0])))

    #LU factorrisation done the good old fashioned way
    for i in range(len(matrix_B[0])):
        matrix_B[i][i] = 0

        for j in range(len(matrix_B[i][i:])):
            if i == i+j:                                     #Make diagonals equal to 1
                matrix_L[i][i] = 1
                matrix_U[i][i] = 1  
            else:                                            #Here we create the upper and lower matrices.  
                matrix_U[i][i+j] =+ matrix_B[i][i+j]

                matrix_L[i+j][i] =+ matrix_B[i+j][i]
    matrix_dB = matrix_B 
    return matrix_L, matrix_U, matrix_dB

def determinant(A):
  """ This function takes a square matrix A and returns its determinant by row/column expansion
      matrix is not in row echelon form, so computationally expensive
  """

  if len(A) == 2:         #Standard 2 by 2 determinant calculation
    value = A[0][0]*A[1][1] - A[1][0]*A[0][1]
    return value

  else:
    total = 0
    for i in range(0,len(A)):  #This recursive method expands over the first row
                               # so it iterates over every element in that row
      sign = (-1)**i           #Calculates altenating signs


      sub_matrix = np.delete(A,0,0)  #Deletes the necessary rows and columns. numpy.delete(arr, obj, axis=None)
      sub_matrix = np.delete(sub_matrix,i,1)

      value = sign * A[0][i]*(determinant(sub_matrix)) #Finds sub determinants
      total += value                                   #with a nested function
    return total

def quotients(s_0,v_0,A):
    d = s_0*np.identity(len(A))
    _ = multi_dot([v_0 , A ,v_0])
    s = (_)/np.dot(v_0, v_0)
    q = np.dot(LA.inv(A-d),v_0)
    v = q/LA.norm(q) 
    return s,v



def rayleigh(A,iterations):
    s = random.randint(A.min(),A.max())
    v = np.random.rand(len(A))
    i =0
    np.random
    while (i < iterations):
        s,v = quotients(s,v,A)
        i +=1

    return s,v

