import numpy as np
x=np.array([1,2,3])
x
np.all(x)
x=np.array([1,2,3,0])
np.all(x)
x=np.array([1,0,0])
np.any(x)
x=np.array([1,2,np.nan,np.inf])
x
np.isfinite(x)
np.isnan(x)
x=np.array([3,5])
y=np.array([2,5])
np.greater(x,y)
np.greater_equal(x,y)
array_2D=np.identity(3)
array_2D
rand_no=np.random.normal(0,1,3)
rand_no
a=np.arange(10,22)
a
################################
import numpy as np
lst=[1,2,3]
arr=np.array(lst)
arr
arr.ndim
arr.shape
type(arr)
arr_two=np.array([[1,2,3],
                 [3,4,5],
                 [6,7,8]])
arr_two.ndim
arr_two.shape
mat=np.matrix([[1,2,3],
               [4,5,6],
               [6,7,8]])
mat.ndim
mat.shape
###############################
import numpy as np
arr=np.random.randint(1,100,9)
arr
arr.ndim
new_arr=arr.reshape(3,3)
new_arr.ravel()
arr[2]
#extracting elements
arr[2:6]
new_arr
"""
array([[12,25,42],
       [83,46,67],
       [63,30,37]])
"""
#extracting 30,37
#row 2 column 1:3
new_arr[2,1:3]












#####################################
arr=np.random.randint(1,100,9)
arr
#finding sqrt of arr
np.sqrt(arr)
#finding sin of arr
np.sin(arr)
#finding exponential of arr
np.exp(arr)
#finding log
np.log(arr)
#finding mean
np.mean(arr)
#finding median
np.median(arr)

######################################
arr1=np.array([10,20,40,30,50,60,70])
arr1
arr1.mean()
np.percentile(arr1,25)
np.percentile(arr1,50)
np.percentile(arr1,75)
###############################
import numpy as np
import matplotlib.pyplot as plt
a=[[11,12,13],[21,22,23],[31,32,33]]
a
#convert list in array
A=np.array(a)
A
#access element in second row and third column
A[1,2]
#
A[0][0:2]
#
A[0:2,2]


#basic opertions
#create numpy array x 
x=np.array([[1,0],[0,1]])
x
#create numpy array y
y=np.array([[2,1],[1,2]])
y
#add x and y
z=x+y
z
#multiplying a numpy array by a scalar
#create a numpy y
y=np.array([[2,1],[1,2]])
y
z=y*2
z

##############################
y=np.array([[2,1],[1,2]])
y
x=np.array([[1,0],[0,1]])
x
z=x*y
z

########################
a=np.array([[0,1,1],[1,0,1]])
a
b=np.array([[1,1],[1,1],[-1,1]])
b
#we use numpy function  dot to multiply array
#calculate dot product
z=np.dot(a,b)
z
#calculate sin value of z
np.sin(z)


#####################################
#we can use the numpy attribute T to calculate the transpose
c=np.array([[1,1],[2,2],[3,3]])
c
#get transpose of c
c.T

###############################
#################################
#write a numpy pgm to get the numpy version and config
import numpy as np
print(np.__version__)
print(np.show_config())


#write numpy pgm to get help with the add function
import numpy as np
print(np.info(np.add))


#write numpy pgm to test wheather none of the






























































#hadamard product
import numpy as np
a=np.array([[1,2],
          [3,4]])
b=np.array([[5,6],
            [7,8]])
hadamard=a*b
print("hadamard product:\n",hadamard)
###########################################
#usually works with 1d vector
import numpy as np
x=np.array([1,2])
y=np.array([3,4,5])
outer=np.outer(x,y)
print("outer product:\n",outer)


#write pgm to compute the cross product
import numpy as np
p=[[1,0],[0,1]]
p=np.array(p)
q=[[1,2],[3,4]]
q=np.array(q)
print("original matrix:")
print(p)
print(q)
result1=np.cross(p,q)
result2=np.cross(q,p)
print("cross product of the said two vectors(p,q)")
print(result1)
print("cross product of the said two vectors(q,p)")
print(result2)


#write pgm to compute the determinent
import numpy as np
from numpy import linalg as LA
a=np.array([[1,0],[1,2]])
print("original 2-d array")
print(a)
print("determinent of the said 2-d array:")
print(LA.det(a))

##########################################
#write pgm to compute the eigenvalues
#and right eigenvectors of given
import numpy as np
from numpy import linalg as LA
m=np.mat("3 -2;1 0")
print("original matrix:")
print("a\n", m)
w,v=np.linalg.eig(m)
print("eigenvector of the said matrix",w)
print("eigenvector of the said matrix",v)


#write pgm to compute the inverse of a given matrix
import numpy as np
m=np.array([[1,2],[3,4]])
print("original matrix:")
print(m)
result=np.linalg.inv(m)
print("inverse of the said matrix:")
print(result)
##########################################
#write a numpy pgm to generate six random number 
import numpy as np
x=np.random.randint(low=10,high=30,size=6)
print(x)
#############################################
#write pgm to create 3x3 array
import numpy as np
x=np.random.random((3,3,3))
print(x)
###############################
#write pgm to create 5x5 array
import numpy as np
x=np.random.random((5,5))
print(x)
xmin,xmax=x.min(),x.max()
print("minimum and maximum values:")
print(xmin,xmax)
#########################################
#write a pgm to get the minimum and amximum value
#of given array along second axis
import numpy as np
x=np.arange(4).reshape((2,2))
print("\noriginal array:")
print(x)
print("\nmaximum vlaue along the second axis:")
print(np.amax(x,1))
#this finds the max vlue along axis 1
#(i.e ,along rows)
print("minimum vlaue along the second axis:")
print(np.amin(x,1))
#this finds the minimum vlaue along axis 1















