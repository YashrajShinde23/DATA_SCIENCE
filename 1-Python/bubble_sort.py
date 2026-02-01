def bubble_sort(lst):
    n=len(lst)
    for i in range(n-1):
        for j in range(n-i-1):
            if lst[j]>lst[j+1]:
                lst[j],lst[j+1]=lst[j+1],lst[j]
    return lst
lst=[5,3,8,4,2]
bubble_sort(lst)
        
        
        
        
#gcd question
def gcd(a,b):
    while b:
        a,b=b, a%b
    return a
num1=int(input("enter first number:"))
num2=int(input("enter second number:"))


ans=gcd(num1,num2)
print(f"gcd of {num1} and {num2} is:{ans}")
      

    

#finnding gcd using math function
import math
num1=int(input("enter first number:"))
num2=int(input("enter second number:"))
ans=math.gcd(num1, num2)
print(ans)




#finding second largest elemnt in array
def second_largest(lst):
    unique_nums=list(set(lst))
    unique_nums.sort(reverse=True)
    if len(unique_nums)>1:
        return unique_nums[1]
    else:
        return None
lst=[10,20,4,45,99,99]
print("second_largest:",second_largest(lst))




#finding second smallest number
def second_smallest(lst):
    unique_nums=list(set(lst))
    unique_nums.sort()
    if len(unique_nums)>1:
        return unique_nums[1]
    else:
        return None
lst=[10,20,4,45,99,99]
print("second smallest:",second_smallest(lst))



#missing number in list
def missing_num(lst,n):
    expected_num=n*(n+1)//2
    actual_num=sum(lst)
    missing_num=expected_num-actual_num
    return missing_num
lst=[1,2,4,5,6]
n=6
print("missing number:",missing_num(lst,n))




#reverse the string without built in function
def rev_string(s):
    rev_s=s[::-1]
    return rev_s
s="my name is yashraj"
rev_s=rev_string(s)
print(rev_s)


# reverse string in one line
rev_string=input("enter the string:")[::-1]
print(rev_string)



#LCM of two numbers
num1=int(input("enter the first number:"))
num2=int(input("enter the second number:"))
def gcd(num1,num2):
    while num2:
        num1,num2=num2,num1%num2
    return num1
gcd(num1,num2)
def lcm(num1,num2):
    lcm=abs(num1*num2)/gcd(num1,num2)
    return lcm
lcm(num1,num2)
    
      
#Gary is an avid hiker. He tracks his hikes meticulously,
#paying close attention to small details like topography.
#During his last hike, he took exactly n steps. 
#For every step he took, he noted if it was an uphill (U) 
#or a downhill (D) step. Gary’s hikes start and end 
#at sea level.

#We define the following terms:

#A mountain is a non-empty sequence of consecutive
#steps above sea level, starting with a step up 
#from sea level and ending with a step down to sea level.
#A valley is a non-
#vfo-pmcf-ptm
#def

def count_valleys(n,path):
    elevation=0
    valley_count=0
    for step in path:
        if step=="u":
          if elevation==0:
            valley_count+=1
        else:
            step=="d"
            elevation-=1
    return valley_count
n=8
path="uddduddu"
print("total valleys:",count_valleys(n, path))    






#left rotation
def left_rotate(lst,d):
    n=len(lst)
    d=d%n
    left_rot=lst[d:]+lst[:d]
    return left_rot
lst=[1,2,3,4,5]
d=2
result=left_rotate(lst,d)
print(result)


#another method for left rotation
lst=[1,2,3,4,5]
lst1=lst[2:]
lst2=lst[:2]
lst=lst1+lst2
print(lst)


#create and print matrix
mat1=[[1,2,3],
      [4,5,6],
      [7,8,9]]
for i in mat1:
    print(i)



#rows and columns
mat1=[
      [1,2,3],
      [4,5,6],
      [7,8,9]
      
      ]
rows=len(mat1)
columns=len(mat1)
print("rows:",rows)
print("columns:",columns[0])
for i in range(rows):
    for j in range(columns):
        print(f"element at [{i}][{j}]={mat[i][j]}")

    
#adding 2 matrix
mat1=[
      [1,2,3],
      [4,5,6],
      [7,8,9]
      
      ]
mat2=[
      [1,2,3],
      [4,5,6],
      [7,8,9]
      
      ]
result=[
        [0,0,0],
        [0,0,0],
        [0,0,0]
        
        
        ]
rows=len(mat1)
columns=len(mat1[0])
for i in range(rows):
    for j in range(columns):
        result[i][j]=mat1[i][j]+mat1[i][j]
result




#print diagonal of matrix
mat1=[
      [1,2,3],
      [4,5,6],
      [7,8,9]
      
      ]
rows=len(mat1)
columns=len(mat1[0])
for i in range(rows):
    for j in range(columns):
        if i==j:
            print(mat1[i][j])




#cheak if the given matrix is sparse
mat1=[
      [1,2,0],
      [0,4,0],
      [0,0,6]
      
      
      ]
rows=len(mat1)
columns=len(mat1[0])
count=0
for i in range(rows):
    for j in range(columns):
        if mat1[i][j]==0:
            count+=1
if count>(rows*columns)/2:
    print("sparse")
else:
    print("not sparse")


#cheak if matrix are identical
mat1=[
      [1,2,3],
      [4,5,6],
      
      
      ]
mat2=[
      [1,2,3],
      [4,5,6],
      
      
      ]
def are_identical(mat1,mat2):
    rows1,column1=len(mat1),len(mat1[0])
    rows2,column2=len(mat2),len(mat2[0])
    if rows1!=rows2 or column1!=column2:
        return False
    for i in range(rows1):
        for j in range(column1):
            if mat1[i][j] != mat2[i][j]:
                return False
    return True
print("are matrix identical?",are_identical(mat1, mat2))




mat1=[
      [1,2,3],
      [4,5,6]
      ]
mat2=[
      [1,2]
      [4,5]
      ]
if are_identical(mat1,mat2):
    print("matrixes are identical")
else:
    print("are not identiacal")




#convert 2d array into 1d array
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16





