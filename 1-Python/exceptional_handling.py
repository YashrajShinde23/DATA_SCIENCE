#zero division error
a=10
b=0
result=a/b
try:
    result=a/b
except ZeroDivisionError:
    print("cannot divide by zero")
    
    


#index error
numbers=[1,2,3]
print (numbers[5])
try:
    print(number[5])
except:
    print("index out of range!")
    
    
    
# handling exceptiption without naming them
#value error
try:
    numerator=50
    denom=int(input("enter the denominator"))
    quotient=(numerator/denom)
    print("division done successfully")
except ValueError:
    print("only INTEGERS should be enter")
except:
    print("oops.....SOME EXCEPTION RAISED")



#handling exception with try...except...else

try:
    numerator=50
    denom=int(input("enter the denominator"))
    quotient=(numerator/denom)
    print("division done successfully")
except ZeroDivisionError:
    print("denominator as zero is not allowed")
except ValueError:
    print(" only INTEGERS should be allowed")
else:
    print("the result of division operation is",quotient)
    
    
    
# handling exception using try..except..else..finally
try:
    numerator=50
    denom=int(input("enter the denominator"))
    quotient=(numerator/denom)
    print("division done successfully")
except ZeroDivisionError:
    print("denominator as zero is not allowed")
except ValueError:
    print(" only INTEGERS should be allowed")
else:
    print("the result of division operation is",quotient)
finally:
    print("OVER AND OUT")
    


#filenotfounderror
try:
    with open('C:/10-python/pi_digits.txt',"r") as file:
        contents=file.read()
except FileNotFoundError:
    print("file not found")


#permission error
try:
    with open('C:/1-python/pi_digits.txt',"r") as file:
        contents=file.read()
    print(contents.rstrip())
except PermissionError:
    print("you dont have access to open the file")
   


#attribute error
obj=None
print(obj.some_attribute)


if obj is not None:
    print(obj.some_attribute)
else:
    print("object is none")    


#memory error
huge_list=[1]*(10**10)
##########
#handling using generator
def generate_numbers():
    for i in range(10**10):
        yield i
gen=generate_numbers()
print(next(gen))



#recursive error
def recursive_function():
    return recursive_function()
recursive_function()
import sys
sys.setrecursionlimit(1000)
def safe_recursive_function(depth=0,max_depth=10):
    if depth>=max_depth:
        return "done"
    return safe_recursive_function(depth+1,max_depth)
print(safe_recursive_function())




#pgm to accept 2 numbers from user and perform
#division.if any exception occurs,print an error
#message or else print the result


try:
    num1=float(input("enter number1:"))
    num2=float(input("enter number2:"))
    num=num1/num2
    print("the result is",num)
except ZeroDivisionError:
    print("Error: division by zero is not allowed.")
except ValueError:
    print("Error: please enter numeric values only.")

    

#pgm for cheaking if user given number is prime
#or not and if error occurs then handle the eror


def is_prime(num):
    if num<2:
        return False
    for i in range(2,int(num**0.5)+1):
        if num%i==0:
            return False
        return True
    
try:
    num=int(input("enter the number:"))
    if is_prime(num):
        print("number is prime")
    else:
        print("the number is not prime")
except ValueError:
    print("Error:please enter valid integer")
    
    
    
#pgm to accept the file name to be opened from
#user,if file exist print the contents of the file
#in title case or else handle the error and print
#error message





try:
    file_name=input("please enter the file name with absolute address:")
    with open(file_name,"r") as file:
        content=file.read()
        print(content.title())
except FileNotFoundError:
    print("error:file not found.please enter the correct file name with absolute")
except PermissionError:
    print("error: you dont have permission to access the file")
    
    
    
#declare a list with 10 integers and ask the user
#to enter an index.cheak whether number in that index is
#positive or negative number.if  any invalid index
#is entered,handle the exception and print an error

 


numbers=[5,-3,7,-1,12,-8,9,-6,15,2]
try:
    index=int(input("enter an index(0-9):"))
    value=numbers[index]
    if value>0:
        print(f"the number at index {index} is positive.")
    else:
        print(f"the number at index {index} is negative.")
except IndexError:
    print("Error: index is out of range.plrase enter valid")
except ValueError:
    print("Error:plaese enter a valid integer.")
    
    
