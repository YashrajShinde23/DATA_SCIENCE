# pre-requesite to decorators
def plus_one(number):
    number1=number+1
    return number1
plus_one(5)


#defining functions inside another function
def plus_one(number):
    def add_one(number):
        number1=number+1
        return number1
    
    result=add_one(number)
    return result
plus_one(4)


#passing fuctions as argument
# to other functions
def plus_one(number):
    result1=number+1
    return result1

def function_call(function):
    result=function(5)
    return result

function_call(plus_one)



#functions returning other functions
def hello_function():
    def say_hi():
        return "hi"
    return say_hi
hello=hello_function() 
hello()#always remember when you call hello_fuction
#directly  then it will display object not hi
#therefore you need to assign it to hello first
#then call hello() function


#need for decorators
import time
def calc_square(numbers):
    start=time.time()
    result=[]
    for number in numbers:
        result.append(number*number)
        end=time.time()
        total_time=(end-start)*1000
        print(f"total time for execution square is{total_time}")
        return result

def calc_cube(numbers):
    start=time.time()
    result=[]
    for number in numbers:
        result.append(number*number*number)
        end=time.time()
        total_time=(end-start)*1000
        print(f"total time for execution cube is {total_time}")
        return result
    
array=range(1,100000)
out_square=calc_square(array)
out_cube=calc_cube(array)
  

    
#
def say_hi():
    return "hello there"
    
def uppercase_decorator(function):
    def wrapper():
        func=function()
        make_uppercase=func.upper()
        return make_uppercase
    return wrapper
decorate=uppercase_decorator(say_hi)
decorate()    





#python provides a much  easier way for us to apply
#decorators.we simply use the @symnol before the
#function we'd like to decorate
def uppercase_decorate(function):
    def wrapper():
        func=function()
        make_uppercase=func.upper()
        return make_uppercase
    return wrapper

@uppercase_decorate
def say_hi():
    return "hello there"
say_hi()



#
def split_string(function):
    def wrapper():
        func=function()
        splitted_string=func.split()
        return splitted_string
    return wrapper
def uppercase_decorator(function):
    def wrapper():
        func=function()
        make_uppercase=func.upper()
        return make_uppercase
    return wrapper

@split_string
@uppercase_decorator
def say_hi():
    return "hello there"
say_hi()






#
import time
def time_it(func):
    #this is  decorator function that takes another function
    
    def wrapper(*args,**kwargs):
        start=time.time()
        result=func(*args,**kwargs)
        #calls the original function
        #with the provided arguments
        end=time.time()
        print(func.__name__+" took "+str((end-start)*1000)+"mil sec")
        return result
    return wrapper

@time_it
def calc_square(numbers):
    result=[]
    for number in numbers:
        result.append(number*number)
    return result

@time_it
def calc_cube(numbers):
    result=[]
    for number in numbers:
        result.append(number*number*number)
        return result

array=range(1,100000)
out_square=calc_square(array)
out_cube =calc_cube(array)




#automatically logs function calls and their arguments
def log_decorator(func):
    def wrapper(*args,**kwargs):
        print(f"calling {func.__name__} with {args} {kwargs}")
        return func(*args,**kwargs)
    return wrapper 

@log_decorator
def add(a,b):
    return a+b

print(add(3,4))#logs the function call




#access control / authentication
#cheacks if a user is authenticated before executing
def auth_required(func):
    def wrapper(user):
        if not user.get("authenticated",False):
            #.get method is used to safefly retrive
            #the value of the "authenticated" key
            #from the dictionary
            print("access denied")
            return
        return func(user)
    return wrapper 

@auth_required
def dashboard(user):
    print(f"welcome {user["name"]}!")
user1={"name":"yashraj","authenticated":True}
user2={"name":"sai","authenticated":False}

dashboard(user1)#access granted
dashboard(user2)#access denied






#input validation
def validate_positive(func):
    def wrapper(x):
        if x<0:
            raise ValueError("negative value not allowed")
        return func(x)
    return wrapper

@validate_positive
def square_root(x):
    return x**0.5
print(square_root(4))#works fine
print(square_root(-4))# raises valueerror




#
import time
def rate_limiter(max_calls,time_frame):
    calls=[]
    def decorator(func):
        def wrapper(*args,**kwargs):
            now=time.time()
            while calls and now-calls[0]>time_frame:
                
                calls.pop(0)
                
            if len(calls)>=max_calls:
                print("rate limit exceeded.try again later")
                return
            
            calls.append(now)
            return func(*args,**kwargs)
        return wrapper
    return decorator

@rate_limiter(3,10)#max 3 calls in 10 seconds
def say_hello():
    print("hello!")
say_hello()
say_hello()
say_hello()
say_hello()#this call will be rate limited
                
