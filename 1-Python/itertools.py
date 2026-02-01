#
lst=[]
for num in range(0,20):
    lst.append(num)
print(lst)


#same method using using list comprehension
lst=[num for num in range(0,20)]
print(lst)


#
names=["dada","mama","kaka"]
lst=[name.capitalize() for name in names]
print(lst)


#list comprehension with if statement
def is_even(num):
    return num%2==0
lst=[num for num in range(10) if is_even(num)]
print(lst)


#
lst=[f"{x}{y}"for x in range(3)for y in range(3)]
print(lst)


#set comprehension
set_one={x for x in range(3)}
print(set_one)


#dictionary coprehension
dict={x:x*x for x in range(3)}
print(dict)


#generator
gen=(x for x in range(3))
print(gen)
for num in gen:
    print(num)
    

#
gen=(x
     for x in range(3)
     
     )
next(gen)



#function which returns multiple values
def range_even(end):
    for num in range(0,end,2):
        yield num

for num in range_even(6):
    print(num)
    
    
#instead of using for loop we can write our own  generator
gen=range_even(6)
next(gen)
next(gen)


#let us hide passwords enterd on screen
#chaining generators
def lengths(itr):
    for ele in itr:
        yield len(ele)
def hide(itr):
    for ele in itr:
        yield ele*'*'
        
passwords=["not-good","give'm-pass","001100=100"]

for passwords in hide(lengths(passwords)):
    print(passwords)
    
    
    
#enumerate
#printig list with index
lst=["milk","egg","bread"]
for index in range(len(lst)):
    print(f"{index+1} {lst[index]}")
    
#same code using enumerate
lst=["milk","egg","bread"]
for index,item in enumerate(lst,start=1):
    print(f"{index} {item}")
    
    
#use of zip function
name=["dada","mama","kaka"]
info=[9850,6038,9785]
for nm,inf in zip(name,info):
    print(nm,inf)
    
#use of zip function with mis match list
name=["dada","mama","kaka","baba"]
info=[9850,6038,9785]
for nm,inf in zip(name,info):
    print(nm,inf)
#it will not display excess mismatch item in name
from itertools import zip_longest
name=["dada","mama","kaka","baba"]
info=[9850,6038,9785]
for nm,inf in zip_longest(name,info):
    print(nm,inf)
    
    
#use of fill value instead none
from itertools import zip_longest
name=["dada","mama","kaka","baba"]
info=[9850,6038,9785]
for nm,inf in zip_longest(name,info,fillvalue=0):
    print(nm,inf)
    

    
#use of all(),if all values are true then it will
#produce output
lst=[2,3,-6,8,9]#values must be non zero,+ve or-ve
if all(lst):
    print("all values are true")
else:
    print("there are null values")
#####################################
lst=[2,3,0,8,9]
if all(lst):
     print("all values are true")
else:
     print("there are null values")
   


#use of any if any one non zero value

lst=[0,0,0,-8,0]
if any(lst):
    print("it has some non zero value")
else:
    print("all values are null in the list")
#########################################
lst=[0,0,0,0,0]
if any(lst):
    print("it has some non zero value")
else:
    print("all values are null in the list")
    
    
    
#count()
from itertools import count
counter=count()
print(next(counter))
print(next(counter))
print(next(counter))


#now let us start from 1
from itertools import count
counter=count(start=1)
print(next(counter))
print(next(counter))
print(next(counter))




#cycle()
#suppose you have repeated task to be done
import itertools

instructions=("eat","code","sleep")
for instructions in itertools.cycle(instructions):
    print(instructions)
    
    
    
#repeat()
from itertools import repeat
for msg in repeat("keep patience",times=3):
    print(msg)
    
    
    
#combinations()
from itertools import combinations
players=["john","jani","janardhan"]
for i in combinations(players,2):
    print(i)
    
    
#permutations
from itertools import permutations
players=["john","jani","janardhan"]
for seat in permutations(players,2):
    print(seat)


#product
from itertools import product
team_a=["rohit","pandya","bumrah"]
team_b=["virat","manish","sami"]
for pair in product(team_a,team_b):
    print(pair)
    
    
    
    
#filter
age=[27,17,21,19]
adults=filter(lambda age:age>=18,age)
print([age for age in adults])



#assingment operations
#this will only create a new variable with the same refrence
list_a=[1,2,3,4,5]
list_b=list_a

list_a[0]=-10
print(list_a)
print(list_b)




#shallow copy
#one level deep
#use copy.copy()
import copy
list_a=[1,2,3,4,5]
list_b=copy.copy(list_a)

#not affects the other list
list_b[0]=-10
print(list_a)
print(list_b)
###########################
import copy
list_a=[[1,2,3,4,5],[6,7,8,9,10]]
list_b=copy.copy(list_a)
#affects the other
list_a[0][0]=-10
print(list_a)
print(list_b)



#deep copy
#use copy.deep.copy()
import copy
list_a=[[1,2,3,4,5],[6,7,8,9,10]]
list_b=copy.deepcopy(list_a)
#not affects the other
list_a[0][0]=-10
print(list_a)
print(list_b)




#shallow and deep copy
old_list=[[1,2,3],[4,5,6],[7,8,"a"]]
new_list=old_list
new_list[2][2]=9

print("old list:",old_list)
print("id of old list:",id(old_list))
print("new list:",new_list)
print("id of new list:",id(new_list))





#unpacking of dictionary 
friends={
    
    
    "dale":9850,
    "male":6032
    
    
    }

contacts={
    
    
    "dada":8530,
    "mama":5286
    
    }  
contacts.update(friends)
print(contacts)




#pipe operator
friends={"satish":99021,
         "ram":97603}

sham={"sham":85305}

all_friends=friends|sham
print(all_friends)





#
num=0
def change():
    num=1
change()
print(num)




