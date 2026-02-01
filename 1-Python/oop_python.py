#blueprint for creating objects
class circle:
    def __init__(self,x,y,r):
        self.x=x
        self.y=y
        self.r=r
        
    def circumference(self):
        return 2*3.14*self.r
    
    def area(self):
        return 3.14*self.r*self.r
    
#creating an object of circle
a_circle=circle(2.0,2.0,1.0)
b_circle=circle(3.0,3.0,2.0)    

#acessing data and methods
print("radius:",a_circle.r)
print("circumference:",a_circle.circumference())
print("area:",a_circle.area())

#accessing data and methods
print("radius:",b_circle.r)
print("circumference:",b_circle.circumference())
print("area:",b_circle.area())






#encapsulation
class circle:
    def __init__(self,x,y,r):
        self.__x=x
        self.__y=y
        self.__r=r
        
    def get_radius(self):
        return self.__r
    
    def set_radius(self,r):
        if r>0:
            self.__r=r
        else:
            print("invalid radius")
            
c2=circle(1,1,3)
print(c2.get_radius())
c2.set_radius(10)





#inheritence
#base class
class circle:
    def __init__(self,x,y,r):
        self.__x=x
        self.__y=y
        self.__r=r
        
    def area(self):
        return 3.14*self.r**2
    
    def cicumference(self):
        return 2*3.14*self.r
    
    def display_info(self):
        print(f"center:({self.x}, {self.y}, radius: {self.r}")
        print(f"area: {self.area():2f}")
        print(f"circumference: {self.cicumference():2f}")

#derives class
class coloredcircle(circle):
    def __init__(self,x,y,r,color):
        super().__init__(x,y,r)
        self.color=color
        
    #overriding the display_info method
    def display_info(self):
        super().display_info()
        print(f"color: {self.color}")

c1=coloredcircle(0, 0, 5, "red")
c1.display_info()
        






#polymorphism
class circle:
    def area(self):
        return"calculating area of circle"

class square:
    def area(self):
        return"calculating area of square"

shapes=[circle(),square()]
for shape in shapes:
    print(shape.area())
    
    



#encapsulation
class bankaccount:
    def __init__(self,balance):
        self.__balance=balance #private attribute
        
    def deposit(self,amount):
        if amount>0:
            self.__balance+=amount
            
    def get_balance(self):
        return self.__balance

ba1=bankaccount(300)
print(ba1.get_balance())
ba1.deposit(100)
#data is protected; direct access is avoided
print(ba1.get_balance())





#inheritance
class animal:
    def speak(self):
        print ('some sound')
        
class dog(animal):
    def speak(self):
        print ('bark')
        
d=dog()
d.speak()





#polymorphism
class cat:
    def speak(self):
        print("meow")

animals=[dog(),cat()]
for animal in animals:
    animal.speak()
    
    




#abstarction
from abc import ABC, abstractmethod

class shape(ABC):
    @abstractmethod
    def area(self):
        pass

class circle(shape):
    def __init__(self,radius):
        self.radius=radius
        
    def area(self):
        return 3.14*self.radius*self.radius

circle1=circle(5)
print("area of cirle:",circle1.area())






    
    
    
        