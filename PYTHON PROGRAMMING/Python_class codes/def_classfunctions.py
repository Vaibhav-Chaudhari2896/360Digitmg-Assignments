# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:00:14 2020

@author: abdul
"""
# custom define functions
# creating a custom functions in Python requires def keyword
# proper structure, and indentation.

def name_of_function(name):
    """
        Documenet string
        explains the function usage
    """
    print("hello " + name)

name_of_function("Jameel")

def add_function(num1,num2):
    return num1+num2

result = add_function(2,2)
result
print(result)
# with return I can save the output but not with print function

def say_hello(name):
    print(f'Hello {name}')

say_hello("jameel")

def sum_numbers(num1,num2):

    return num1 + num2

sum_numbers('10','20')

# functions with logics

def even_check(number):
    result =  number%2 == 0
    return result
    #return number %2 == 0

even_check(20)
#True
even_check(21)
# False

# return true if any number is even inside a list

def check_even_list(num_list):
    # return all the even number in a list

    # place holder variables

    #even_number = []

    for number in num_list:
        if number % 2 ==0:
            #even_number.append(number)
            return True
        else:
            pass
    #return even_number
    return False


check_even_list([1,3,5])

check_even_list([1,2,5])

check_even_list([1,2,4,6,5,7])

# tuples unpacking with functions

stock_prices = [('Appl',200),("goog",400),("MSFT",100)]

for item in stock_prices:
    print(item)
for ticker, price in stock_prices:
    print(ticker)
    print(price+(0.1*price))

work_hours = [('abby',100),('billu',400),('cassie',800)]

def employee_check(work_hours):

    current_max = 0
    employee_of_month = ""

    for employee, hours in work_hours:
        if hours> current_max:
            current_max = hours
            employee_of_month = employee
        else:
            pass


    #return
    return(employee_of_month,current_max)

result = employee_check(work_hours)
result
name,hours = employee_check(work_hours)
name
hours

## *args **kwargs  arguments , keyword arguments

def myfunc(a,b,c =0,d =0, e = 0):
    # returns 5% of the sum of a and b
    return sum((a,b,c,d,e))*0.05
myfunc(40,60)
myfunc(40,60,100,3,4)

# a and b are positional arguments
def myfunc(*args):
    print(args)

    return sum(args)*0.05

myfunc(40,60,100,2,34)

# *keyword argumnets
# it builds a dictionary with key value pairs

def myfunc(** kwargs):
    if 'fruit' in kwargs:
        print('my fruit of choice is {}'.format(kwargs['fruit']))
    else:
        print('I did not find any fruit here')

myfunc(fruit = 'apple', veggie = 'lettuce')


def myfunc(*args, **kwargs):

    print('I would like  {} {}'.format(args[0],kwargs['food']))

myfunc(10,20,30,fruit = 'orange', food = 'eggs',animals = 'dog')


#test
# check for even number using def and *args

#test 2
# define a function that take in a string and returns a matching string
# where every even letter is upper case and every odd number is lowe case

# lambda expressions ,Map and filter

def square(num):
    return num**2

my_nums = [1,2,3,4,5]

for item in map(square,my_nums):
    print (item)
list(map(square,my_nums))

def splicer(mystring):
    if len(mystring)%2 ==0:
        return 'even'
    else:
        return mystring[0]
names = ['Andy','Eve','Sally']
list(map(splicer,names))

# filter functions returns either true or false

def check_even(num):
    return num%2 == 0

mynums = [1,2,3,4,5,6,8]
#filters based on the def condition
filter(check_even,mynums)
for n in filter(check_even,mynums):
    print(n)

def square(num):
    #result = num**2
    return result
square(5)

def square(num): return num**2
# lambda functions is also called as anonymous function
# we don't name them
square = lambda num: num**2
# using map lambda

list(map(lambda num: num**2,mynums))

filter(lambda num:num%2 ==0,mynums)
# reverse a string
list(map(lambda name:name[::-1],names))
#########################################################


#object oreiented programmin allows users to create their own
#objects that have methods and attributes
# .append(), .extend()
# these methods act as funcion that uses the informatio about the object as well as
# the object itself to return results or change the current object
# for example this includes appending an element to the list getting the length of a list or tuple
# syntax

class NameOfClass():
# init method, self keyword, parameters
    def __init__(self,param1,param2):
        self.param1 = param1
        self.param2 = param2
# this method is connected to the class
    def some_method(self):
        # perform some action
        print(self.param1)
        print(self.param2)
# user defined objects using class

#class is a blue print that defines the nature of the object
# from class we can then define the instance of the object and
# instance is a specific object created form specific class
# for class we use CamelCasing
class Sample():
    pass
# we have just created a simple class
# instance of the class
my_sample = Sample()
type(my_sample)

#mylits = [2,3,4,5]
#mylits
class Dog():
    # class object attributes
    # same for any instance of a class
    # we donot use self keyword here
    species = 'mammal'
#__init__ method acts as a constructor
    def __init__(self,mybreed, name, spots):
#self.attribute
        self.breed = mybreed
        self.name = name
        # expect booleab True/false
        self.spots = spots

    # operations/Actions -> methods
    # user should define numnber
    def bark(self,number):
        print('Woof! My name is {} and the number is {}'.format(self.name,number))

my_dog = Dog(mybreed = 'Lab',name = 'Sammy', spots = False)

type(my_dog)

my_dog.breed
my_dog.name
my_dog.spots
my_dog.species

my_dog.bark(2)

class Circle():
    # class object attribute
    pi = 3.14

    def __init__(self, radius = 1):
        self.radius = radius
        self.area = radius*radius*Circle.pi

    # method
    def get_circumference(self):
        return self.radius*Circle.pi*2
# change radius
my_circle  = Circle(40)

my_circle.pi
my_circle.radius
my_circle.get_circumference()
my_circle.area

# Inheritence and Polymorphism
# inheritence is basically used to form new classess using classess already defined
# base class

class Animal():

    def __init__(self):
        print("Animal Created")

    def who_am_i(self):
        print("I am an animal")
    def eat(self):
        print("I am eating")

class Dog(Animal):
    def __init__(self):
        Animal.__init__(self)
        print("Dog created")
# overwriting our base class methods
    def who_i_am(self):
        print("I am dog")

mydog = Dog()
mydog.eat()

#Polymorphism
#polyorphism refers to the way in which different object
#classess share same method name

class Dog():
    def __init__(self,name):
        self.name = name
    def speak(self):
        return self.name + ' Says woof!'
class Cat():

    def __init__(self,name):
        self.name = name
    def speak(self):
        return self.name + " Says meow!"

nia = Dog("nia")

felix = Cat("felix")

print(nia.speak())

print(felix.speak())

for pet in [nia, felix]:
    print(type(pet))
    print(pet.speak())

# speak is same methos used in different class which is called polymorphism
def pet_speak(pet):
    print(pet.speak())
pet_speak(nia)

pet_speak(felix)


class Book():

    def __init__(self, title, author, pages):

        self.title = title
        self.author = author
        self.pages = pages

    # return string representation of user defined objects
    def __str__(self):
        return f'{self.title} by {self.author} with pages {self.pages}'
    # to get of method
    def __len__(self):
        return self.pages
    # to delete the variabe
    def __del__(self):
        print("A book has been deleted")
b = Book('Python course','jamees',200)
print(b)
len(b)
# gets string information
str(b)
# delete the book variable its different form del in built function
del(b)

#


# task
# create a line class method to accept coordinates (coord1, coord2)as a pair of tuples and return the slope and distance of the line

# slope= y2-y1/x2-x1

class Line:

    def __init__(self,coor1,coor2):

        self.coor1 = coor1
        self.coor2 = coor2

    def distance(self):
        x1,y1 = self.coor1
        x2,y2 = self.coor2


        return ((x2-x2)**2 + (y2-y1)**2)**0.5
    def slope(self):
        x1,y1 = self.coor1
        x2,y2 = self.coor2

        return (y2-y1)/(x2-x1)

c1 = (3,2)
c2 = (8,10)

myline = Line(c1,c2)
myline.distance()
myline.slope()

# create a cylinder class and then find out voulume and surface area of it
#task 2
class Cylinder:

    def __init__(self,height = 1, radius = 1):

        self.height = height
        self.radius = radius

    def volume(self):

        return self.height *3.14 * (self.radius)**2

    def surface_area(self):

        top =  3.14 * (self.radius**2)

        return (2*top)+ (2*3.14 *self.radius*self.height)

mycyl = Cylinder(2,3)

mycyl.volume()
mycyl.surface_area()
