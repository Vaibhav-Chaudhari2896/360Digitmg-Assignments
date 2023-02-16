########################## ###################################
##### Basic Data Types
# Integer -----> int ----> Whole numbers such as 3,300,200
# Floating Point ---> float ---> numbers with decimal point 3.56,2.45,76.87
# string ---> str ---> ordered sequence of characters
# lists ---> list ---> ordered sequence of objects
#  Dictionaries ---> dict ---> unordered key value pairs {'mykey':'value'}
# Tuples --> tup --> ordered immutable sequence of objects
# sets --> set ---> unorderd collections of unique objects
#  Boolean --> bool --> logical values indicating True or False
# Data type

#Integer

print(1)

print(2+4)  #addition

print(12*3) # Multiplication 
print(12/3)


# Float

print(11.5)

x = 55.45

print(x)

type(x)


# Complex Number

print(2+3j)

type(2+3j)

# Boolean Type

type(True)

type(False)                                                                                
  
#b=3  
# Variable Assignment Rules
# Names cannot start with a number
# there can be no space in the name , use _ instead ex :- name_last
# can't use any of these symbols  "",<>,\,|,?,!,@,#,%,$,&,*,~, +
# It's considered best practice to use lowe case names as variable assignment
# Avoid using words that have special meaning in Pyhton like "list"
                                     
#######Strings###############
##############################################################
## Strings are sequence of characters using the syntax of either single quote
# or double quote, because strings are ordered sequence it means we can use indexing
# and slicing to grab sub sections of the string, Indexing notation uses[]
# these actions uses [] and a number index to indicate positons of what you
# wish to grab
# this is a ordered sequence of characters, then we can index it and slice it

# multiline string uses triple quote or a tripe single quote
a = """ This data has three types of flower classes: 
    Setosa, Versicolour, and Virginica. The dataset is available in the 
    scikit-learn library, or you can also download it from the UCI Machine 
    Learning Library."""
print(a)
type(a)
len(a)
# syntax of indexing of strings

#  [start:stop:step] this is called as slicing of strings

# Start is a numerical index for slice start
# stop is the index you go upto(but not included)
# step is the jump you take

#Accesing Value in Strings   

Name = "Aditya"
  
print(Name[0])

print(Name[3])

print(Name[1:4])

print(Name[-1])

print(Name[-3:-1])
print(len(Name))
    
#Update Strings    
var1 = 'Hello World!  '
print('Hello')
print ("Updated String :- ", var1 + 'Python')
print ("Updated String :- ", var1[:6] + 'Python')

print("hello \nworld") # \n used to get new line

# String Formating 
print("My name is %s and weight is %d kgs!" % ('Nikhil', 20.5))


#Ex :1 

Name = input("Enter your name: ")
type(Name)
Weight = int(input("Enter your Weight: "))
type(Weight)

print("My name is %s and my Weight is %d"%(Name,Weight))

print('This is a string {}'.format('inserted'))

print("My name is {} and my Weight is {}".format(Name,Weight))

print("My name is {1} and my Weight is {0}".format(Name,Weight))

print('the output is {r}'.format(r = Weight))


print(f'{Name} is {Weight} kgs') # fstrings

type(Weight)

#Triple Quotes


Statement = """my name is "Bahubali" and my age is "25".
I stay in Maahishmathi"""

type(Statement)


Name = "nikhil"
type(Name)
print(Name)
# To make first letter capital
print(Name.capitalize())

# To make string at center in given spaces
Name.center(50)
len(Name.center(50))

#count() method returns the number of occurrences of the substring in the given string.
string = "nikhil is trainer"

substring = "i"

count = string.count(substring)

print("The count is:", count)


#Count number of occurrences of a given substring using start and end
# define string
string = "Sharat is a trainer"
substring = "i"

# count after first 'i' and before the last 'i'
#len(string)
count = string.count(substring,8,25)

# print count
print("The count is:", count)


##Returns true if string has at least 1 character and all characters are alphanumeric and false otherwise.

Num = 'thishi34'  # No space in this string
print(Num.isalnum())

Num = "this is string examplehi 123";
Num.isalnum()

#This method returns true if all characters in the string are alphabetic and there is at least one character, false otherwise.
Num = "thisis";  # No space & digit in this string
Num.isalpha()

Num = "this is string example0909090!!!";
Num.isalpha()

#This method returns true if all characters in the string are digits and there is at least one character, false otherwise.

Num = "123456";  # Only digit in this string
Num.isdigit()

Num = "this is string example!!!";
Num.isdigit()

#his method returns a copy of the string in which all case-based characters have been lowercased.
Num = "THIS IS STRING EXAMPLE!!!";
Num.lower()

#his method returns a copy of the string in which all case-based characters have been Uppercase.
Num = "this is string example!!!";

Num.upper()


#The following example shows the usage of replace() method.

reply = "it is string example!!! is really a string"

print(reply.replace("is", "was"))

print(reply.replace("is", "was", 1))




#The following example shows the usage of split() method.
split1 = "Line1-abcdef Line2-abc \nLine4-abcd"
print(split1)

print(split1.split( ))

print(split1.split(' ', 1 ))

type(split1.split())




###############################   List   ################################

# A list data type is given in square brackets and each element is separated by comma.
 
list1 = ['Sharat', '360DigiTMG', 2013, 2018]
list1 =['ssss',5,5.5,40+55j,True]
list1
#Access values in the variable using index numbers
  
print(list1[0])

print(list1[3])

print(list1[1:4])


#functions in list data type

# Append : add  new element to the existing list

aList = [123, 'xyz', 'zara', 'abc']

aList[2]=55+40j

print(aList)

aList.append( 2009 )
print(aList)



#Pop : remove the elment from existing list
print (aList.pop())

print(aList)
# pop the element using index number
print (aList.pop(2))
aList

#Insert: insert a value using index number 
aList = [123, 'xyz', 'tommy', 'abc', 123]

aList.insert( 3, 2009)
print (aList)


#Extend: append vs extend  

#append
aList = [123, 'xyz', 'tommy', 'abc', 123]
bList = [2009, 'beneli']

aList.append(bList)

print(aList)

#extend
aList = [123, 'xyz', 'tommy', 'abc', 123]
bList = [2009, 'beneli']
aList.extend(bList)
print(aList)

bList.append(aList)
bList
bList.extend(aList)
bList

#Reverse: to reverse the given list 
aList = [123, 'xyz', 'tommy', 'abc', 123]
aList.reverse()
print(aList)


#Sort: sort the given list from ascending or descending

blist = [8,99,45,33]
blist.sort()
print(blist)

#count: count the value in given list of elements

aList = [123, 'xyz', 'zara', 'abc', 123, "zara"]
print(aList.count(123))

#index
print(aList.index('zara'))


list1=[1,2,3]
list2=[1,2,3]

1+2
'a'+'b'

list3=list1 + list2

print(list3)

print(list1*3)

dir(list)
##################################### Tuples ####################################

## Create a tuple dataset
tup1 = ('Street triple','Detona','Beneli', 8055)
tup2 = (1, 2, 3, 4, 5 )


### Create a empty tuple 
tup1 = ()
tup1
#Create a single tuple
tup1 = (50,)
type(tup1)

#Accessing Values in Tuples
tup1 = ('Street triple','Detona','Beneli', 8055,2+3j,False)
print(tup1[0])

tup2 = (11, 12, 13, 14, 15, 16, 17 )
print(tup2[1:5])

#Updating Tuples
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')

#Tules are immutable
tup1[1]=40

s='sam'
s[0]
s[0]='z'

# So,create a new tuple as follows
tup3 = tup1 + tup2
print(tup3)

#index, count
tup=(1,2,3,4,1,2)
tup.index(1)

tup.count(2)

#Delete Tuple Elements
tup = ('Street triple','Detona','Beneli', 8055)
print(tup)

# to delete the given tuple 
del(tup)
tup

help(list)

############################## Dictionary #################################

#Accessing Values in Dictionary
dict1 = {'Name': 'Sharat', 'Age': 40, 'bike': 'Beneli'}

print(dict1)

print(dict1['Name'])

print(dict1['Age'])   

aList[1]=40

##Updating Dictionary
dict1 = {'Name': 'Nikhil', 'Age': 25, 'bike': 'Beneli'}

# update existing entry
dict1['Age'] = 27

# Add new entry
dict1['School'] = "DPS School"

dict1['sal'] = 50000


print(dict1['Age'])
print(dict1['School'])
print(dict1['sal'])

dict1.keys()
dict1.values()
dict1.items()

#Delete Dictionary Elements
dict1 = {'Name': 'Sharat', 'Age': 40, 'bike': 'Beneli'}

# remove entry with key 'Name'
del(dict1['Name'])
dict1
# remove all entries in given dictionary
dict1.clear()
dict1
# delete entire dictionary
del(dict1) ;       

print(dict1['Age'])
print(dict1['School'])


d = {'k':123,'k2':[8,9,10],'k3':{'inside_key':100}}
d
d['k']

d['k2']

d['k2'][1]

d['k3']['inside_key']



############################## Sets #################################
#### sets are unordered collection of unique elements and do not allow 
# duplicate values

# A normal Set
normal_set={1,2,3}
normal_set = set(["a","b","c","f"]) 
normal_set = {'a','b','c','d'}
normal_set.add("d") 
   
print(normal_set) 
  

my_list = [1,1,2,3,4,4,5,5,6,7]
my_set=set(my_list)
my_set

set1 = {1,2,3}
set2 ={1,9,0}

lis = set1.intersection(set2)
lis
#Find common elements of set and list

intersection_as_list = list(lis)

print(intersection_as_list)

# Python3 program for union() function

set1 = {2, 4, 5, 6}
set2 = {4, 6, 7, 8}
set3 = {7, 8, 9, 10}

# union of two sets
print("set1 U set2 : ", set1.union(set2))

# union of three sets
print("set1 U set2 U set3 :", set1.union(set2, set3))



####################Operators#########################3

#Arithmetic operators

#consider
a = 10
b = 20

#Addition
a + b

#subtraction
a - b

#Multiplication
a * b

#division 
a/b

#Floor Division
a//b


#Modules
b%a


#exponential
a**b 



#Comparision Operator

#It gives in bool values

a == b
 
a != b
 
a > b
 
a < b

a >= b

a <= b

#Assignment Operators

c = a+b

c

c += b  #its nothing but c = c+b
c
c -= b
c
c *= b
c
c /= b
c
c %= b
c
c **= b
c
c //= b
c

x=1;y=2
x/y

x=1;y=2
x//y

x=-1;y=2
x//y #(gives output less than quotient)

#Membership Operators
# It will check that left value is a member of right value or not 

"y" in "Python"

"l" in "Python"

"p" in "Python"

"P" not in "python"

#Identity Operators
# It Check will check that left value is equal to the right value or not

"y" is "Python"

1 is 1

2 is 1

"python" is "Python"

1 is 0

1 is not 1

"hi hello"  is not  "hello hi"



################## Variables ##################### 
#Variable name should start with a letter 
#It should contain onl alpha numerics
#It should not contain any other symbols except _

#Assigning Integer values to a variable 
car = 10
print(cars)
type(cars)

car 2s=20
car_1=20
#Variable name should start with letter
#It can contain alphanumerics and underscore only
car 1=15

#Assigning decimal values to a variable 
wt = 60.25
print(wt)
type(wt)

#Assigning string value to variable    

new_car = 'cars'

String_Name = "Python string"

# Multiple Assignment

a = b = c = 34

print(a)

print(b)

print(c)

# Integer values

a,b,c,d,e = 0,2,5,7,8


print(d)
print(a)
print(b)
print(e)
print(c)

new,old="iphone","MI"

print(new)
print(old)


# User assigning the value to variable using input function 

num_1 = input("hi, Enter a value =  ")

print ("you have entered ", num_1)

type(num_1)

# Restricting user input using int() function
  
age = int(input("enter the age = "))

print("your age is: ", age)

type(age)

# Restricting user input using int() and Float() function
num_1 = float(input("Enter a 1st value = "))
num_2 = int(input("Enter a 2nd value = "))
results = num_1+num_2
print("final result", results)
type(results)

# Restricting user input using Float() function, eval() helps us to evaluate according to user input
x = eval(input("enter the 1st value = "))
x
y = eval(input("enter the 2nd value = "))
results = x+y
print("final results = ",results)
type(results)



################################## Conditional Statement ####################################


# Example:1 - simple if and else conditions when you have  two  conditions
is_male = True

if is_male:
    print("your male")
else:
    print("your female")

    
# Example: 2 - if you have more than two condition use elif 

is_male = True
is_tall = False 

if is_male and is_tall:              ## and & not operator 
    print("you are tall male")
elif is_male and not(is_tall):
    print("you are short male")
elif not(is_male) and is_tall:
    print("you are not male but tall")
else:
    print("you are not male and not tall")
    
    
if is_male or is_tall:
  print(" you are male or a tall or both") ## or operator 
  
  
#Example:  3 - conditional inside a conditional 
### Nested if Statement

score = 500
money = 2000
age = 65

if score > 100:
    print("You got good points")
    
    if money >= 5000:
        print("you win")
        
        if age >= 30:
            print("You win in middle age")
        else:
            print("You are win in young age")
    else:
        print("you have a high points but you do not have enough money")
        
else:
    print("your loser")
 

#Example: 4
    
name = "human"
animalName = "dog"

if name == "animal":
    print("Name Entered is Animal")
    if animalName == "dog":
        print("valid Animal")
    else:
        print("animalName invalid")
else:
    print("the name entered is not valied")
    print ("your entered name is not a animal")
    
 


############################### Loops #########################

   
############### While Loop

#Example for 'While loop'

i=10
while i>0:
    print(i)
    i=i-1

#GAME
random_number = int(20*5+5-10)

guess = 0

while guess != random_number:
    guess = int(input("New Number: "))
    if guess > 0:
        if guess > random_number:
            print("number is too large")
        elif guess < random_number:
            print("number is too small")
    else:
        print("sorry that you are giveup!")
        break
else:
    print("Congratulations. YOU WON!")
    
    
            

################ for Loop

#Example for 'for loop' 

#Example : 1
    
snacks = ['pizza','burger','shawarma','franky']

for snack in snacks:
    print("current snack: ", snack)
    
     


#Example: 2

for i in range(10):
    print(i)
        
for i in range(1,15):
    print(i)
    
for i in range(1,20,2):
    print(i)

#Ex 3
stock_prices = [['Apple',300],["samsung",400],["nokia",100]]

for items in stock_prices:
    print(items)
     
     
d = {'k1':1,'k2':2,'k3':3}

for item in d.items():
    print(item)

for key, value in d.items():
    print(key)
    print(value)   
    
       
#Nested Loops
#Nested 'for in for' and for in while will be asignments problems                       
#Nested while in for 

travelling  = input("yes or no" )

while travelling == 'yes':
    
    num = eval(input("number of people travelling: " ))
    
    for num in range(1,num+1):
        name = input("Name: ")
        
        age = input("Age: ")
        
        sex= input("Male or Female: ")
        
        print(name)
        
        print(age)
        
        print(sex)
        
    travelling = input ("Oops! forgot someone")
  
    
  
    
############################### define function ##############################

#Example for advantage of functions 
    
#Imagine there are 1000 of code is created based on requirement without using functions 
# If same requirement is repeated in the same code as given below with reference of line numbers
print("Hello Function.Sharat?") #10
print("Hello Function.Sharat?")#14
print("Hello Function.Sharat?")#40
print("Hello Function.Sharat?")#67

#how to create function? 
def hello_func():
    i=4
    while i>0:
        print("Hello Function.Sharat")
        i=i-1

#call the function as and when required than writing whole code again 
hello_func()#10


#Advantages about functions - optimize code, good performance, fast, easy to access etc 
  
#simple function 
def hello(name,age,sal):                 
    print("hi",name,"your age:",age,"your salary:",sal)

hello("Sharat",25,50000)
 
#add two numbers using return value 
def add(x,y,z):                   
    return (x-y+z)

add(10,20,50)

#cube of n value 
def cube(num):                  
    return (num*num*num)

cube(3) 


def even_check(number):
    if(number%2 == 0):
        print('given number is even')
    else:
        print("given number is odd")
   

even_check(5)


# not defined any value but just defined function 

def odd(list1):
    for num in list1:
        if num%2==0:
            pass
        else:
            print(num)
    
numbers=[1,2,3,4,5,6]
odd(numbers)

########
def myfunc(a,b,c =0,d =0, e = 0):
    # returns 5% of the sum of a and b
    return sum((a,b,c,d,e))*0.05

myfunc(10,20,30,40,50)
myfunc(40,60)

# Defining the lambda function , map ,reduce

s = lambda x: x * x
s
s(12)

# Functional Programming Packages
val = [1, 2, 3, 4, 5, 6]

list(map(lambda x: x * 2, val))


    
############################# Modules ################################

# Modules are also called as packages 
# how to call package is : Import package_name
    
#Example:1

#maths related package
import math

math.sqrt(16)

math.pow(2,5)

dir(math)



#Example:2

#calendar related package 
import calendar 

cal = calendar.month(2018,1)

print(cal)


calendar.isleap(2018)

calendar.isleap(2020)

calendar.isleap()

dir(calendar)



#########3 Python Packages for Data Analysis ################

import numpy as np

# A list of elements in variable 'x'

x = [1,2,3,4,5]

# how to multiply the list values with 2
x*2 #  provides dual list

# Numpy array will help access the values

y = np.array(x)

type(y)
y*2
y>2
y[y>2]

# operator ,Description general information only 
np.array([1,2,3]) # 1d array
np.array([(1,2,3),(4,5,6)]) # 2d array
np.arange(start,stop,step) # range array
np.linspace(0,2,9) # add evenly spaced values btw interval to aray of length
np.zeros((1,2)) # create array filled with zeros
np.ones((1,2)) # creates an array filled with ones
np.random.random((5,5)) # creates random array
np.empty((2,2)) #creates an empty array
array.shape # gives information on dimensions
len(array) 
array.ndim # number of  array dimension
array.dtype # Data Type


# Numpy matrics 
#Ex:1
a = np.matrix('1 2; 3 4')
type(a)
a

#Ex:2
b = np.matrix([[1, 2], [3, 4]])
b

b.shape


# create a sequence of integers with specific values

f = np.arange(0,50,5)
f
#np.arange vs range
f=range(0,50,5)
f
f=list(range(0,50,5))
f

#reshaping the numpy array
ary = np.array([[2,3,4,5],[6,8,4,7],[9,5,1,3]])
ary
ary.shape
ary[1]
ary[0,3]
newary = ary.reshape(6,2)
newary.shape
newary

#flatten
ar2 = newary.flatten()
ar2

#sort
ary = np.array([[2,3,4,5],[6,8,4,7],[9,5,1,3]])
ary
ary.sort()
ary

#axis
ary = np.array([[2,3,4,5],[6,8,4,7],[9,5,1,3]])
ary
ary2=np.delete(ary,1,axis=1)
ary2

ary = np.array([[2,3,4,5],[6,8,4,7],[9,5,1,3]])
ary
ary3=np.delete(ary,1,axis=0)
ary3
dir(np)

# Pandas
import pandas as pd # importing pandas = > useful for creating dataframes


x1 = [1, 2, 3, 4,4] # list format 
x2 = [10, 11, 12,13,14]  # list format 

x3 = list(range(5))

x1, x2,x3
# Creating a data frame using explicits lists
X = pd.DataFrame(columns = ["X1","X2","X3"]) 
X

X["X1"] = x1 # Converting list format into pandas series format
X["X2"] = x2 # Converting list format into pandas series format
X["X3"] = x3
X

X["X1"] = pd.Series(x1)  # Converting list format into pandas series format
X["X2"] = pd.Series(x2) # Converting list format into pandas series format
X["X3"] = pd.Series(x3)
X

# Creating a data frame using explicits lists
X_new = pd.DataFrame(columns= ['X1','X2','X3'],index = [101,102,103,104,105])
X_new

X_new["X1"] = x1 
X_new["X2"] = x2 
X_new["X3"] = x3
X_new
# accessing columns using "." (dot) operation
X.X1
# accessing columns alternative way
X["X1"]

# Accessing multiple columns : giving column names as input in list format
X[["X1","X2"]]

# Accessing elements using ".iloc" : accessing each cell by row and column 
# index values
X.iloc[0:3,1]

X.iloc[:,:] # to get entire data frame 
X.loc[0:2,["X1","X2"]]

#Stattistics
X
X['X3'].mean()
X['X3'].median()
X['X3'].mode()

X.describe()
# Merge operation using pandas 

df1 = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12],})
df2 = pd.DataFrame({"X1":[1,2,3,4],"X3":[14,18,112,15],})
df1,df2
merge = pd.merge(df1,df2, on = "X1") # merge function
merge


# Replace index name
df = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12]})
df

df.set_index("X1", inplace = True) #Assiging index names using column names
df
# Change the column names 
df = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12],})

df  = df.rename(columns = {"X3":"X4"}) #Change column names

print(df)

# Concatenation

df1 = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12],},index = {2000,2001,2002})
df2 = pd.DataFrame({"X1":[4,5,6],"X2":[14,16,18],},index = {2003,2004,2005})

Concatenate = pd.concat([df1,df2])

print(Concatenate)

x1 = [1, 2, 3, 4,5,np.nan] 
x2 = [np.nan, 11, 12,100,np.nan,200] 
df=pd.DataFrame()
df['x1']=x1
df['x2']=x2
df

#finding null values
df.isna().sum()
df.dropna()
# another way to create dataframe
df = pd.DataFrame(
    {"a" : [4,5,6],
     "b" : [7,8,9],           ## Dictionary Key value pairs                                                          
     "c" : [10,11,12]},
    index = [1,2,3])
# another way to create dataframe
df = pd.DataFrame(
    [[4,7,10],
     [5,8,11],
     [6,9,12]],
    index = [1,2,3],
    columns = ['a','b','c'])
df
a = pd.Series([50,40,34,30,22,28,17,19,20,13,9,15,10,7,3])
len(a)
a.plot()
a.plot(figsize =(8,6),
       color = 'green',title = 'line plot',fontsize = 12)
b = pd.Series([45,22,12,9,20,34,28,19,26,38,41,24,14,32])
len(b)
c = pd.Series([25,38,33,38,23,12,30,37,34,22,16,24,12,9])
len(c)
d = pd.DataFrame({'a':a,'b':b,'c':c})
d
d.plot.area(figsize = (9,6),title = 'Area plot')
d.plot.area(alpha= 0.4, color = ['coral','purple','lightgreen'],figsize = (8,6),fontsize = 12)

##############3 reading extrnal file
import pandas as pd
help(pd.read_csv)
# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv("C:\\Users\\pc\\Downloads\\Python Codes\\Python Codes\\mba.csv")

type(mba) # pandas data frame
mba

mba.groupby('gmat').count()

mba.groupby('gmat').count()['datasrno']

list(mba.groupby('gmat'))

mba.groupby('gmat').sum().sort_values(by='workex')

mba.groupby('gmat').sum().sort_values(by='workex',ascending=False)

###########Matplotlib package
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# We have considered MBA dataset and executed few important vizualizations


#Histogram
plt.hist(mba['gmat'])
plt.xlabel('gmat')


#Boxplot
plt.boxplot(mba['gmat']);plt.ylabel("GMAT")   # for vertical

help(plt.boxplot)

#Barplot
import numpy as np

plt.bar(height = mba["gmat"], x = np.arange(773)) # initializing the parameter
mba['gmat']

#line plot
plt.plot(mba['workex'],mba['gmat'])
plt.xlabel('workex')
plt.ylabel('gmat')

#plt.plot([1,2,3,4,5],[5,2,1,25,2])

#scatter plot
plt.scatter(mba['workex'],mba['gmat'])

#Import the dataset of cars
mtcars = pd.read_csv("C:\\Users\\pc\\Downloads\\Python Codes\\Python Codes\\mtcars.csv")


#Line Chart
mtcars.mpg.groupby(mtcars.gear).plot(kind="line")
plt.legend()

mtcars.gear.value_counts()

# Pie chart
mtcars.gear.value_counts().plot(kind="pie")

# Area Chart
mtcars.mpg.plot(kind='area') 

mtcars.mpg.plot(kind='line')
#Scatter Plot
plt.plot(np.arange(32),mtcars.mpg,"ro")






