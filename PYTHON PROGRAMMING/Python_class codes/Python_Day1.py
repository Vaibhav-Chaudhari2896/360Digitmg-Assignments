# Python Numbers(int, float,complex)
a = 5  # integer
type(a)

 A = 2.0 # float
type(A)

AA = 1+2j # complex
type(AA)


##############################################################
## Strings

# this is a ordered sequence of characters, then we can index it and slice it

name = "Data Science"

type(name)

# syntax of indexing of strings

#  [start:stop:step] this ic called as slicing of strings

# Start is a numerical index for slice start
# stop is the index you go upto(but not included)
# step is the jump you take


name= "hello"
print(name)

name[0]
name[4]

# reverse indexing
name[-1]
name[-2]
name[-3]
name[-4]
name[-5]

my_string = "abcdefghijk"
type(my_string) # str
len(my_string) # 26
print(my_string)
# indexing of string

my_string[2:]
my_string[6:]
my_string[:7]
my_string[:8]

my_string[2:2:2]
# reverse the string
my_string[::-1]

print("hello \nworld") # \n used to get new line

print("hello \tworld")
# string immutability
name = "sam"
last_letters = name[1:]
'p'+last_letters # string concatenation

'2'+'3'
x = "Hello World"
len(x)
x.upper() # upper case
x.lower() # lower case
x.split() # splitting string based on  white space
x = 'hi this is a string'

x.split('i') # based on i so removes i
print('hello')
# print formatting with Strings
# .format()
# f-strings( formatted string literals)

# syntax :-
# 'string here {} then also {}'.format('something1', 'something2)

print('This is a string {}'.format('inserted'))
print('The {2} {1} {0}'.format('fox','brown','quick')

# Float Formatting follows value width precision

result = 100/777
result
print('the result was {r}'.format(r = result))

print('the result was {r:10.6f}'.format(r = result))

# fstring literalls
name = 'sam'
age = 5

print(f'{name} is {age} years old')

###############################   List   ################################

# list are ordered sequence of elements that can hold different
# data type objects

list1 = ['Python', 'DataScience', 2013, 2018]
list1
print(list1)
list2 = [1, 2, 3, 4, 5 ]
list3 = ["a", "b", "c", "d"]


python_class = ["Basics", "of", "Python"]


list1 = ['of', 'Basics', 2014, 2018]

print(list1[0])
list1[0]
print(list1[-3])

my_list = ['one','two','three']

my_list[2]
my_list[1:]
# append()
my_list.append("four")
my_list
my_list.extend("five")
# remove the last item of the list
my_list.pop()
my_list.pop(0)
my_list.pop(-1)
my_list

# reverse()
my_list.reverse()
my_list.sort()
list2 = [13, 42, 53, 34, 25, 86, 79 ]
list2.sort()
print(list2)

list2.append(100)

list2.extend('121')
print(list2)
print(list2[0:3])




list1 = ['Python', 'Basics', 2013, 2018]
print(list1[2])
list1[2] = 22334
list1[2] = 8888
list1
list1[2] = 8055
print(list1)
k= 10.5e5
#10.5e2=10.5*pow(10,2)= 10.5*100=1050.0
print(k)
9**2
pow(9,2)
y= 9**2
y
k
k=5
x = 2*k
print(x)
list1[0] = "Basics"
list1
list1 = ['Python', 'Basics', 2013, 2018]
print(list1)
del(list1)
list1
print(list1)


# Append

aList = [123, 'xyz', 'zara', 'abc']
aList.append( 2009 )
aList
aList.append(444)
print(aList)

#Pop # it will the last element of the list

aList.pop()
print(aList)
aList
aList.pop(2)
aList

############Insert

aList.insert(3,2010)
print (aList)

aList.insert(2,"Anjali")
print (aList)

#Extend

aList = [123, 'xyz', 'tommy', 'abc', 123]
bList = [2009, 'beneli']

bList.extend(aList)

aList.append(bList)
aList.extend(bList)


print(aList)

#Reverse

aList.reverse()
print(aList)


#Sort


blist = [8,99,45,33]
blist.sort(reverse = False)
print(blist)

#count

aList = [123, 'xyz', 'zara', 'abc', 123, "zara",1,1,1,1,1,1,1,1,1,1]
aList.count('zara')
print(aList.count(1))


c = 3 +6j
print(type(c))
list1 = ['Python',2018, c ,True,23.56]
list88 = ['Python', 2018, (3+6j), True, 23.56]

list1 = ['Python',2018, (6+2j) ,True,23.56]
list2 = [2300,'Hello',False,0.7888,(3j)]
list1.extend(list2)
print(list1.count(2300))

import statistics

# declaring a simple data-set consisting of real valued
# positive integers.
set1 ={ 2, 3, 3, 4, 4, 4, 5, 5, 6}

statistics.mode(set1)

list1 = [1, 2]
list2 = [1, 3]


list1_as_set = set(list1)
intersection = list1_as_set.intersection(list2)

set1 = {1,2,3}
set2 ={1,9,0}

lis = set1.intersection(set2)
#Find common elements of set and list

intersection_as_list = list(intersection)

print(intersection_as_list)

# Python3 program for union() function

set1 = {2, 4, 5, 6}
set2 = {4, 6, 7, 8}
set3 = {7, 8, 9, 10}

# union of two sets
print("set1 U set2 : ", set1.union(set2))

# union of three sets
print("set1 U set2 U set3 :", set1.union(set2, set3))

##################################### Tuples ####################################

# Tuples are very similar to the lists , however they
# have a key difference which is immutability
# once an element is inside the tuple they cannot be
# changed or reassigned
# list is represented []
# Tuples is represented by ()


## Create a tuple dataset
tup1 = ('Street triple','Detona','Beneli', 8055)
tup2 = (1, 2, 3, 4, 5 )
tup3 = ("a", "b", "c", "d")

type(tup1)

tup1[2]
tup1[-1]

# count , and index
t = ('a','a','a','b','b','c','c','c')

t.count('a') # 3
t.index('a') # 0 very first time it appears
t.index('c')

my_list = [1,2,3]
my_list[0] = 'NEW'
my_list

t_1 = (1,2,3)
type(t_1)
t_1[0] = 'new' # immutability
### Create a empty tuple
tup1 = ()

#Create a single tuple
tup1 = (50)

#Accessing Values in Tuples
tup1 = ('Street triple','Detona','Beneli', 8055)
print(tup1[0])
tup1[0]=99

tup2 = (1, 2, 3, 4, 5, 6, 7 )
print(tup2[1:6])
#tup2.append(tup1)

#Updating Tuples
tup1 = (12, 34.56)
tup2 = ('abc', 'xyz')
#tup1.append(tup2)
"""python"""

tup2

# So,create a new tuple as follows
tup1 = tup1 + tup2
print(tup1)

#Delete Tuple Elements
tup = ('Street triple','Detona','Beneli', 8055)
print (tup)
del(tup)

print ("After deleting tup : ")
print(tup)


#Basic Tuples Operations

#To know length of the tuple
tup = (1,2,3,'Basics','Python')
len(tup)
kk= 'Hello'
len(kk)


#To add two elements
tup2 =(4,5,6)

tup3 = tup+tup2
tup3
tup3*3


tup4 = (' Hi! ')
tup4*3

tup2[2]

tup2[2] = 8

# Membership
2 in (1, 2, 3)
'o' in kk
'u' in 'Fruit'



# Max and min in tuple
tuple1 = (456, 700, 200)
print(max(tuple1))
max(tuple1)

print(min(tuple1))
r='Apple'
max(r)
min(r)

############################## Dictionary #################################

# Dictionaries are unordered mappings for storing objects
# key value pairs , this will allow users to quickly grab
# objects without needing to know about index location
# the representation of your dictionary is done by
# {'key1':'value1','key2':'value2','key3':'value3',.....}

price_lookup = {'apple':2.99,'oranges':1.99,'milk':3.99}

price_lookup['oranges']

d = {'k':123,'k2':[8,9,10],'k3':{'inside_key':100}}
d['k']
d['k2'][2]
d['k3']
d['k3']['inside_key']
d['k4']= 500
d
d.keys()
d.values()
d.items()
d['k']= 'new_value'
d
#Accessing Values in Dictionary
dict1 = {'Name': 'Vinod', 'Age': 25, 'bike': 'Apache'}
print(dict1)
print(dict1['Name'])
print(dict1['Age'])
print(dict1['bike'])

dict1['bike']='Pulsar'
dict1
##Updating Dictionary
dict1 = {'Name': 'Vinod', 'Age': 25, 'bike': 'Beneli'}
dict1['Age'] = 8; # update existing entry

dict1

dict1['Name'] = "Sri Vinod" # Add new entry
dict1[] = "Honda"
dict1["Area"] = "Bangalore"
dict1['ABHI']= 'DS'

dict1['hfsuhg']=666

dict2 = {'Name': ('Vinod','adfwe'),'Age': 25, 'bike': 'Beneli'}
dict2['Name'] = 'SRi','ooo'
dict2['Age']=33434,1243
dict2['bike']='KTM',
dict2
dict2
print(dict1['Age'])
print(dict1['Bike'])
dict1 = {'State': ('Delhi','Mumbai ','Goa'),'Covid-19 Cases':('3Lacs','6lacks','1lak')}
dict1


#Delete Dictionary Elements
dict1 = {'Name': 'Vinod', 'Age': 25, 'bike': 'Apache'}
del(dict1['Name']); # remove entry with key 'Name'
dict1
dict1.clear()    # remove all entries in dict
dict1
del(dict1)        # delete entire dictionary
dict1
############################## Sets #################################

# sets are unordered collection of unique elements
# meaning there can only be one representative of the
# same object

my_set = set()
my_set.add(1)
my_set.add(2)
my_set
my_list = [1,1,1,1,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,7]
set(my_list)

### mississippi
### parallel
### DataScience

set('mississippi')
set('parallel')
set('Datascience')

list1 = ['aaaa','AAAAA',"India",'mississippi','parallel','Datascience','Datascience']
set(list1)

######## Booleans #########
# Booleans are operators that allow you to convey
# True or False  statements

1>2
1==1

#### Python Comparision Operators
### chaining comparision operators with logical operators
# logical operators And, OR, Not

1==1

2!= 3
2>3
2<3
2<=3
3<=3
4>=5
4>=4

# chaining operators

1<2
2>3

1<2>3

('h'=='h') and (3==2)

100==1 or 3==2

1==1
not(1==1)
not 400>5000
400!=5000

################## Python Statements#############

# Python statements are control flow logics or syntax
# remeber use of colons : , and indentation this is crucial
#  Python

## if else statement
# syntax for if else statement

if some condition:
    # execute some code
else:
    # do something else


# if elif else statements

if  some condition:
    # excute some code
elif:
    # do something different
else:
    # do something else

# example

if True:
    print('it''s True')

hungry = False
if hungry == True:
    print("Feedme")
else:
    print("i am not hungry")


loc = 'gym'

if loc == 'Autoshop':
    print('cars are cool')
elif loc == 'Bank':
    print("this is my bank")
elif loc == 'park':
    print('I love parks')
elif loc == 'institute':
    print("I learn Datascience")
else:
    print("I do not know much")



