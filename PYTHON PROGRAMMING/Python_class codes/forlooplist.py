# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 12:50:15 2020

@author: abdul
"""
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

###############################################
############ For loops###########

# many objects in Python are iterable meaning we
# we can iterate over every element in the object
# such as every element in a list or every character
# in a string or tuple or dictionary

# syntax for loop: -

my_iterable = [1,2,3,4]

for item  in my_iterable:
    print(item)

mylist = [1,2,3,4,5,6,7,8,9,10]

for num in mylist:
    print('DS')



from itertools import islice
import itertools
# islice(iterable, start, stop, step )# syntax
# Slicing the range function
for i in islice(range(20), 5):
    print(i)
li = [2, 4, 5, 7, 8, 10, 20]

# Slicing the list
print(list(itertools.islice(li, 1, 6, 2)))


mylist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
newli = list(itertools.islice(mylist,0,31,2))

for num in newli:

    # check for even numbers
    if num%2==0:
        print(num)

    else:
        print(f'odd number: {num}')


mylist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
odd = []
even = []
for i in range(len(mylist)):
    if i % 2:
        even.append(mylist[i])
    else :
        odd.append(mylist[i])

res = odd+ even
odd
even



list_sum = 0
for num in mylist:
    list_sum = list_sum+num
    print(list_sum)

print(list_sum)

#### strings

mystring = "hello world"

for letter in mystring:
    print(letter)

tup = (1,2,3)

for item in tup:
    print(item)

# tuple unpacking

mylist = [(1,2),(3,4),(5,6),(7,8),(9,10)]
len(mylist)

for item in mylist:
    print(item)

# tuple unpacking
for a,b in mylist:
    print(a)
    print(b)
my = [(1,2,3,4),(5,6,7,8)]

for a,b,c,d in my:
    print(a)
    print(b)
    print(c)
    print(d)

# dictionary

d = {'k1':1,'k2':2,'k3':3}

for item in d.items():
    print(item)

for key, value in d.items():
    print(key)
    print(value)

###################################################

# while loops

# while loops will continue to execute a block
# of code while some condition is True

# while my dogs are still hungry , I will keep
# feeding my dogs

# syntax: -

while some boolean condition:
    # do something

else:
    # do someother thing

x = 0
while x<5:
    print(f'the current value of x is {x}')
    x = x+1
    # x+= 1

else:
    print("x is not less than 5 ")

# break, continue , pass

# pass : - does nothing all
# break : - break out of the current closest enclosing loop
# continue:- goes to the top of the closest enclosing loop

x  = [1,2,3]

for item in x:
    pass

print("end of script")

mystring = "sammy"

for letter in mystring:
    if letter == 'a':
        continue
    print(letter)

for letter in mystring:
    if letter == 'a':
        break
    print(letter)

#### useful operator in Python

# range
# range(start, stop,step)

for num in range(0,10,2):
    print(num)


list(range(0,11,2))

# enumerator returns index count and object

word = "abcde"

index_count = 0

for letter in word:
    print('at index {} the letter is {}'.format(index_count, letter))
    index_count+= 1

for item in enumerate(word):
    print(item)

for index, value in enumerate(word):
    print(index)
    print(value)

# zip

mylist1 = [1,2,3,8,9,10]
mylist2 = [4,5,6]

for item in zip(mylist1,mylist2):
    print(item)

list(zip(mylist1,mylist2))

# in
'a' in ['x','y','z']

# min and max function

mylist = [10,20,30,50,100,200]
min(mylist)
max(mylist)

# shuffle which shuffles list

from random import shuffle

mylist = [1,2,3,4,5,6,7,8,9,10]
shuffle(mylist)
mylist

from random import randint

randint(0,100)

# input

result = float(input("favourite number: "))
result
type(result)

#list comprehension are a unique way of quickly
# creating a list in Python

mystring = "hello world"

mylist = []

for letter in mystring:
    mylist.append(letter)

mylist

mylist1 = [x for x in mystring]

[x for x in range(0,11)]

[x**2 for x in range(0,11)]

[x for x in range(0,11) if x%2== 0]

celcius = [0,10,20,30,34.5,50]

fahrenheit = [(9/5)*temp+32 for temp in celcius]
fahrenheit
fah = []
for temp in celcius:

    fah.append((9/5)*temp+32)
fah
