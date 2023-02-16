
#Exception Handling
# try: and except: are key words for exceptional handling, try: provide the code ; except: except the error with a proper statement   

#Ex - 1:
try:
    x = eval(input("enter the 1st value = "))
    y = eval(input("enter the 2nd value = "))
    results = x+y
    print(results)
except:
    print("please enter a valid number")

    
#Ex - 2: 
try:
    x = eval(input("enter the 1st value = "))
    y = eval(input("enter the 2nd value = "))
    results = x/y
    print("final results = ", results)
except(ZeroDivisionError):
    print("please enter a non-zero value for the divisor")
except(NameError):
    print("Please enter valid number")
except(TypeError):
    print("Please enter both same type")
   


# Multi Threding
#----multithreading using sum of square---#
import time

def calc_square(numbers):
    print("calculate square numbers")
    for n in numbers:
        time.sleep(0.2)
        print('square:', n*n)

def calc_cube(numbers):
    print("calculate cube of numbers")
    for n in numbers:
        time.sleep(0.2)
        print('cube:', n*n*n)
        
arr = [2,5,7,9]

t = time.time()
     
calc_square(arr)
calc_cube(arr)
print("done in: ",time.time()-t)
print("completed my way to work!")


#---inorder to process exit with code 0---#

import time
import threading

def calc_square(numbers):
    print("calculate square numbers")
    for n in numbers:
        time.sleep(0.2)
        print('square:',n*n)

def calc_cube(numbers):
    print("calculate cube of numbers")
    for n in numbers:
        time.sleep(0.2)
        print('cube:',n*n*n)
        
arr = [2,5,7,9] 

t1 = threading.Thread(target=calc_square, args=(arr,))
t2 = threading.Thread(target=calc_cube, args=(arr,))

t = time.time()
t1.start()
t2.start()
t1.join()
t2.join()

print("done in: ",time.time()-t)
print("completed my way to work!")
  