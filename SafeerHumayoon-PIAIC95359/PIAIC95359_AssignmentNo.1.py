#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


x = np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


v = np.arange(10,49)


# 4. Find the shape of previous array in question 3

# In[4]:


np.shape(v)


# 5. Print the type of the previous array in question 3

# In[9]:


print(type(v))


# 6. Print the numpy version and the configuration
# 

# In[6]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[7]:


print(v.ndim)


# 8. Create a boolean array with all the True values

# In[8]:


a = np.ones(10, dtype=bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[10]:


x = np.array([[2, 4, 6], [6, 8, 10]], np.int32)


# 10. Create a three dimensional array
# 
# 

# In[11]:


newarr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[12]:


new = np.arange(12, 38)
new = new[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[13]:


vec = np.zeros(10)
vec[4] = 1


# 13. Create a 3x3 identity matrix

# In[14]:


mat = np.eye(3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[15]:


arr = np.array([1, 2, 3, 4, 5])
print("Original type: " + str(type(arr[0]))) 
arr = arr.astype(np.float)
print("Final type: " + str(type(arr[0])))


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[16]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr_result = np.multiply(arr1, arr2)
print(arr_result)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[29]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
print("Array made by comparing both the arrays provided: ")
print(np.in1d(arr1, arr2))


# 17. Extract all odd numbers from arr with values(0-9)

# In[21]:


arr = np.arange(10) 
print("Original array: ", arr)
result = []    
for i in arr:
    if i % 2 != 0:
        result.append(i)
print("Extracted odd values: ", np.array(result))


# 18. Replace all odd numbers to -1 from previous array

# In[22]:


for i in arr:
    if i % 2 != 0:
        arr[i]=-1
print("Replaced odd number values: ", arr)    


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[25]:


arr = np.arange(10)
print("Original array: ", arr)
arr[5]=12
arr[6]=12
arr[7]=12
arr[8]=12
print("Replaced values: ", arr)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[26]:


x = np.ones((5,5))
print("Original array: ", x)
print("Array modified with 1 on the border and 0 inside in the array: ")
x[1:-1,1:-1] = 0
print(x)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[27]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1][1]=12
print("Replaced value: ")
print(arr2d)


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[37]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0] = 64
arr3d[1] = 64
print("Converted Array: ")
print(arr3d)


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[43]:


array2d = np.arange(0,9)
print("2D array: ") 
print(array2d)
array2d.reshape(3,3)
array2d[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[42]:


arraynew = np.arange(0,9).reshape(3,3)
print("2D array: ") 
print(arraynew)
arraynew[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[41]:


arraynewest = np.arange(0,9).reshape(3,3)
print("2D array: ") 
print(arraynewest)
print("slice out the third column but only the first two rows:", arraynewest[2,0:2])


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[44]:


z = np.random.random((10,10))
print("10x10 Array:")
print(z) 
minvalue, maxvalue = x.min(), x.max()
print("Minimum and Maximum Values:")
print(minvalue, maxvalue)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[46]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("Common items between a and b: ", np.intersect1d(a, b)) 


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[48]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
print("Positions where elements of a and b match: ", np.where(np.in1d(a, b))[0])


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[53]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print("Original Array of data: ")
print(data)
print("Values from array data where the values from array names are not equal to Will: ")
print(data[names != "Will"])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[52]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print("Original Array of data: ")
print(data)
print("Values from array data where the values from array names are not equal to Will: ")
print(data[names != "Will"])
print("Values from array data where the values from array names are not equal to Joe: ")
print(data[names != "Joe"])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[55]:


arrdec = np.random.randn(1,15).reshape(5,3)
print("Array:")
print(arrdec)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[56]:


arrnew = np.random.randn(1,16).reshape(2,2,4)
print("Array:")
print(arrnew)


# 33. Swap axes of the array you created in Question 32

# In[58]:


print("Swapped axes: ")
arrnew.T


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[60]:


arri = np.arange(10)
print("Original Array: ")
print(arri)
arri = np.sqrt(arri)
print("Values replaced: ")
np.where(arr<0.5,0,arr)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[61]:


firstarr = np.random.randint(12)
print("First array: ")
print(firstarr)
secondarr = np.random.randint(12)
print("Second array: ")
print(secondarr)
x = np.maximum(firstarr,secondarr)
print("Array with the maximum values between each element of the two arrays: ")
print(x)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[62]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
names = set(names)
print("These are the unique names in the array: ")
names


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[64]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
print("Array a with removed items that were present in array b: ")
a[b[np.searchsorted(b,a)] !=  a]


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[74]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
print("Original Array: ")
print(sampleArray)
sampleArray = np.delete(sampleArray, 1, 1)
print("Column two deleted: ")
print(sampleArray)
newColumn = np.array([[10,10,10]])
sampleArray = np.insert(sampleArray, 1, newColumn, axis=1)
print("New column inserted: ")
print(sampleArray)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[75]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print("Dot product: ")
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[77]:


A = np.random.randint(10, size=(20))
print("Matrix of 20 random values: ")
print(A)
print("Cumulative sum: ")
print(np.sum(A))


# In[ ]:




