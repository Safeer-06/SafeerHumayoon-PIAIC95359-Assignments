#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Original Array: ")
print(arr)
arr_2d = np.reshape(arr, (2, 5))
print("Converted Array: ")
print(arr_2d)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[4]:


a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
print("Vertically stacked Arrays: ")
np.vstack((a,b))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[5]:


x = np.array((3,5,7))
y = np.array((5,7,9))
print("Horizontally stacked Arrays: ")
np.hstack((x,y))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[11]:


origarr = np.arange(15).reshape(1,3,5)
print("Original array: ")
print(origarr)
newarr = origarr.flatten()
print("Flat 1d array: ")
newarr


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[14]:


a = np.array([[1,2,3], [4,5,6]])
c = a.flatten()
c


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[13]:


arrdim = np.arange(15).reshape(-1,3)
arrdim


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[16]:


arrayin = np.arange(25).reshape(5,5)
print("Original Array: ")
print(arrayin)
print("Square of the array: ")
np.square(arrayin)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[18]:


np.random.seed(123)
arraynrr = np.random.randint(30, size = (5,6))
print("Original Array: ")
print(arraynrr)
print("Mean of the array: ")
arraynrr.mean()


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[19]:


np.std(arraynrr)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[20]:


np.median(arraynrr)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[21]:


np.transpose(arraynrr)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[25]:


arraynewi = np.arange(16).reshape(4,4)
print(arraynewi)
x = np.diagonal(arraynewi)
print("Sum of diagonal elements: ")
print(x.sum())


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[26]:


np.linalg.det(arraynewi)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[27]:


arrayque = np.arange(10)
print(arrayque)
print(np.percentile(arrayque,5))
print(np.percentile(arrayque,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[30]:


arrayque = np.arange(10)
print(arrayque)
np.isnan(arrayque)


# In[ ]:




