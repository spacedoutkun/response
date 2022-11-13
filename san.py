#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[4]:


import pandas as pd
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


fmnist=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fmnist.load_data()


# In[12]:


len(y_train)


# In[13]:


len(y_test)


# In[15]:


x_train.shape


# In[11]:


x_train[0]


# In[16]:


x_train=x_train/255


# In[17]:


x_test=x_test/255


# In[18]:


x_train[0]


# In[20]:


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])


# In[21]:


model.summary()


# In[23]:


model.compile(optimizer='sgd',
    loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[24]:


history=model.fit(x_train,y_train,validation_data=(x_train,y_train),epochs=10)


# In[25]:


(test_loss,test_acc)=model.evaluate(x_test,y_test)
print("acc = %.3f"%test_acc)
print("loss =%.3f"%test_loss)


# In[29]:


plt.matshow(x_train[0])


# In[30]:


x_test.shape


# In[27]:


predicted_value=model.predict(x_test)


# In[31]:


predicted_value.shape


# In[32]:


predicted_value[0]


# In[33]:


np.argmax(predicted_value[0])


# In[34]:


class_labels=["tshirt","trouser","pullover","dress","coat","sandal","shirt","sneakers","bag","shoes"]


# In[35]:


class_labels[np.argmax(predicted_value[0])]


# In[36]:


get_ipython().run_line_magic('pinfo2', 'history.history')


# In[37]:


history.history.keys()


# In[ ]:




