#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[4]:


import tensorflow as tf


# In[5]:


print(tf.__version__)


# In[6]:


from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
get_ipython().run_line_magic("matplotlib","inline")


# In[7]:


mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[8]:


len(x_train)


# In[9]:


len(x_test)


# In[10]:


x_train.shape


# In[11]:


x


# In[12]:


x_test.shape


# In[13]:


x_train[0]#features of data, with intensities from 0 to 255


# In[14]:


plt.matshow(x_train[11])# visualize the data


# In[16]:


plt.matshow(x_train[21])


# In[17]:


x_train=x_train/255 #normalize the images by scaling the pixel intensities to range 0,1 ...helps to speed up the training


# In[18]:


x_test=x_test/255


# In[19]:


x_train[11] #normalized data


# In[20]:


plt.matshow(x_train[11])


# In[21]:


#creating model


# In[23]:


model=keras.Sequential(
[
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])


# In[24]:


model.summary()


# In[26]:


model.compile(optimizer='sgd',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[28]:


history=model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)


# In[30]:


test_loss,test_acc=model.evaluate(x_test,y_test)
print("Loss=%.3f"%test_loss)
print("Accuracy=%.3f"%test_acc)


# In[31]:


n=random.randint(0,9999)
plt.imshow(x_test[n])
plt.show()


# In[34]:


predicted_value=model.predict(x_test)
print("Value is= %d" %np.argmax(predicted_value[n]))


# In[37]:


get_ipython().run_line_magic('pinfo2', 'history.history')


# In[38]:


history.history.keys()


# In[40]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model_accuracy')
plt.xlabel('accuracy')
plt.ylabel('value accuracy')
plt.legend(['Train','Validation'],loc='upper left')
plt.show()


# In[45]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('loss')
plt.xlabel('accuracy')
plt.ylabel('loss')
plt.legend(['Train','Validation'],loc='center right')
plt.show()


# In[49]:


keras_model_path='/random'
model.save(keras_model_path)


# In[ ]:




