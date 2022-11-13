#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nltk


# In[ ]:


text = "The 2014 FIFA World Cup was the 20th FIFA World Cup, the quadrennial world championship for men's national football teams organised by FIFA."


# In[ ]:


text = text.lower()


# In[ ]:


from nltk.tokenize import word_tokenize


# In[ ]:


result = word_tokenize(text)


# In[ ]:


print(result)


# In[ ]:


count = 0


# In[ ]:


for i in range(0, len(result)):
    for j in range(0, len(result)):
        if(result[i]==result[j]):
            count = count + 1
    
    print(result[i],count)
    count = 0


# In[ ]:


for i in range(0, len(result)):
    for j in range(0, len(result)):
        if(result[i]==result[j]):
            count = count + 1
    
    if(count>=2):
        print(result[i], count)
        count=0
    else:
        count=0


# In[ ]:


dict_ = {}


# In[ ]:


for i in range(0, len(result)):
    for j in range(0, len(result)):
        if(result[i]==result[j]):
            count = count + 1
    
    if(count>=2):
        dict_[result[i]]=count
        count=0
    else:
        count=0


# In[ ]:


dict_


# In[ ]:


final = []


# In[ ]:


flag = 0


# In[ ]:


for i in range(0,len(result)):
    flag=0
    for j in dict_:
        if(result[i]==j):
            final.append(1)
            flag = 1
            break
        else:
            continue
    if(flag==0):
        final.append(0)


# In[ ]:


print(final)

