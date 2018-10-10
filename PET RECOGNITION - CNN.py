
# coding: utf-8

# # Pet Recognition system with a Convolutional Neural Network

# # Building the CNN

# In[1]:


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


# Initialising the CNN
classifier = Sequential()


# In[3]:


# Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))


# In[4]:


# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[5]:


# Second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# In[6]:


# Flattening
classifier.add(Flatten())


# In[7]:


# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))


# In[8]:


# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# # Applying the CNN to the images

# In[9]:


from keras.preprocessing.image import ImageDataGenerator


# In[10]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


# In[11]:


test_datagen = ImageDataGenerator(rescale = 1./255)


# In[12]:


training_set = train_datagen.flow_from_directory('user/train_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[13]:


test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[ ]:


classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)

