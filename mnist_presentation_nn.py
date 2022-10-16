import tensorflow.compat.v2 as tf
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import io
import os
import subprocess
import PIL.ImageOps
from PIL import Image
import pickle
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


saved_model=tf.keras.models.load_model('digitnn_model')
classifier = pickle.load(open("model.pkl", "rb"))
digits = datasets.load_digits()


def normalize_img(image):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255.

def cl():
   canvas.postscript(file="file_name.ps", colormode='color')
   psimage=Image.open('file_name.ps')
   psimage.save('file_name.png')
   image = Image.open('file_name.png')
   new_image = image.resize((28,28))
   new_image.save('new_image.png')
   img = Image.open('new_image.png').convert('L')

   img2 = PIL.ImageOps.invert(img)

   img2.save('greyscale.png')

   #img_array=np.array(Image.open('greyscale.png'))
   img_array=np.array(img2)
   img_to_be_predicted=tf.cast(img_array, tf.float32) / 255.
   img_to_be_predicted=tf.reshape(img_to_be_predicted,[1,28,28])

   #saved_model=tf.keras.models.load_model('digitnn_model')
   predictions=saved_model.predict(img_to_be_predicted,batch_size=1) #print probabilities of classes

   classes = np.argmax(predictions, axis = 1)
  
   print("Result: ",classes,"\n")

   canvas.delete("all")
   

def svm_run():
   canvas.postscript(file="file_name.ps", colormode='color')
   psimage=Image.open('file_name.ps')
   psimage.save('file_name.png')
   image = Image.open('file_name.png')
   new_image = image.resize((8,8))
   new_image.save('new_image.png')
   #img = Image.open('new_image.png').convert('L')
   
   img = Image.open('new_image.png').convert('L')
   
   img = PIL.ImageOps.invert(img)
    
   img.save('greyscale.png')
   img_array = np.array(img)
   #print(img_array)
   data = img_array.reshape((1, -1)) 

   for i in range(64):
       data[0][i] = (data[0][i]*16)/255
   data2 = [[None] * 64]
   for i in range(64):
       data2[0][i] = float( data[0][i] )
   data3 = np.array(data2)
   
   #imgplot = plt.imshow(var)
   #plt.show()
   #print(data3) 
   predicted = classifier.predict(data3)
   print(predicted)
  
   #predicted = classifier.predict([ digits.data[1] ])
   #print(digits.data)
   #print(digits.data[1])
   #print(predicted)
  
   canvas.delete("all")

def savePosn(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    #canvas.create_oval( lastx, lasty, event.x, event.y , width=35, fill='black')
    #canvas.create_line((lastx, lasty, event.x, event.y), width=25, fill='black')
    #svm painting
    canvas.create_oval( lastx, lasty, event.x, event.y , width=35, fill='black')
    savePosn(event)

root = Tk()
root.geometry("500x500")
#saved_model=tf.keras.models.load_model('digitnn_model')
button1 = Button(root, text = "Neural Network",height=5, width=13,command = cl )
button1.pack()

#button-2
button2 = Button(root, text = "SVM",height=5, width=13,command = svm_run)
button2.pack()

canvas = Canvas(root, bg = "white", width = 500 ,height = 500)
canvas.bind("<Button-1>", savePosn)
canvas.bind("<B1-Motion>", addLine)
canvas.pack()

root.mainloop()



