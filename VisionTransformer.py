import tkinter
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageTk
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from vit_keras import vit, utils, visualize #=====loading VIT (vision transformer) model
from PIL import Image,ImageTk
from yoloDetection import detectObject, displayImage
import cv2

global filename, image_size, classes, vit_model, accuracy
global class_labels, cnn_model, cnn_layer_names, yolo_index

def loadModel():
    text.delete('1.0', END)
    global class_labels, cnn_model, cnn_layer_names, yolo_index
    global  image_size, classes, vit_model
    image_size = 384
    classes = utils.get_imagenet_classes()
    #loading VIT Model
    vit_model  = vit.vit_b16(image_size = image_size, activation='sigmoid', pretrained=True, include_top=True, pretrained_top=True)
    #loading yolo model
    class_labels = open('model/yolov-labels').read().strip().split('\n') #reading labels from yolov3 model
    cnn_model = cv2.dnn.readNetFromDarknet('model/yolov.cfg', 'model/yolov.weights') #reading model
    cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
    cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers
            
    text.insert(END,"Vision Transformer Model Loaded\n\n")
    text.insert(END,"Total Objects can be classified by VIT : "+str(len(classes))+"\n\n")
    text.insert(END,"List of classification objects name : "+str(classes))

def loadImage():
    global canvas, images, root
    global filename
    filename = filedialog.askopenfilename(initialdir="testImages")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    img = Image.open(filename)
    img = img.resize((700, 300))
    picture = ImageTk.PhotoImage(img)
    canvas.configure(image = picture)
    canvas.image = picture
    root.update_idletasks()

def VITclassification():
    global canvas, images, root, accuracy, vit_model
    global filename
    accuracy = []
    text.delete('1.0', END)
    image = utils.read(filename, image_size)
    classification = classes[vit_model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
    acc = vit_model.predict(vit.preprocess_inputs(image)[np.newaxis])[0]
    print(acc)
    accuracy.append(np.sum(acc) / len(acc))
    text.insert(END,"VIT Accuracy : "+str(np.sum(acc) / len(acc))+"\n")
    image = cv2.imread(filename)
    img = cv2.resize(image, (700,300))
    cv2.putText(img, classification, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 2)
    cv2.imwrite("output/output.png", img)
    img = Image.open("output/output.png")
    img = img.resize((700, 300))
    picture = ImageTk.PhotoImage(img)
    canvas.configure(image = picture)
    canvas.image = picture
    root.update_idletasks()

def detectFromImage(imagename): #function to detect object from images
    #random colors to assign unique color to each label
    label_colors = np.random.randint(0,255,size=(len(class_labels),3),dtype='uint8')
    try:
        image = cv2.imread(imagename) #image reading
        image_height, image_width = image.shape[:2] #converting image to two dimensional array
    except:
        raise 'Invalid image path'
    finally:
        image, Boundingboxes, confidence_value, class_ids, ids = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels)#calling detection function
        accuracy.append(np.sum(confidence_value) / len(confidence_value))
        text.insert(END,"YOLO Accuracy : "+str(np.sum(confidence_value) / len(confidence_value))+"\n")
        displayImage(image)#display image with detected objects label


def yoloclassification():
    global filename
    detectFromImage(filename)

def graph():
    global accuracy
    height = accuracy
    bars = ('VIT Accuracy','Yolo Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Algorithm Names")
    plt.ylabel("Accuracy")
    plt.title("VIT & Yolo Accuracy Graph")
    plt.show()

def close():
    root.destroy()

def Main():
    global text, canvas, images, root

    root = tkinter.Tk()
    root.geometry("1300x1200")
    root.title("A Survey on Vision Transformer")
    root.resizable(True,True)
    font = ('times', 14, 'bold')
    title = Label(root, text='A Survey on Vision Transformer')
    title.config(bg='yellow3', fg='white')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)
    
    font1 = ('times', 12, 'bold')

    img = Image.open("output/vit.png")
    img.resize((600, 300))
    picture = ImageTk.PhotoImage(img)
    canvas = Label(root, image=picture)
    canvas.place(x=300,y=200)

    loadButton = Button(root, text="Load Vision Transformer Model", command=loadModel)
    loadButton.place(x=60,y=80)
    loadButton.config(font=font1)

    uploadButton = Button(root, text="Upload Image", command=loadImage)
    uploadButton.place(x=400,y=80)
    uploadButton.config(font=font1)

    vitButton = Button(root, text="Classify Image using VIT", command=VITclassification)
    vitButton.place(x=600,y=80)
    vitButton.config(font=font1)

    yoloButton = Button(root, text="Classify Image using YOLO", command=yoloclassification)
    yoloButton.place(x=60,y=130)
    yoloButton.config(font=font1)

    graphButton = Button(root, text="Comparison Graph", command=graph)
    graphButton.place(x=400,y=130)
    graphButton.config(font=font1)

    exitButton = Button(root, text="Exit", command=close)
    exitButton.place(x=600,y=130)
    exitButton.config(font=font1)

    text=Text(root,height=10,width=140)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=510)    
    
    root.mainloop()
   
 
if __name__== '__main__' :
    Main ()
    
