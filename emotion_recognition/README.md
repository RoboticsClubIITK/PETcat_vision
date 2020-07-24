# Emotion Recognition   
The emotion classifier is wrapped inside a class and is present in ```emotion_class.py```.   
The model has been implemented using ***Keras***    

### The class : emotionRecognizer 
### class variables : 
  img_rows, img_cols    
  face_cascade    
  modelFile
  configFile   
  emotion   
  model   
  
### The methods : 
```loadModel()``` : Loads the saved model   
```predict(x)``` : Predicts the emotion using the input image *x*.      
```getFaces(img)``` : Uses ```openCV Haar Cascade to detect faces in the image *img*.   

### The working :    
From the video feed, we first run ```getFaces()``` method to detect faces in the image.   
If a face is found, it is then fed into ```predict()``` to classify the emotion.
