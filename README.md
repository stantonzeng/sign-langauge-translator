# Sign Language Translator

This is my first attempt at implementing deep learning and using Neural Networks. Here, I created a Convolutional Neural Network(CNN) trained using a custom dataset that I made. It recognizes hand genstures by using openCV's library. The neural network was built using Tensorflow and Keras.

## Libraries

* Tensorflow
* Keras
* OpenCV

Make sure you have python installed.

## Procedure

If you want to create your own custom dataset, you can run the picTakingScript.py and use the create_dataset(number_imgs) where number_imgs == number of images that you want to have for each letter. You must also have a folder called "dataset" with all of the letters in it as folders. The same can be done with validation and test data.

```
/dataset/
      /a/
          a1.jpg  <--- You will create this
          a2.jpg
          .
          .
          .
      /b/
          b1.jpg
          b2.jpg
          .
          .
          .
      /c/
      .
      .
      .
      /z/
```

## Classify

Classifying your images first requires you to train your model. As long as you have all of your images in the dataset, valset, and testset folders, it should work. If it doesn't work, just notify me, as I originally did this using jupyter notebook, so a lot of the code is not IDE friendly. The CNN model will train itself using the training data and validate itself using the validation data. It will then retrain itself using the values that it got wrong in both the training set and validation set, and then train itself again. I found this method the best way to increase accuracy.

## Accuracy

The best accuracy my model could achieve was 92% with a validation accuracy of 85%. 

## Misc (Trials and Tribulations)

While this project was fun to do, there were definite periods of trial and error to figure out how to get the most optmial neural network to work with my dataset. The biggest problem was mostly just overfitting and the size of the project. Since it was a custom dataset built by me, my original dataset was... horrible. Lots of noise and inconsistent data lead to the machine learning only about 20% of the time. I had to remake the data 8 different times in order for it to work, so maybe it might be best to get a noncustom data set. 
