# Sign Language Translator

This is my first attempt at implementing deep learning and using Neural Networks. Here, I created a Convolutional Neural Network(CNN) trained using a custom dataset that I made. It recognizes hand genstures by using openCV's library. The neural network was built using Tensorflow and Keras.

## Libraries

* Tensorflow
* Keras
* OpenCV

## Procedure

If you want to create your own custom dataset, you can run the picTakingScript.py and use the create_dataset(number_imgs) where number_imgs == number of images that you want to have for each letter. You must also have a folder called "dataset" within that same folder with all of the letters in it.

```
/dataset/
      /a/
          a1.jpg
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
