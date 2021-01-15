# Shopee-Code-League-2020
This consists of 2 projects that me and my team attempted. All images and datasets belong to Shopee Code League 2020 with full credits.
## 1. Product Detection

For product detection, we were given a problems consisiting of 100k images. The goal was to create a robust detection system that will improve the listing and categorisation efficiency. My team developed used computer vision to engineer an image classification model via "tensorflow" to determine the correct product categorisations.

### Example of images
 ![CV images](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/Image_classification(1).PNG)
 
 ### File parsing
 We first accessed the 105,405 files and 44 folders using Python's OpenCV library so that the images can be reshaped into Numpy arrays so that they can be read by other python libaries in our machine learning model. Our team also set up 41 categories(as per stated in the competition) for the OpenCV library to act on too.
 
 ![parsing](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/OpenCV.PNG)
 
 ### Feature engineering
 Afterwards, we added the features into our X_test to be used to train our machine learning model
 ![ft](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/Feature%20Engineering.PNG)
 
 ### Model development and training
 We used python's keras library with "Sequential" model that is flattened to 1-dimension to fit the input that we are feeding 28x28 pixels consisting of 128 hidden units using "adam" optimiser. 
 
 ![Model development and training](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/Feature%20Engineering.PNG)
 
 We then used the same file parsing technique to read the 12,192 files into a dataframe to be predicted,followed by feature engineering,to be predicted after training. 
 ![ft](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/OpenCV(1).PNG)
 
 I then used a tensorflow-keras model with softmax layer, which normalises the output into a probability distribution, as the machine learning model with the training set as inputs.
 ![training](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/Training.PNG)

 ### Predictions
 After training, I used the model to predict the categories of the images inserted and exported it as a csv file.
 
 ![testing](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/Testing.PNG)
 
 ### Results
 We had an overall of 40.327% accuracy of product categories between test and the answer datasets. More improvements should be made to our model to improve accuracy

## 2. Sentimental Analysis

For sentimental analysis, we were given product review datasets. The goal was to create a classification model for sentimental analysis so as to predict the rating based on the reviews given using Natural Language Processing(NLP). Our team used nltk-for NLP-, tensorflow and keras for machine learning and sklearn for classification, regression and clustering algorithms.

### Data Pre-Processing
To clean the data, we removed punctuations,special characters and stopwords from all 28 languages in the nltk library

 ![Pre-processing1](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Sentimental%20Analysis/pre_processing(1).PNG)
 ![Pre-processing2](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Sentimental%20Analysis/pre_processing(2).PNG)
 
  ### Feature engineering
  We first set up 5 categories as the features to determine the ratings for the neural network to detect.
  
 ![ft](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Sentimental%20Analysis/Ft.PNG)
 
 ### Tokenisations
 Then we used the tokeniser function to sperate the text in the product review for better reading into the machine learning model.
 ![Tokenisation](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Sentimental%20Analysis/tokeniser.PNG)
 
 ### Model development and training
 For the machine learning model, we used seqeuntial model from keras and tensorflow library with 
 * 10 epoch
 * 4 layers of size 64,64,512,6
 * First layer consisting of embedding layer of size 64
 * Second layer consistiing the Bidirectional LSTSM Classifier  of size 64
 * Third layer conisting l2 kernel regularsers,l2 bias regularisers and activation 'relu' with size 512
 * Fourth layer consisting softmax activation function with size 6
 * Dropout of 0.25
 * Batch Normalisation
 ![Model](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Sentimental%20Analysis/Model.PNG)
 
 ### Predictions
 After training, we used the model to predict the ratings based on the product review of the test set
 ![Model](https://github.com/JiaJun98/Shopee-Code-League-2020/blob/main/Product%20Detection/Predictions.PNG)

