# 📰 Fake News Classification — LSTM Text Classifier

AUTHORS: Loïc Delattre, Guillaume Le Chartier, Mattis Roellinger


The aim of this project is to train a neural network to classify news articles as FAKE or REAL using TensorFlow.

It loads a dataset, preprocesses text, trains an LSTM-based classifier, evaluates accuracy, and allows users to test new articles.



## 📁 Project Structure

```.

├── fakenews.zip                     # Downloaded dataset (fake + real CSVs)

├── DataSet\_Misinfo\_FAKE.csv         # Fake news samples

├── DataSet\_Misinfo\_TRUE.csv         # True news samples

└── fake\_news\_classifier.ipynb       # Main notebook / script

```



## 🔧 How the Code Works

##### 1\. Download \& Load Dataset



The script downloads a ZIP archive containing two CSV files:



* DataSet\_Misinfo\_FAKE.csv
* DataSet\_Misinfo\_TRUE.csv



These are loaded with pandas, merged, labeled (0 = fake, 1 = true), shuffled, and cleaned.

###### 

##### 2\. Preprocessing



* Removes articles shorter than 10 words
* Limits each article to the first 400 words
* Tokenizes text using TensorFlow’s Tokenizer
* Converts text → word index sequences
* Pads all sequences to a fixed length



##### 3\. Train/Test Split



The dataset is split:



* 70% training
* 30% testing



Both text and labels are aligned and shuffled.



##### 4\. Model Architecture



A TensorFlow Sequential model:



* Embedding layer
* Dropout
* Bidirectional LSTM
* Dense layers with ReLU
* Output: sigmoid (binary classification)



A custom callback stops training early when accuracy exceeds 99%.



##### 5\. Training



The model is trained with:



* binary\_crossentropy loss
* Adam optimizer
* Learning rate scheduler
* Validation split (20%)



Training history (accuracy + loss) is plotted.



##### 6\. Evaluation



The model is evaluated on the test set, and accuracy is printed.

##### 

##### 7\. User Input Prediction



The user can paste any news article into the console.

The script processes it the same way as the training data and outputs:



* Probability the article is TRUE
* Classification result (FAKE/REAL)



## ▶️ Running the Project



##### Install requirements:



`pip install tensorflow pandas matplotlib requests`





Run the script or notebook:



`python fake\_news\_classifier.py`





or open the notebook in Google Colab.



After training, enter your article when prompted.



## 📊 Output Example

`MODEL ACCURACY ON TEST DATA: 97.5%

article: "The economy will grow by 500% next week..."

MODEL PREDICTED TRUENESS: FAKE NEWS (Article is detected as 3% true)`

