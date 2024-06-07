Welcome to @faceWelcome's GitHub
👋 Hi, I’m @faceWelcome!

I am a graduate student majoring in artificial intelligence, currently in my first year. I welcome you to communicate with me through private messages or comments. My current research focuses on natural language processing and time series prediction.

Text Prediction Overview
For text prediction, there are three main steps:

Vocabulary Creation:

First, you need to create a vocabulary base, either from a book or an existing vocabulary base on the Internet. The vocabulary is represented by a dictionary. For example: {'我': 1, '们': 2, '你': 3, '爱': 4}. The input text is then divided into words and mapped to the unique expressions in the vocabulary. For instance, the sentence "我们爱你" can be mapped to (1, 2, 4, 3).
Next, generate text input and labels by moving the text input backward one space for the output label. For example, with a window size of 3, the sentence "We love you" will generate input and label groups: inputs: (1, 2, 4) and targets: (2, 4, 3). This completes the preprocessing task.
Model Prediction:

The inputs are fed into our model to get an output. The specific output will depend on your model. For example, in an RNN model with an RNN unit size of 3, each input consists of 3 words. Each step accepts a word, and you can choose to get an output from each neural unit or only one output based on your model.
You will ultimately get a matrix the size of your vocabulary. The core of text prediction is to select the word mapping corresponding to the highest probability. We apply Softmax to this output to get a probability matrix and then choose the word with the highest probability. For example: (0.3, 0.2, 0.4, 0.1). Here, the word at the third position has the highest probability, corresponding to "你" in our example vocabulary. Understanding this is key to grasping text prediction.
Model Evaluation and Training:

Finally, we get a complete prediction output. For example, a prediction of (3, 2, 4) corresponds to "爱你我". If the result is not very good, we calculate the loss between the prediction and the label, get a value, then perform backpropagation to update the model and reduce the loss. This process is repeated to improve the model's accuracy.
How to Use This Repository
👋 Thank you for visiting and downloading this simple model based on my current understanding. Your feedback and guidance are highly appreciated, and you are welcome to comment and communicate with me below. I will continue to update this repository with more basic model-based forecasting projects.

Repository Structure
data/ -> Contains the glossary and training text.
parameter/ -> Holds the model parameters.
RNN/ -> Includes models and training scripts.
ults/ -> Stores results.
You can run UsingTest.py directly to test a pre-trained lightweight model. Due to hardware limitations, the parameter tuning is relatively basic. You can retrain a model yourself by adjusting parameters and using better vocabularies and text.

![image](https://github.com/faceWelcome/Text-prediction-model-based-on-RNN/tree/main/IMG)
