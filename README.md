- 👋 Hi, I’m @faceWelcome

- I am a graduate student majoring in artificial intelligence, and I am currently in the first year of my graduate studies. I welcome you to communicate with me through private messages or comments. My current research focuses on natural language 
processing and time series prediction

For text prediction, there are three main steps:
1. First of all, you need to have a vocabulary base, either from this book or from an existing vocabulary base on the Internet. The vocabulary is represented by a dictionary, for example: {' 我 ': 1,' 们' : 2, '你' : 3, '爱' : 4}. Then, the input text is divided into words and mapped to the unique expression of the vocabulary, for example, the sentence "我们爱你" can be mapped to (1,2,4,3).We then generate a text input and label by moving the text input backward one space for the output label, for example, for the sentence "We love you", setting a window to 3 will generate the input and label group, respectively: inputs:(1,2,4) targets :(2,4,5). This is the end of the preprocessing task.
2. In forecasting, we input inputs into our model and get an output. The specific output will be determined by your model, for example, the RNN model, we decide that the RNN neural unit is 3, then the number of data processed each time is also 3, that is, we will input 3 words into it, and then accept a word at each step, you can choose each neural unit as to get an output or only get one output, based on your model. You will definitely end up with a matrix the size of your vocabulary. The core of predictive text is to select the word mapping corresponding to probability. We will do Softmax on this output, get a probability matrix, and then choose the word with the highest probability as the output, for example: (0.3, 0.2, 0.4, 0.1), from which we can see that the position with the highest probability is the output word, that is, the third position, corresponding to the word "你" in the vocabulary we just illustrated. So understanding the above section, you understand the core of textual prediction.
3. Finally, we will get a complete prediction proof, if we get a prediction of (3,2,4), that is, "爱你我", it can be seen that the effect is not very good, we will calculate the prediction and label Loss, get a value, then perform reverse gradient update, and constantly train the model to reduce Loss.

- 👋 Thank you so much for seeing here and downloading this simple model based on my current understanding. Your guidance is very welcome, and you are welcome to comment and communicate with me below. I will update more basic model-based forecasting projects in the future, thank you.

catalogue-| data -----------> Glossary and training text
          | parameter ------> Model parameter
          | RNN ------------> Models, related tools, training scripts

You can run UsingTest.py directly to test a pre-trained lightweight model. Due to hardware limitations, parameter tuning is relatively low. You can retrain a model yourself by adjusting parameters and better vocabularies and text.
