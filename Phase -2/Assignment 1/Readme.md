The Loss and Accuracy curves is obtained after training the model for **50** epochs on IMDB review dataset 
with pretrained GLOVE word embeddings. 


![](https://github.com/LokeshJatangi/EVA-vision/blob/master/Phase%20-2/Assignment%201/Loss.png) 


![](https://github.com/LokeshJatangi/EVA-vision/blob/master/Phase%20-2/Assignment%201/accuracy.png) 


Q.What are the words considered inside the embedding layer from the 'max_words=10000' ?

A. When the tokenizer object is created and texts to sequences method is applied on the training data(text) ,
   only the top 10000 most common words are considered for the embedding layer and only these words Glove
   word embeddings are updated.
