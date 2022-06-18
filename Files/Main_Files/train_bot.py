import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import random
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#keras.optimizers.SGD
from keras.optimizers import *

lemmatizer = WordNetLemmatizer()

words=[]
classes=[]
documents=[]
ignore_words=['?','!','.']
data_file = open('.\Main_Files\intents.json').read()
intents = json.loads(data_file)

for intent in intents['intents']:
	for pattern in intent['patterns']:
		#tokenize here
		w=nltk.word_tokenize(pattern)
		#print('Token is: {}'.format(w))
		words.extend(w)
		#(['hey', 'you'], 'greeting')
		documents.append((w, intent['tag']))
		# add the tag to classes list
		if intent['tag'] not in classes:
			classes.append(intent['tag'])
	
	# Final lists
	# print('Words list is: {}'.format(words))
	# print('Docs are: {}'.format(documents))
	# print('Classes are: {}'.format(classes))
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
classes = list(set(classes))
#print(words)
pickle.dump(words, open('.\Other_Files\words.pkl', 'wb'))
pickle.dump(classes, open('.\Other_Files\classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)
# [0,0,0,0,0,0,0,0]
for doc in documents:
	bag = []
	pattern_words = doc[0]
	pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
	#print('Current Pattern Words: {}'.format(pattern_words))

	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)

	#print('Current Bag: {}'.format(bag))

	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1
	#print('Current Output: {}'.format(output_row))

	training.append([bag, output_row])

#print('Training: {}'.format(training))
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
#print('X: {}'.format(train_x))
#print('Y: {}'.format(train_y))

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compiling the model & define an optimizer function
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

mfit = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('.\Other_Files\chatbot_model.h5', mfit)

print("--------------------------------------------")
print('Completed Training')
print("--------------------------------------------")
print('Kindness Chatbot by Pratyaksh Kwatra')
print('You can find the new files created in the same folder named as chatbot_model.h5 , classes.pkl and words.pkl')
print("--------------------------------------------")
print('For using GUI Application for the Chatbot , you need to run the gui.py file which you can find in the same folder')
print("--------------------------------------------")
print('The code was executed without any errors')
print('You may close this Terminal and continue your work')
print("--------------------------------------------")
print('Kindness Chatbot by Pratyaksh Kwatra')