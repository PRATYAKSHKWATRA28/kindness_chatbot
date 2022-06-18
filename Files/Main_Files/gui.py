import tkinter
from tkinter import *
from unittest import result
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np
import os
#import pyttsx3 as pp

#engine=pp.init()
#voices=engine.getProperty('voices')

#engine.setProperty('voice',voices[1].id)
#rate = engine.getProperty('rate')
#engine.setProperty('rate', 250)

#def speak(word):
#	engine.say(word)
#	engine.runAndWait()

intents = json.loads(open('.\Main_Files\intents.json').read())
model = load_model('.\Other_Files\chatbot_model.h5')
words = pickle.load(open('.\Other_Files\words.pkl', 'rb'))
classes = pickle.load(open('.\Other_Files\classes.pkl', 'rb'))

def bow(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	bag = [0]*len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i]=1
	return (np.array(bag))

def predict_class(sentence):
	sentence_bag = bow(sentence)
	res = model.predict(np.array([sentence_bag]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
	#sort by probablity
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent':classes[r[0]], 'probablity':str(r[1])})
	return return_list

def getResponse(ints):
	tag = ints[0]['intent']
	list_of_intents = intents['intents']
	for i in list_of_intents:
		if(i['tag']==tag):
			result=random.choice(i['responses'])
			break

	return result

def chatbot_response(msg):
	ints = predict_class(msg)
	res = getResponse(ints)
	return res

def send():
	msg = TextEntryBox.get("1.0", 'end-1c').strip()
	TextEntryBox.delete('1.0', 'end')

	if msg != '':
		ChatHistory.config(state=NORMAL)
		ChatHistory.insert('end', "You: " + msg + "\n\n")
		res = chatbot_response(msg)
		ChatHistory.insert('end', "Bot: " + res + "\n\n")
		ChatHistory.config(state=DISABLED)
		ChatHistory.yview('end')
		#speak(res)

def info_open():
	os.system("notepad.exe .\Extra\Documents\info.txt")


base = Tk()
base.title("Kindness Chatbot")
base.iconbitmap(".\Extra\Images\smiling-face.ico")
base.geometry("400x500")
base.resizable(width=False, height=False)

#chat history textview
ChatHistory = Text(base, font=('Inter', 12, 'bold'), bd=0, bg="#e6b9b8") #Add smiley image in background
ChatHistory.config(state=DISABLED)

TextEntryBox = Text(base, font=('Inter', 12, 'bold'), bd=0, bg='#0e1d31', fg="#ffffff", insertbackground="#ffffff", insertwidth=4)

button_exit = Button(base, font=('Inter', 8, 'bold'), 
	text="Exit", bg="#000000", activebackground="#ffff99", fg="#ffffff", command=base.quit)

button_info = Button(base, font=('Inter', 8, 'bold'), 
	text="Info", bg="#000000", activebackground="#ffff99", fg="#ffffff", command=info_open)

SendButton = Button(base, font=('Nunito', 18, 'bold'), 
	text="Send", bg="#d61e4c", activebackground="#3e3e3e", fg="#ffffff", borderwidth=10, command=send)

def send_press():
	send()
	SendButton.configure(bg="#3e3e3e", font=('Nunito', 15, 'bold'), borderwidth=10)

def send_release():
	SendButton.configure(bg="#d61e4c", font=('Nunito', 18, 'bold'), borderwidth=10)

base.bind('<Return>',lambda event:send_press())
base.bind('<KeyRelease-Return>',lambda event:send_release())
base.bind('<Escape>',lambda event:base.quit())
base.bind('<Shift-I>',lambda event:info_open())

ChatHistory.place(x=6, y=28, height=367, width=386)
TextEntryBox.place(x=6, y=400, height=90, width=265)
SendButton.place(x=273, y=400, height=90, width=120)
button_exit.place(x=200, y=3, width=192)
button_info.place(x=6, y=3, width=192)

print('Kindness Chatbot by Pratyaksh Kwatra')
print('Kindness Chatbot by Pratyaksh Kwatra')

base.mainloop()
