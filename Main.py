import tkinter as tk
from tkinter import Message ,Text
from PIL import Image, ImageTk
import pandas as pd

import tkinter.ttk as ttk
import tkinter.font as font
import tkinter.messagebox as tm
import matplotlib.pyplot as plt
import sqlite3
import csv
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as tm

import datasetcreater as dsc
import preprocess as pre
import RFALG as RF
import DTALG as DT
import singlequery as sq

bgcolor="#00ffff"
bgcolor1="#fff"
fgcolor="black"

def Home():
	global window
		


	window = tk.Tk()
	window.title("Music Mood")

 
	window.geometry('1280x720')
	window.configure(background=bgcolor)
	#window.attributes('-fullscreen', True)

	window.grid_rowconfigure(0, weight=1)
	window.grid_columnconfigure(0, weight=1)
	

	message1 = tk.Label(window, text="Lyrics Mood Prediction" ,bg=bgcolor  ,fg=fgcolor  ,width=50  ,height=3,font=('Helvetica', 30, ' bold')) 
	message1.place(x=100, y=20)

	lbl = tk.Label(window, text="Select Dataset",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('Helvetica', 15, ' bold ') ) 
	lbl.place(x=100, y=200)
	
	txt = tk.Entry(window,width=20,bg="white" ,fg="black",font=('Helvetica', 15, ' bold '))
	txt.place(x=400, y=215)


	lbl1 = tk.Label(window, text="Lyrics",width=20  ,height=2  ,fg=fgcolor  ,bg=bgcolor ,font=('Helvetica', 15, ' bold ') ) 
	lbl1.place(x=100, y=300)
	
	#txt1 = tk.Entry(window,width=30,bg="white" ,fg="black",font=('Helvetica', 15, ' bold '))
	#txt1.place(x=400, y=315)

	txt1 = Text(window,height=7, width=80)
	txt1.place(x=400, y=315)

	lbl4 = tk.Label(window, text="Predicted Mood : ",width=20  ,fg=fgcolor,bg=bgcolor  ,height=2 ,font=('Helvetica', 15, ' bold ')) 
	lbl4.place(x=100, y=460)

	message = tk.Label(window, text="" ,bg="white"  ,fg="black",width=20  ,height=2, activebackground = bgcolor ,font=('Helvetica', 15, ' bold ')) 
	message.place(x=400, y=460)

	def clear():
		txt.delete(0, 'end')
		txt1.delete(0, 'end') 
		res = ""
		message.configure(text= res)
		


	def browse():
		res = ""
		message.configure(text= res)
		path=filedialog.askopenfilename()
		print(path)
		txt.delete(0, 'end')
		txt.insert('end',path)
		if path !="":
			print(path)
		else:
			tm.showinfo("Input error", "Select Dataset")	

	def createdataset():
		sym=txt.get()
		res = ""
		message.configure(text= res)
		if sym != "":
			dsc.process(sym)
			tm.showinfo("Input", "Dataset Created Successfully")
		else:
			tm.showinfo("Input error", "Select Dataset")

	def preprocess():
		sym=txt.get()
		res = ""
		message.configure(text= res)
		if sym != "" :
			pre.process(sym)
			print("preprocess")
			tm.showinfo("Input", "Preprocess Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")
	
	def RFprocess():
		sym=txt.get()
		res = ""
		message.configure(text= res)
		if sym != "":
			RF.process(sym)
			tm.showinfo("Input", "RandomForest Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")
		
	def DTprocess():
		sym=txt.get()
		res = ""
		message.configure(text= res)
		if sym != "":
			DT.process(sym)
			print("DT")
			tm.showinfo("Input", "DT Successfully Finished")
		else:
			tm.showinfo("Input error", "Select Dataset")
			
	def predict():
		sym=txt.get()
		sym1=txt1.get('0.0','end')
		res = ""
		message.configure(text= res)
		if sym != "" and sym1 != "" :
			res=sq.process(sym,sym1)
			message.configure(text= res)
		else:
			tm.showinfo("Input error", "Select Dataset and Enter Lyrics")


	browse = tk.Button(window, text="Browse", command=browse  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	browse.place(x=650, y=200)
	
	#browse1 = tk.Button(window, text="Browse", command=browse1  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2, activebackground = "Red" ,font=('times', 15, ' bold '))
	#browse1.place(x=650, y=300)

	clearButton = tk.Button(window, text="Clear", command=clear  ,fg=fgcolor  ,bg=bgcolor1  ,width=20  ,height=2 ,activebackground = "Green" ,font=('times', 15, ' bold '))
	clearButton.place(x=950, y=200)

	#datasetcreate = tk.Button(window, text="DataSet Create", command=createdataset  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	#datasetcreate.place(x=10, y=600)
	
	process = tk.Button(window, text="Preprocess", command=preprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	process.place(x=210, y=600)
	
	rfbutton = tk.Button(window, text="RANDOM FOREST", command=RFprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	rfbutton.place(x=420, y=600)

	DTreebutton = tk.Button(window, text="Decission Tree", command=DTprocess  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	DTreebutton.place(x=620, y=600)

	prebutton = tk.Button(window, text="Predict", command=predict  ,fg=fgcolor   ,bg=bgcolor1   ,width=15  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	prebutton.place(x=820, y=600)

	#quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg=fgcolor   ,bg=bgcolor1  ,width=15  ,height=2, activebackground = "Green" ,font=('times', 15, ' bold '))
	#quitWindow.place(x=1020, y=600)

	window.mainloop()
Home()


