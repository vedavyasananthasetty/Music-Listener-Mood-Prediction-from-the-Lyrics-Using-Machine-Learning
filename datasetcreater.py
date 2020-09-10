import csv
from PyLyrics import *
import re
import os

data=[]

def process(path):
	with open(path, 'r') as csvFile:
		reader = csv.reader(csvFile)
		i=0
		for row in reader:
			if i!=0:
				print(row[0],row[1],row[2],i)
				print(PyLyrics.getLyrics(row[0],row[1]))
				data.append([row[0],row[1],row[2],PyLyrics.getLyrics(row[0],row[1])])
			i=i+1
	csvFile.close()		

	with open('dataset.csv', 'w', newline='') as f:
	    writer = csv.writer(f)
	    writer.writerow(["Artists","Album","Genre","Lyrics"])
	    i=0
	    for row in data:
	    	if i!=0:
	    		writer.writerow(row)
	    	i=i+1
	f.close()
