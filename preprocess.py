import csv
import re
import os
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from textblob import TextBlob

cachedStopWords = stopwords.words("english")
jaz=[]
roc=[]
countr=[]
christ=[]
pops=[]
hip=[]
data=[]

def process(path):
	with open(path, 'r') as csvFile:
		reader = csv.reader(csvFile)
		i=0
		for row in reader:
			if i!=0:
				print(row[0],row[1],row[2],i)
				all_words=all_words=row[3]
				all_words = re.sub(r'[\(\[].*?[\)\]]', '', all_words)
				all_words = os.linesep.join([s for s in all_words.splitlines() if s])
				#words = all_words.split(" ")
				filtered_words = ' '.join([word for word in all_words.split() if word not in cachedStopWords])
				blob = TextBlob(filtered_words)
				if blob.sentiment.polarity>0.0:
					data.append([row[0],row[1],row[2],filtered_words,"Happy",1])
				else:
					data.append([row[0],row[1],row[2],filtered_words,"Sad",0])
			if row[2]=="Jazz":
				jaz.append(filtered_words)
			if row[2]=="Rock":
				roc.append(filtered_words)
			if row[2]=="Christian":
				christ.append(filtered_words)
			if row[2]=="Country":
				countr.append(filtered_words)
			if row[2]=="Pop":
				pops.append(filtered_words)
			if row[2]=="Hip Hop/Rap":
				hip.append(filtered_words)
				
			i=i+1
	csvFile.close()		
	print(len(jaz))
	print(len(roc))
	print(len(countr))
	print(len(christ))
	print(len(pops))
	print(len(hip))	
	
	text=""
	for m in jaz:
		text=text+m
	# Create and generate a word cloud image:
	wordcloud = WordCloud().generate(text)
	wordcloud.to_file("results/Jazz.png")
	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	text=""
	for m in roc:
		text=text+m
	# Create and generate a word cloud image:
	wordcloud = WordCloud().generate(text)
	wordcloud.to_file("results/Rock.png")
	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	text=""
	for m in countr:
		text=text+m
	# Create and generate a word cloud image:
	wordcloud = WordCloud().generate(text)
	wordcloud.to_file("results/Country.png")
	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	text=""
	for m in christ:
		text=text+m
	# Create and generate a word cloud image:
	wordcloud = WordCloud().generate(text)
	wordcloud.to_file("results/Christian.png")
	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	text=""
	for m in pops:
		text=text+m
	# Create and generate a word cloud image:
	wordcloud = WordCloud().generate(text)
	wordcloud.to_file("results/Pop.png")
	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	text=""
	for m in hip:
		text=text+m
	# Create and generate a word cloud image:
	wordcloud = WordCloud().generate(text)
	wordcloud.to_file("results/Hip.png")
	# Display the generated image:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")
	plt.pause(5)
	plt.show(block=False)
	plt.close()



	with open('cleaneddataset.csv', 'w', newline='') as f:
	    writer = csv.writer(f)
	    writer.writerow(["Artists","Album","Genre","Lyrics","Mood","MoodValue"])
	    i=0
	    for row in data:
	    	if i!=0:
	    		writer.writerow(row)
	    	i=i+1
	f.close()
