import numpy as np
import mlpy

### Helper functions ###

#Extracts the feature from the XML file
def extract(feature, text):
	pattern = "<" + feature + ">(.*)</" + feature + ">"
	return re.findall(pattern,text,re.DOTALL)[0].strip()





def parseDataPoints(directory,regexParse):

#Dictionaries mapping feature/word to index in vocab and index to feature/word
	vocab = {}
	indexToWord = {}

	##Store full data in here
	dataPoints = []

	for root, dirs, files in os.walk(directory):
		for file in files:
			fullFile = os.path.join(root,file)
			if file.endswith(".xml"):
				text = open(fullFile).read()
				typ = extract("Type",text)
				duration = extract("Duration",text)
				stack = extract("ErrorS",text)
				message = extract("ErrorM",text)
				dataPoints.append([typ, duration, stack, message])
				for word in stack.split(regexParse) + message.split(regexParse):
					if word not in vocab:
				 		vocab[word] = len(vocab)
				 		indexToWord[len(vocab)-1] = word



#Return the x and y
#X is a matrix n x p which represents a set of n samples in R^P
#Y is a vector n which represents the target values (integers in classification problems, 1 product error, 0 automation error)
#Data is the data which is returned from parseDataPoints

def packageData(fullData,regexParse,vocab):
	X = []
	Y = []
	for data in fullData:
		[typ, duration, stack, message] = data
		dataPoint = []
		dataPoint.append(float(duration))
		for word in stack.split(regexParse) + message.split(regexParse):
			dataPoint.append(vocab[word])
		X.append(dataPoint)
		if typ == "true":
			Y.append(1)
		if typ =="false":
			Y.append(0)

	return [X,Y]


#Prints the accuracy of the model using the difference in predicted labels vs actual labels
def printAccuracy(pred_labels, actual_label, model_type):
	accuracy = 0.0
	for a,b in zip(pred_labels,actual_label):
		if a == b:
			accuracy+=1
	accuracy/=len(Y)
	print "With model", model_type, ", the accuracy is: ", accuracy*100, "%"




directory = raw_input("What directory are the XML files located:\n")
regexParse = raw_input("How would you like to parse the words, leave it blank if you would like to parse by whitespace:\n")
model_type = LDAC
if(regexParse == ""):
	regexParse = None
[vocab,indexToWord,fullDataPoints] = parseDataPoints(directory,regexParse)
[X,Y] = packageData(fullDataPoints,regexParse,vocab)
print len(X), len(X[0]), len(Y)

ldac = mlpy.LDAC()
ldac.learn(X,Y)
Y_PRED = ldac.predict(X)


printAccuracy(Y_PRED, Y, model_type)








