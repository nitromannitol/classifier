import numpy as np
import mlpy
import os, os.path,sys,re


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
	return [vocab, indexToWord, dataPoints]



#Return the x and y
#X is a matrix n x p which represents a set of n samples in R^P
#Y is a vector n which represents the target values (integers in classification problems, 1 product error, 0 automation error)
#Data is the data which is returned from parseDataPoints

def packageData(fullData,regexParse,vocab, indexToWord):
	X = [ [0.1 for z in range(len(vocab) + 1)   ] for j in range(len(fullData))]
	Y = []
	count = 0
	for data in fullData:
		[typ, duration, stack, message] = data
		#dataPoint = [1*(len(vocab)+1) for j in range(len(vocab) +1)]
		X[count][0] = float(duration)
		for word in stack.split(regexParse) + message.split(regexParse):
			X[count][vocab[word]+1]+=1
		if typ == "true":
			Y.append(1)
		if typ =="false":
			Y.append(0)
		count+=1

	return [X,Y]


#Prints the accuracy of the model using the difference in predicted labels vs actual labels
def printAccuracy(pred_labels, actual_label, model_type):
	accuracy = 0.0
	for a,b in zip(pred_labels,actual_label):
		if a == b:
			accuracy+=1
	accuracy/=len(Y)
	print "With model type,", model_type, ", the accuracy is: ", accuracy*100, "%"



#Divides the dataset into pieces determined by the divideMarker
def divide(X,Y, divideMarker):
	divide = int(float(len(X))*divideMarker)
	return X[:divide], X[divide:], Y[:divide], Y[divide:]



#Tests the model without bias
def testModel(model, X, Y, modelName):
	[xTrain, xTest, yTrain, yTest] = divide(X, Y, 0.1)
	try:
		model.learn(xTrain, yTrain)
		printAccuracy(model.pred(xTest),yTest, modelName)
	except:
		print "Error in", modelName




directory = raw_input("What directory are the XML files located:\n")
regexParse = raw_input("How would you like to parse the words, leave it blank if you would like to parse by whitespace:\n")
if(regexParse == ""):
	regexParse = None
[vocab,indexToWord,fullDataPoints] = parseDataPoints(directory,regexParse)
[X,Y] = packageData(fullDataPoints,regexParse,vocab, indexToWord)


testModel(mlpy.Perceptron(alpha=0.1, thr=0.05, maxiters=1000), X, Y, "Perceptron")
testModel(mlpy.ElasticNetC(lmb=0.01, eps=0.001),X,Y, "ElasticNet")
testModel(mlpy.LibLinear(solver_type='l2r_l2loss_svc_dual', C=1), X, Y, "LibLinear")
testModel(mlpy.DLDA(delta=0.1), X, Y, "DLDA")
testModel(mlpy.Golub(), X, Y, "Golub")
testModel(mlpy.Parzen(),X,Y,"Parzen")
testModel(mlpy.KNN(2),X,Y,"KNN")
testModel(mlpy.ClassTree(),X,Y,"Classification Tree")
testModel(mlpy.MaximumLikelihoodC(),X,Y,"Maximum Likelihood Classifer")













