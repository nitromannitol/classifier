import os, os.path,re,math, random, sys
from operator import mul

##### Helper functions ##########

#Extracts the feature from the XML file
def extract(feature, text):
	pattern = "<" + feature + ">(.*)</" + feature + ">"
	return re.findall(pattern,text,re.DOTALL)[0].strip()


#Gets the x most common words in the array using the indexToWord dictionary
def getCommon(array, dic, x):
	list = []
	for i in xrange(0,x):
		maxNum = max(array)
		wordIndex = array.index(maxNum)
		list.append(dic[wordIndex])
		array[wordIndex] = 0
	return list



#Get the mean and varience of a list of numbers 
def getMean(array, total):
	return sum(array)/total

def getVar(array,total,mean):
	var = 1.0
	for x in array:
		var += ((float(x) - mean)**2)/float(total)
	return var


#Calculates the probability that the sample is taken from the given distribution
# Maybe? Add cC for continuity correction
def calculateProb(sample,mean,var):
	pi = 3.1415926535897932384626433
	#var+= cC
	return 1/(math.sqrt(2*pi*var))*math.exp((-((sample-mean)**2)/(2*var)))


#Calculates the numerator of the posterior calculation
def calculatePosteriorNum(prior, durationMean, durationVar, wordMeanArray, wordVarArray, inputDuration, inputWordArray):
	pi = 3.1415926535897932384626433
	durationProb = calculateProb(inputDuration,durationMean,durationVar)
	wordProb = []
	for mean,var,sample in zip(wordMeanArray, wordVarArray, inputWordArray):
		wordProb.append(calculateProb(sample,mean,var))
#	return prior #only taking into account probability of appearance
#	return durationProb*prior #only taking into account duration and prior
	return durationProb*reduce(mul,wordProb,1)*prior #taking into account word apperance




#Calculates the posterior for both classes
def calculatePosterior(priorProd, durationMeanProd, durationVarProd, wordMeansProd, wordVarsProd, 
						priorAuto, durationMeanAuto, duratonVarAuto, wordMeansAuto, wordVarsAuto,
						inputDuration, inputWordArray):

	prodNum = calculatePosteriorNum(priorProd, durationMeanProd, durationVarProd, wordMeansProd, wordVarsProd, inputDuration, inputWordArray)
	autoNum = calculatePosteriorNum(priorAuto, durationMeanAuto, duratonVarAuto, wordMeansAuto, wordVarsAuto, inputDuration, inputWordArray)
	evidence = prodNum + autoNum +1

	prodPosterior = prodNum/(evidence)
	autoPosterior = autoNum/(evidence)

	return [prodPosterior, autoPosterior]

#Determines the error class of an input sample given the model statistics
def classify(productPrior, pDurAverage,pDurVar,pWordAv,pWordVar, 
					 autoPrior, aDurAverage, aDurVar, aWordAv, aWordVar,
						 inputDuration, inputWordArray):
	pos =  calculatePosterior(productPrior, pDurAverage,pDurVar,pWordAv,pWordVar, 
						 autoPrior, aDurAverage, aDurVar, aWordAv, aWordVar,
						 inputDuration, inputWordArray)

	if pos[0] > pos[1]:
		print "Product Error"
		return "true"  ##Product Error
	if pos[1] > pos[0]:  ##Automation Error
		print "Automation Error"
		return "false"
	##In this case both probabilities are equal so randomly choose one
	if random.randint(1,2) == 1:
		 return "false"
	else:
		 return "true" 
	

######################################################################################################################################

directory = raw_input("Input directory\n")


#Dictionaries mapping word to index in vocab and index to word
vocab = {}
indexToWord = {}

#Product Bug and Auto Bug statistics 
pDuration = []
aDuration = []
numProdBug = 0
numAutoBug = 0
totalWords = 0

total = 0


#Build the dictionaries
for root, dirs, files in os.walk(directory):
	for file in files:
		fullFile = os.path.join(root,file)
		if file.endswith(".xml"):
			total = total + 1
			text = open(fullFile).read()
			stack = extract("ErrorS",text)
			message = extract("ErrorM",text)
			for word in stack.split() + message.split():
				totalWords = totalWords + 1
				if word not in vocab:
				 	vocab[word] = len(vocab)
				 	indexToWord[len(vocab)-1] = word

##create a matrix for of word / training examples

pArray = [[1]*total for j in range(len(vocab))]
aArray = [[1]*total for j in range(len(vocab))]

#Weird bug below lol
#pVarArray = [[0]*total] * len(vocab)
#aVarArray = [[0]*total] * len(vocab)

currExampleIndex = 0

#Populate the data
for root, dirs, files in os.walk(directory):
	for file in files:
		full_file = os.path.join(root,file)
		if file.endswith(".xml"):
			text = open(full_file).read()

			#Collect the features
			typ = extract("Type",text)
			duration = extract("Duration",text)
			stack = extract("ErrorS",text)
			message = extract("ErrorM",text)

			#Update the statistics
			#For the varience we need to collect a full array for each word
			if typ == "true":
				numProdBug+=1
				pDuration.append(float(duration))
				for word in stack.split() + message.split():
					pArray[vocab[word]][currExampleIndex]+=1

			if typ == 'false':
				 numAutoBug+= 1
				 aDuration.append(float(duration))
				 for word in stack.split() + message.split():
					aArray[vocab[word]][currExampleIndex]+=1

			currExampleIndex = currExampleIndex + 1


#Calculate mean duration and sample varience
pDurAverage = getMean(pDuration,total)
pDurVar = getVar(pDuration,total,pDurAverage)
aDurAverage = getMean(aDuration,total)
aDurVar = getVar(aDuration, total, aDurAverage)


#Calculate mean word apperance and sample varience for each word

aWordAv = [getMean(x,total) for x in aArray]
pWordAv = [getMean(x,total) for x in pArray]

aWordVar = [getVar(x,total,getMean(x,total)) for x in aArray]
pWordVar = [getVar(x,total,getMean(x,total)) for x in pArray]


#Calulate priors for product bug and automation bug
productPrior = (float(numProdBug)/total)
autoPrior = (float(numAutoBug)/total)


print "Product Bug Percent: ", productPrior*100, "%"
print "Product Bug Average Duration: ", pDurAverage
#print "Product Bug Average Apperance: ", pWordAv
print "Product Bug Most Common Words", getCommon(pArray,indexToWord,3)	

print "Automation Bug Percent: ", autoPrior*100, "%"
print "Automation Bug Average Duration: ", aDurAverage
#print "Automation Bug Average Appereance: ", aWordAv
print "Automation Bug Most Common Words: ", getCommon(aArray,indexToWord,3)



#To do: cross-validation accuracy testing

# classify(productPrior, pDurAverage,pDurVar,pWordAv,pWordVar, 
# 						 autoPrior, aDurAverage, aDurVar, aWordAv, aWordVar,
# 						 inputDuration, inputWordArray)

#Note training and testing on the same data set introduces heavy bias, must do cross validation 

accuracyCount = 0
accuracyCount2 = 0



#Checking against itself 
for root, dirs, files in os.walk(directory):
	for file in files:
		full_file = os.path.join(root,file)
		if file.endswith(".xml"):
			text = open(full_file).read()

			#Collect the features
			typ = extract("Type",text)
			duration = float(extract("Duration",text))
			stack = extract("ErrorS",text)
			message = extract("ErrorM",text)

			featureArray = [1]*len(vocab)

			#Create an array of word feature vector with numbers in vocab positons
			for word in stack.split() + message.split():
					featureArray[vocab[word]]+=1

			predictType = classify(productPrior, pDurAverage,pDurVar,pWordAv,pWordVar, 
						 autoPrior, aDurAverage, aDurVar, aWordAv, aWordVar,
						 duration, featureArray)

			if predictType == typ:
				accuracyCount+=1 


#Checking against random subsets of the data
for root, dirs, files in os.walk(directory):
	for file in files:
		full_file = os.path.join(root,file)
		if file.endswith(".xml") and random.randint(1,5) == 1:  #Pick half the data

			text = open(full_file).read()
			#Collect the features
			typ = extract("Type",text)
			duration = float(extract("Duration",text))
			stack = extract("ErrorS",text)
			message = extract("ErrorM",text)

			featureArray = [1]*len(vocab)

			#Create an array of word feature vector with numbers in vocab positons
			for word in stack.split() + message.split():
					featureArray[vocab[word]]+=1

			predictType = classify(productPrior, pDurAverage,pDurVar,pWordAv,pWordVar, 
						 autoPrior, aDurAverage, aDurVar, aWordAv, aWordVar,
						 duration, featureArray)

			if predictType == typ:
				accuracyCount2+=1 


print "Accuracy when checking against itself: ", (float(accuracyCount)/total)*100, "%"
print "Accuracy when checking against randomly selected subsets of the data: ", (float(accuracyCount2)/total)*100, "%"



