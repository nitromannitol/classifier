import os, os.path,re,math, random, sys
from operator import mul

###Given the directory of testng data results, this script generates a Naive Bayes model 

##### Helper functions ##########

#Extracts the feature from the XML file
def extract(feature, text):
	pattern = "<" + feature + ">(.*)</" + feature + ">"
	return re.findall(pattern,text,re.DOTALL)[0].strip()


#Gets the x most common words in the array using the indexToWord dictionary
#Not used right now
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
	return float(sum(array)/total)

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
		wordProb.append(calculateProb(float(sample),float(mean),float(var)))
#	return prior #only taking into account probability of appearance
#	return durationProb*prior #only taking into account duration and prior
#	return reduce(mul,wordProb,1)*prior
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
		#print "Product Error"
		return "true"  ##Product Error
	if pos[1] > pos[0]:  ##Automation Error
		#print "Automation Error"
		return "false"
	##In this case both probabilities are equal so randomly choose one
	if random.randint(1,2) == 1:
		 return "false"
	else:
		 return "true" 
	

#dataSet is an array of training examples 
#regexParse is how to parse the words
#Returns priors, and average and varience for each of the features
def train(dataSet, regexParse):

	#Product Bug and Auto Bug statistics 
	pDuration = []
	aDuration = []
	numProdBug = 0
	numAutoBug = 0

	##create a matrix for of word / training examples
	pArray = [[1]*len(dataSet) for j in range(len(vocab))]
	aArray = [[1]*len(dataSet) for j in range(len(vocab))]

	currExampleIndex = 0
	for data in dataSet:
		typ = data[0]
		duration = data[1]
		stack = data[2]
		message = data[3]


		#Update the statistics
		#For the varience we need to collect a full array for each word
		if typ == "true":
			numProdBug+=1
			pDuration.append(float(duration))
			for word in stack.split(regexParse) + message.split(regexParse):
				pArray[vocab[word]][currExampleIndex]+=1

		if typ == 'false':
			 numAutoBug+= 1
			 aDuration.append(float(duration))
			 for word in stack.split(regexParse) + message.split(regexParse):
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

	return [productPrior, autoPrior, pDurAverage, aDurAverage, pDurVar, aDurVar, aWordAv, pWordAv, aWordVar, pWordVar]


#Build the dictionaries and populate the datapoints
def parseDataPoints(directory, regexParse):
	#Dictionaries mapping feature/word to index in vocab and index to feature/word
	vocab = {}
	indexToWord = {}

	##Store files in here
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


#Tests the accuracy of the statistics generated by the model on the testData
def test(testData,stats, vocab, regexParse):
	##Unbox the stats
	productPrior = float(stats[0])
	autoPrior = float(stats[1])
	pDurAverage = float(stats[2])
	aDurAverage = float(stats[3])
	pDurVar = float(stats[4])
	aDurVar = float(stats[5])
	aWordAv = stats[6]
	pWordAv = stats[7]
	aWordVar = stats[8]
	pWordVar = stats[9]

	numProdBug = 0.0
	numAutoBug = 0.0
	accuracyCount = 0.0

	for data in testData:
		typ = data[0]
		duration = float(data[1])
		stack = data[2]
		message = data[3]

		featureArray = [1]*len(vocab)

		#Create an array of word feature vector with numbers in vocab positons
		for word in stack.split(regexParse) + message.split(regexParse):
				featureArray[vocab[word]]+=1.0

		predictType = classify(productPrior, pDurAverage,pDurVar,pWordAv,pWordVar, 
					 autoPrior, aDurAverage, aDurVar, aWordAv, aWordVar,
					 duration, featureArray)

		if predictType == "true":
			numProdBug+=1
		else: numAutoBug+=1
		if predictType == typ:
			accuracyCount+=1 	


	return [accuracyCount/len(testData), numProdBug, numAutoBug]

def printStats(testStats, regexParse):
	if(regexParse == None):
		print "Parsing by whitespace"
	else:
		print "Parsing by: ", regexParse
	print "Accuracy: ", testStats[0]*100, "%"
	total = testStats[1] + testStats[2]
	print "Classifier predicted: ", (float(testStats[1])/total)*100, "%", "product bugs and ", (testStats[2]/total)*100, "%",  "automation bugs"


#Test the data on randomly selected subsets of fullDataPoints, returns the array of averaged stats that are of the 
#same format as the one returned by test
def randomlyTest(timesRun, stats, fullDataPoints):
	totalAccuracy = [0, 0, 0]

	#Test the data on randmoly selected subsets timesRun times
	for i in xrange(0,timesRun):

		testDataPoints = []

		while(len(testDataPoints) <=5): ##Ensure that we test at least 5 data points
			randomStart = random.randint(0,len(fullDataPoints)/2)
			randomEnd = random.randint(randomStart,len(fullDataPoints))
			testDataPoints = fullDataPoints[randomStart:randomEnd]


		#Test the model on the same dataPoints
		testStats = test(testDataPoints, stats, vocab, regexParse)
		totalAccuracy[0]+=testStats[0]
		totalAccuracy[1]+=testStats[1]
		totalAccuracy[2]+=testStats[2]

	return [x/timesRun for x in totalAccuracy]

######################################################################################################################################

directory = raw_input("What directory are the XML files located:\n")
regexParse = raw_input("How would you like to parse the words, leave it blank if you would like to parse by whitespace:\n")
timesRun = int(raw_input("How many times would you like to run the test:\n"))
while(timesRun <= 0):
	timesRun = int(raw_input("Please enter an integer greater than 0:\n"))

if(regexParse == ""):
	regexParse = None

#Build the dictionaries and populate the datapoints
[vocab,indexToWord,fullDataPoints] = parseDataPoints(directory, regexParse)
total = len(fullDataPoints)


#Collect the statistics, i.e., train the model on the dataset provided
stats = train(fullDataPoints, regexParse)

averageAccuracy = randomlyTest(timesRun,stats,fullDataPoints)

#Print out the test statistics 
printStats(averageAccuracy, regexParse) 



#print "Product Bug Percent: ", productPrior*100, "%"
#print "Product Bug Average Duration: ", pDurAverage
#print "Product Bug Average Apperance: ", pWordAv
#print "Product Bug Most Common Words", getCommon(pArray,indexToWord,3)	

#print "Automation Bug Percent: ", autoPrior*100, "%"
#print "Automation Bug Average Duration: ", aDurAverage
#print "Automation Bug Average Appereance: ", aWordAv
#print "Automation Bug Most Common Words: ", getCommon(aArray,indexToWord,3)



