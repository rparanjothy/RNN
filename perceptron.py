'''
Ram Paranjothy
Nov 28 2019
Rishi Neural Net
'''
import random

class Perceptron(object):
    inputCt=None
    i=[]
    weight=[]
    bias=0
    lrate=0.01
    target=None

    def __init__(self,iCt,oCt):
        # Activate only once. 
        #  so for the second set of training data, the weights from first should be adjusted.
        self.inputCt=iCt
        self.activate()

        # initialize weights when a perceptron is created.
        #  so we can tune it as we train it
        # if not isinstance(i,list): 
        #     raise Exception("Input should be a list")
        # self.i=i
        # self.activate()
        # self.normalize()
            
        # self.output=output
  
    def activate(self):
        # randomly initialize weigths between 1 and -1
        self.weight=[format(random.uniform(-1,1),"1.2f") for _ in range(self.inputCt)]

    def guess(self):
        # compute the weighted sum for input and map to 1 and -1
        x=reduce(lambda x,o:o+x,map(lambda x:float(x[0])*float(x[1]), zip(self.weight,self.i)))
        return 1 if x>=0 else -1

    def setData(self,x):
        #this will be used to set the input list and the target label to each observation
        trainInput=x[0]
        trainTgt=x[1]
        self.i=trainInput
        self.target=trainTgt

    def normalize(self):
        maxInput=max([abs(_) for _ in self.i])
        if maxInput>1 or abs(min(self.i)) > 1:
            self.i=[format(float(_)/float(maxInput),'1.2f') for _ in self.i]
            print("*"*5)
            # print('Normalized inputs : {0}'.format(self.i))

    def train(self,x):
        self.setData(x)
        self.normalize()
        self._train()

    def _train(self):
        # This method mutates the weights if guess and target doesnt match
        print("\tCurrent Input - {0}".format(self.i))
        print("\tCurrent weights - {0}".format(self.weight))
        g=self.guess()
        error=float(g)-float(self.target)
        print ("\tError - {0} : Guess: {1} Target: {2}".format(error,g,self.target))
        # now tune weights per error -> weight = weight + error*each weight
        if float(error) != 0.0:  
            # increment if error is + and decrement if error is -ve
            self.weight=map(lambda x:(float(x)*float(error)*float(self.lrate))+float(x),self.i)  if error < 0 else map(lambda x:(float(x)*float(error)*float(self.lrate))-float(x),self.i)
            print("***** Tuning weights - {0}".format(self.weight))
            self._train()
        else:
            print("\tFinal weights : {0}".format(self.weight))
            return error
       
    def getWeights(self):
        return self.weight


    def predict(self,inputs,weight):
        '''
        THis is the predict guy
        '''
        maxInput=max([abs(_) for _ in inputs[0]])
        lbl=inputs[1]
        if maxInput>1 or abs(min(self.i)) > 1:
            inputsNormalized=[format(float(_)/float(maxInput),'1.2f') for _ in inputs[0]]
            print("*"*5)

        # for this to be accurate, your weight shuld have been trained.
        x=reduce(lambda x,o:o+x,map(lambda x:float(x[0])*float(x[1]), zip(weight,inputsNormalized)))
        out= 1 if x>=0 else -1
        # return ('Input: {0}| Label: {1}| Prediction: {2}| Matching: {3}'.format(inputs[0],lbl,out,lbl==out))
        return {
                'input':inputs[0],
                'label':lbl,
                'prediction':out,
                'matched':lbl==out
        }



if __name__ == "__main__":
    trainingData=[([2,4,5],1),([2,4,-5],-1),([2,4,-44],-1),([2,4,55],1),([22,24,-55],-1),
    ([12,14,5],1),([22,24,-5],-1),([32,34,-44],-1),([42,4,55],1),([22,24,-55],-1),([22,24,55],1)]
    p=Perceptron(3,1)    
    for d in trainingData:
        p.train(d)

    print("Training Done...")

    tunedWeight=p.getWeights()
    print("Tuned Weight: {0}".format(tunedWeight))

    validationData=[([20,40,50],1),([20,40,-50],-1),([200,400,-440],-1),([200,400,550],1),([202,204,-505],-1)]
    
    print([p.predict(vData,tunedWeight) for vData in validationData])
        # print(p.predict(vData,tunedWeight))



        
        