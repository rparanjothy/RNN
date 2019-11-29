import random
import itertools as it


def genCoor(s):
    x=random.randint(10,200)
    y=random.randint(20,204)
    z=random.randint(-20,20)

    return ([x,y,z],-1 if z<1 else 1)

ct=it.count()


a=it.imap(genCoor,ct)

coor=it.islice(a,0,300)
v=it.islice(a,301,320)

trainData=coor
valData=v
# for i in xrange(100):
#     print trainData.next()