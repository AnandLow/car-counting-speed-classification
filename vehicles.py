from random import randint
import time
import numpy as np
from scipy.spatial import distance as dist
from collections import OrderedDict

class Car:
    tracks=[]
    def __init__(self,i,xi,yi,max_age):
        self.i=i
        self.x=xi
        self.y=yi
        self.tracks=[]
        self.done=False
        self.state='0'
        self.age=0
        self.max_age=max_age
        self.dir=None

    def getTracks(self):
        return self.tracks

    def getId(self): #For the ID
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):  #for x coordinate
        return self.x

    def getY(self):  #for y coordinate
        return self.y

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def timedOut(self):
        return self.done

    def going_LEFT(self, mid_start, mid_end):
        if len(self.tracks)>=2:
            if self.state=='0':
                if self.tracks[-1][0]>mid_end and self.tracks[-2][0]<=mid_end: # nested listing
                    state='1'
                    self.dir='left'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def going_RIGHT(self,mid_start,mid_end):
        if len(self.tracks)>=2:
            if self.state=='0':
                if self.tracks[-1][0]<mid_start and self.tracks[-2][0]>=mid_start:
                    start='1'
                    self.dir='right'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age+=1
        if self.age>self.max_age:
            self.done=True
        return  True

#Class2

class MultiCar:
    def __init__(self,cars,xi,yi):
        self.cars=cars
        self.x=xi
        self.y=yi
        self.tracks=[]
        self.done=False


class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        # initialize a dictionaries to store the timestamp and
        # position of the object at various points
        self.timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.position = {"A": None, "B": None, "C": None, "D": None}
        self.lastPoint = False
        # initialize the object speeds in MPH and KMPH
        self.speedMPH = None
        self.speedKMPH = None
        # initialize two booleans, (1) used to indicate if the
        # object's speed has already been estimated or not, and (2)
        # used to indidicate if the object's speed has been logged or
        # not
        self.estimated = False
        self.logged = False
        # initialize the direction of the object
        self.direction = None
        #initialize vehicle Class
        self.vehicleClass = None

    def calculate_speed(self, estimatedSpeeds):
        # calculate the speed in KMPH and MPH
        self.speedKMPH = np.average(estimatedSpeeds)
        MILES_PER_ONE_KILOMETER = 0.621371
        self.speedMPH = self.speedKMPH * MILES_PER_ONE_KILOMETER


