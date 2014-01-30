#!/usr/bin/python
# -*- coding: utf-8 -*-

import cairo,Image
from operator import itemgetter
import numpy as np
from numpy import sin, cos, pi, arctan2, square,sqrt, logical_not, linspace, array
from numpy.random import random, randint
import gtk, gobject


PI = pi
PII = PI*2.

N = 800 # size of png image
NUM = 200 # number of nodes
BACK = 1. # background color 
GRAINS = 50
STP = 0.001 # scale motion in each iteration by this
MAXFS = 3 # max friendships pr node
ALPHA = 0.05 # opacity of drawn points
ONE = 1./N

RAD = 0.25 # radius of starting circle
FARL  = 0.14 # ignore "enimies" beyond this radius
NEARL = 0.04 # do not attempt to approach friends close than this

FRIENDSHIP_RATIO = 0.5 # probability of frindship attempt


class Render(object):

  def __init__(self):

    self.__init_cairo()
    self.__init_data()

    window = gtk.Window()
    window.resize(N,N)
    window.connect("destroy", gtk.main_quit)
    darea = gtk.DrawingArea()
    darea.connect("expose-event", self.expose)
    window.add(darea)
    window.show_all()

    self.darea = darea

    gobject.idle_add(self.step_wrap)
    gtk.main()

  def __init_data(self):

    self.X = np.zeros(NUM,'float')
    self.Y = np.zeros(NUM,'float')
    self.SX = np.zeros(NUM,'float')
    self.SY = np.zeros(NUM,'float')
    self.R = np.zeros((NUM,NUM),'float')
    self.A = np.zeros((NUM,NUM),'float')
    self.F = np.zeros((NUM,NUM),'byte')

    for i in xrange(NUM):
      the = random()*PII
      x = RAD * sin(the)
      y = RAD * cos(the)
      self.X[i] = 0.5+x
      self.Y[i] = 0.5+y

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,N,N)
    ctx = cairo.Context(sur)
    ctx.scale(N,N)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def expose(self,*args):

    cr = self.darea.window.cairo_create()
    cr.set_source_surface(self.sur,0,0)
    cr.paint()

  def step_wrap(self,*args):

    res = self.step()
    self.expose()

    return res

  def set_distances(self):

    for i in xrange(NUM):
      dx = self.X[i] - self.X
      dy = self.Y[i] - self.Y
      self.R[i,:] = square(dx)+square(dy)
      self.A[i,:] = arctan2(dy,dx)

    sqrt(self.R,self.R)

  def make_friends(self,i):

    cand_num = self.F.sum(axis=1)
    
    if cand_num[i]>=MAXFS:
      return

    cand_mask = cand_num<MAXFS
    cand_mask[i] = False
    cand_ind = cand_mask.nonzero()[0]

    cand_dist = self.R[i,cand_ind].flatten()
    cand_sorted_dist = cand_dist.argsort()
    cand_ind = cand_ind[cand_sorted_dist]

    cand_n = len(cand_ind)

    if cand_n<1:
      return

    #j = randint(cand_n)
    #limit = 1.-(float(j)/cand_n)**2
    #draw = random()

    for k in xrange(cand_n):

      if random() < 0.1:

        j = cand_ind[k]
        self.F[[i,j],[j,i]] = True
        return

  def render_connections(self):

    self.ctx.set_source_rgba(0,0,0,ALPHA)

    indsx,indsy = self.F.nonzero()
    mask = indsx >= indsy 
    for i,j in zip(indsx[mask],indsy[mask]):
      a = self.A[i,j]
      d = self.R[i,j]
      scales = np.random.random(GRAINS)*d
      xp = self.X[i] - scales*cos(a)
      yp = self.Y[i] - scales*sin(a)

      for x,y in zip(xp,yp):
        self.ctx.rectangle(x,y,ONE,ONE)
        self.ctx.fill()

  def step(self):

    self.set_distances()
    
    self.SX[:] = 0.
    self.SY[:] = 0.
    
    for i in xrange(NUM):
      xF = logical_not(self.F[i,:])
      d = self.R[i,:]
      a = self.A[i,:]
      near = d > NEARL
      near[xF] = False
      far = d < FARL
      far[near] = False
      near[i] = False
      far[i] = False
      speed = FARL - d[far]

      self.SX[near] += cos(a[near])
      self.SY[near] += sin(a[near])
      self.SX[far] -= speed*cos(a[far])
      self.SY[far] -= speed*sin(a[far])

    self.X += self.SX*STP
    self.Y += self.SY*STP

    #if random()<FRIENDSHIP_RATIO:
    i = randint(NUM)
    self.make_friends(i)

    self.render_connections()

    return True

def main():

  render = Render()
  

if __name__ == '__main__' :
  main()

