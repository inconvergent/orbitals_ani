#!/usr/bin/python
# -*- coding: utf-8 -*-

import cairo,Image
from operator import itemgetter
import numpy as np
from numpy import sin, cos, pi, arctan2, sqrt, logical_not
from numpy.random import random
import gtk, gobject


PI = pi
PII = PI*2.

N = 800 # size of png image
NUM = 100 # number of nodes
BACK = 1. # background color 
RAD = 0.2 # radius of starting circle
GRAINS = 150
STP = 0.001 # scale motion in each iteration by this
MAXFS = 50 # max friendships pr node
ALPHA = 0.05 # opacity of drawn points
ONE = 1./N

FARL  = 0.15
NEARL = 0.02


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
      self.R[i,:] = sqrt(dx*dx+dy*dy)
      self.A[i,:] = arctan2(dy,dx)

  def make_friends(self,i):
    
    if self.F[i,:].sum() > MAXFS:
      return

    r = []
    for j in xrange(NUM):
      if i != j and self.F[j,:].sum() < MAXFS\
        and not self.F[j,i]:
          r.append((self.R[i,j],j))
    if not len(r):
      return
    r = sorted(r, key=itemgetter(0))

    index = len(r)-1
    for k in xrange(len(r)):
      if random() < 0.1:
        index = k
        break

    self.F[i,r[index][1]] = True
    self.F[r[index][1],i] = True
    return

  def render_connection_points(self):

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

    self.make_friends(int(random()*NUM))

    self.render_connection_points()

    return True

def main():

  render = Render()
  

if __name__ == '__main__' :
  main()

