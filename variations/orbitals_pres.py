#!/usr/bin/python
# -*- coding: utf-8 -*-

import cairo,Image
from operator import itemgetter
import numpy as np
from numpy import sin, cos, pi, arctan2, square,sqrt, logical_not, linspace, array
from numpy.random import random, randint
import gtk, gobject

## comment out to get new results on each run
np.random.seed(1)


PI = pi
PII = PI*2.

N = 800 # size of png image
NUM = 200 # number of nodes
BACK = 1. # background color 
GRAINS = 5
MAXFS = 5 # max friendships pr node
ALPHA = 0.05 # opacity of drawn points
ONE = 1./N

STP = ONE/40.

RAD = 0.28 # radius of starting circle
FARL  = 0.14 # ignore "enemies" beyond this radius
NEARL = 0.03 # do not attempt to approach friends close than this

UPDATE_NUM = 100

FRIENDSHIP_RATIO = 0.1 # probability of friendship dens
FRIENDSHIP_INITIATE_PROB = 0.3 # probability of friendship initation attempt

BACK = [1]*3
FRONT = [0,0,0,0.07]

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

    self.num_img = 0
    self.itt = 0

    self.cr = self.darea.window.cairo_create()

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
    ctx.set_source_rgb(*BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def __get_colors(self,f):
    scale = 1./255.
    im = Image.open(f)
    w,h = im.size
    rgbim = im.convert('RGB')
    res = []
    for i in xrange(0,w):
      for j in xrange(0,h):
        r,g,b = rgbim.getpixel((i,j))
        res.append((r*scale,g*scale,b*scale))

    np.random.shuffle(res)
    self.colors = res
    self.n_colors = len(res)

  def expose(self,*args):

    #cr = self.darea.window.cairo_create()
    self.cr.set_source_surface(self.sur,0,0)
    self.cr.paint()

  def step_wrap(self,*args):

    res, added_new = self.step()

    if not self.itt%UPDATE_NUM:
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

    for k in xrange(cand_n):

      if random()<FRIENDSHIP_RATIO:

        j = cand_ind[k]
        self.F[[i,j],[j,i]] = True
        return

  def render_connections(self):

    # everything is black
    self.ctx.set_source_rgba(*FRONT)

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

    self.itt+=1

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

      #speed = (FARL - d[far])
      speed = (FARL - d[far])/FARL

      self.SX[near] += cos(a[near])
      self.SY[near] += sin(a[near])

      self.SX[far] -= speed*cos(a[far])
      self.SY[far] -= speed*sin(a[far])

    self.X += self.SX*STP
    self.Y += self.SY*STP

    if random()<FRIENDSHIP_INITIATE_PROB:
      i = randint(NUM)
      self.make_friends(i)

    self.render_connections()

    return True, True

def main():

  render = Render()
  

if __name__ == '__main__' :
  main()

