#!/usr/bin/env python

#from Box2D import *
import Box2D.b2 as b2
import math

import pygame
from pygame.locals import K_RIGHT, K_LEFT, KEYDOWN, KEYUP
import numpy as np
import copy
import scipy.linalg

import numpy as np
import math
import matplotlib.pyplot as plt


class CartPole:
    def __init__(self):
        self.trackWidth = 10.0
        self.cartWidth = 0.3
        self.cartHeight = 0.2
        self.cartMass = 10
        self.poleMass = 1
        self.force = 0.2
        self.trackThickness = self.cartHeight
        self.poleLength = 0.5
        self.poleThickness = 0.04

        self.screenSize = (640,480) #origin upper left
        self.worldSize = (float(self.trackWidth),float(self.trackWidth)) #origin at center

        self.world = b2.world(gravity=(0,-9.81),doSleep=True)
        self.framesPerSecond = 10 # used for dynamics update and for graphics update
        self.velocityIterations = 8
        self.positionIterations = 6

        # Make track bodies and fixtures
        self.trackColor = (100,100,100)
        poleCategory = 0x0002

        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.trackWidth/2,self.trackThickness/2)),
                          friction=0.000, categoryBits=0x0001, maskBits=(0xFFFF & ~poleCategory))
        self.track = self.world.CreateStaticBody(position = (0,0), 
                                                 fixtures=f, userData={'color': self.trackColor})
        self.trackTop = self.world.CreateStaticBody(position = (0,self.trackThickness+self.cartHeight*1.1),
                                                    fixtures = f)

        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.trackThickness/2,self.trackThickness/2)),
                          friction=0.000, categoryBits=0x0001, maskBits=(0xFFFF & ~poleCategory))
        self.wallLeft = self.world.CreateStaticBody(position = (-self.trackWidth/2+self.trackThickness/2, self.trackThickness),
                                               fixtures=f, userData={'color': self.trackColor})
        self.wallRight = self.world.CreateStaticBody(position = (self.trackWidth/2-self.trackThickness/2, self.trackThickness),
                                                fixtures=f,userData={'color': self.trackColor})

        # Make cart body and fixture
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.cartWidth/2,self.cartHeight/2)),
                          density=self.cartMass, friction=0.000, restitution=0.5, categoryBits=0x0001,
                          maskBits=(0xFFFF & ~poleCategory))
        self.cart = self.world.CreateDynamicBody(position=(0,self.trackThickness),
                                            fixtures=f, userData={'color':(20,200,0)})

        # Make pole pody and fixture
        # Initially pole is hanging down, which defines the zero angle.
        f = b2.fixtureDef(shape=b2.polygonShape(box=(self.poleThickness/2, self.poleLength/2)),
                          density=self.poleMass, categoryBits=poleCategory)
        self.pole = self.world.CreateDynamicBody(position=(0,self.trackThickness+self.cartHeight/2+self.poleThickness-self.poleLength/2),
                                            fixtures=f, userData={'color':(200,20,0)})

        # Make pole-cart joint
        self.world.CreateRevoluteJoint(bodyA = self.pole, bodyB = self.cart,
                                  anchor=(0,self.trackThickness+self.cartHeight/2+self.poleThickness))


    def sense(self):
        x = self.cart.position[0]
        xdot = self.cart.linearVelocity[0]
        a = self.pole.angle #because pole defined with angle zero being straight down
        # convert to range -pi to pi
        #if a > 0:
        #    a = a - 2*math.pi * math.ceil(a/(2*math.pi))
        #a = math.fmod(a-math.pi, 2*math.pi) + math.pi
        #a = math.fmod(a,2*math.pi)
        adot = self.pole.angularVelocity
        return x, a, xdot, adot

    def setState(self,s):
        x,a,xdot,adot = s
        self.cart.position[0] = x[0]
        self.cart.linearVelocity[0] = xdot[0]
        self.pole.angle = a[0]
        self.pole.angularVelocity = adot[0]
        print "angle", a[0]

    def act(self,action):
        """CartPole.act(action): action is -1, 0 or 1"""
        self.action = action
        f = (self.force*action, 0)
        p = self.cart.GetWorldPoint(localPoint=(0.0, self.cartHeight/2))
        self.cart.ApplyForce(f, p, True)
        timeStep = 1.0/self.framesPerSecond
        self.world.Step(timeStep, self.velocityIterations, self.positionIterations)
        self.world.ClearForces()
        
    def actforce(self,force,dt=None):
        """CartPole.act(action): action is -1, 0 or 1"""
        f = (force[0][0], 0)
        p = self.cart.GetWorldPoint(localPoint=(0.0, self.cartHeight/2))
        self.cart.ApplyForce(f, p, True)
        timeStep = 1.0/self.framesPerSecond
        #if dt:
        #    timeStep = dt
        self.world.Step(timeStep, self.velocityIterations, self.positionIterations)
        self.world.ClearForces()

    def dynamics(self,x,u):
        #self.actforce(u)
        mc = self.cartMass
        mp = self.poleMass
        l = self.poleLength
        g= 9.81
        I = 0.25;
        s = np.sin(x[1,:])
        c = np.cos(x[1,:])

        xddot = (u + mp*s*(l*x[3,:]**2 + g*c))/(mc+mp*s**2)
        tddot = (-u*c - mp*l*x[3,:]**2*c*s - (mc+mp)*g*s)/(l*(mc+mp*s**2))
        xdot = np.vstack([x[2:4,:],xddot,tddot])
        print "xdot",xdot

        return xdot

    def initDisplay(self):

        self.screen = pygame.display.set_mode(self.screenSize, 0, 32)
        pygame.display.set_caption('Cart')
        self.clock = pygame.time.Clock()
        
    def draw(self):
        # Clear screen
        self.screen.fill((250,250,250))
        # Draw circle for joint. Do before cart, so will appear as half circle.
        jointCoord = self.w2p(self.cart.GetWorldPoint((0,self.cartHeight/2)))
        junk,radius = self.dw2dp((0,2*self.poleThickness))
        pygame.draw.circle(self.screen, self.cart.userData['color'], jointCoord, radius, 0)
        # Draw other bodies
        for body in (self.track,self.wallLeft,self.wallRight,self.cart,self.pole): # or world.bodies
            for fixture in body.fixtures:
                shape = fixture.shape
                # Assume polygon shapes!!!
                vertices = [self.w2p((body.transform * v)) for v in shape.vertices]
                pygame.draw.polygon(self.screen, body.userData['color'], vertices)

        #print self.cart.position,self.cart.linearVelocity,self.pole.angle,self.pole.angularVelocity
        # Draw arrow showing force    
        # if self.action != 0:
        #     cartCenter = self.w2p(self.cart.GetWorldPoint((0,0)))
        #     arrowEnd = (cartCenter[0]+self.action*20, cartCenter[1])
        #     pygame.draw.line(self.screen, (250,250,0), cartCenter, arrowEnd, 3)
        #     pygame.draw.line(self.screen, (250,250,0), arrowEnd,
        #                      (arrowEnd[0]-self.action*5, arrowEnd[1]+5), 3)
        #     pygame.draw.line(self.screen, (250,250,0), arrowEnd,
        #                      (arrowEnd[0]-self.action*5, arrowEnd[1]-5), 3)

        pygame.display.flip()
        self.clock.tick(self.framesPerSecond)

    # Now print the position and angle of the body.

    def w2p(self,(x,y)):
        """ Convert world coordinates to screen (pixel) coordinates"""
        return (int(0.5+(x+self.worldSize[0]/2) / self.worldSize[0] * self.screenSize[0]),
                int(0.5+self.screenSize[1] - (y+self.worldSize[1]/2) / self.worldSize[1] * self.screenSize[1]))
    def p2w(self,(x,y)):
        """ Convert screen (pixel) coordinates to world coordinates"""
        return (x / self.screenSize[0] * self.worldSize[0] - self.worldSize[0]/2,
                (self.screenSize[1]-y) / self.screenSize[1] * self.worldSize[1] - self.worldSize[1]/2)
    def dw2dp(self,(dx,dy)):
        """ Convert delta world coordinates to delta screen (pixel) coordinates"""
        return (int(0.5+dx/self.worldSize[0] * self.screenSize[0]),
                int(0.5+dy/self.worldSize[1] * self.screenSize[1]))


def sim_cartpole(cpole,x0, u, dt):
    DT =0.1; t=0;
    x1 = x0.copy()
    while t < dt:
        current_dt = min(DT, dt-t);
        x1 = x1 + current_dt*cpole.dynamics(x1,u);
        t = t+current_dt;
    
        # cpole.setState(x0)
        # print "Before: ",cpole.sense()
        # cpole.actforce(u)
        # [x,a,xdot,adot] = cpole.sense()
        # print "After: ",[x,a,xdot,adot]
        # x1 = np.asarray([x,a,xdot,adot],np.float64)
        # x1.shape = (4,1)

    #x0 = x0 + current_dt*x1
    #print x1
    return x1

def lqr_infinite_horizon_solution(A, B, Q, R):

    inv = np.linalg.inv
    norm = np.linalg.norm

    
    Pprev = np.zeros(np.shape(Q));
    K_current = -inv(R+B.T.dot(Pprev).dot(B)).dot(B.T).dot(Pprev).dot(A)
    
    Pprev = Q+K_current.T.dot(R).dot(K_current) + (A+B.dot(K_current)).T.dot(Pprev).dot(A+B.dot(K_current))

    K_new = -inv(R+B.T.dot(Pprev).dot(B)).dot(B.T).dot(Pprev).dot(A);
    Pprev = Q+K_new.T.dot(R).dot(K_new) + (A+B.dot(K_new)).T.dot(Pprev).dot(A+B.dot(K_new));
    val = norm(K_new-K_current,2);
    t=0;
    while norm(K_new-K_current,2)>1e-4:
        t = t+1;
        K_current = K_new;
        K_new = -inv(R+B.T.dot(Pprev).dot(B)).dot(B.T).dot(Pprev).dot(A);
        Pprev = Q+K_new.T.dot(R).dot(K_new) + (A+B.dot(K_new)).T.dot(Pprev).dot(A+B.dot(K_new));
    K = K_new
    P = Pprev
    return K,P
    
def linearize_dynamics(cpole,x_ref, u_ref, dt, my_eps):

    f = sim_cartpole

    A = np.zeros([np.size(x_ref),np.size(x_ref)])

    B = np.zeros([np.size(x_ref),np.size(u_ref)])

    print B.shape
    
    n = np.size(x_ref);

    fx = f(cpole,x_ref,u_ref,dt)

    for i in range(n):
        perturb1 = x_ref.copy()
        perturb1[i] += my_eps
        perturb2 = x_ref.copy()
        perturb2[i] -= my_eps
        fxplush = f(cpole,perturb1,u_ref,dt)
        fxminush = f(cpole,perturb2,u_ref,dt)
        #A[:,i] = ((fxplush-fxminush)/(2*my_eps))[:,0]
        A[:,i] = ((fxplush-fx)/(my_eps))[:,0]


    n = np.size(u_ref)
    for i in range(n):
        perturb1 = np.asarray(u_ref.copy(),dtype=np.float64)
        perturb1[0][0] += my_eps
        perturb2 = np.asarray(u_ref.copy(),dtype=np.float64)
        perturb2[0][0] -= my_eps
        print perturb1
        print perturb2
        #B = ((f(cpole,x_ref,perturb1,dt)-f(cpole,x_ref,perturb2,dt))/(2*my_eps))
        B = ((f(cpole,x_ref,perturb1,dt)-fx)/(my_eps))

    c = fx-x_ref;

    return A,B,c


if __name__ == "__main__":

    cartpole = CartPole()
    cartpole.initDisplay()
    cartpole.action=0;

    dt = 0.01
    running = True
    reps = 0

    x_ref = np.asarray([0,math.pi,0,0],np.float64)
    x_ref.shape = (4,1)
    u_ref = np.asarray([0],np.float64)
    u_ref.shape = (1,1)
    my_eps = 0.1

    [A, B, c] = linearize_dynamics(cartpole,x_ref.copy(), u_ref, dt, my_eps)

    print "Linearized Dynamics"
    print A
    print B
    print c

    Q = np.eye(4)
    R = np.eye(1)
    

    [K_inf, P_inf] = lqr_infinite_horizon_solution(A, B, Q, R)
    x_ref = np.asarray([0,np.pi,0,0],np.float64)
    x_ref.shape = (4,1)    

    x = np.asarray([0,np.pi-np.pi/4,0,0],np.float64)
    x.shape = (4,1)    

    # cartpole.setState(x)
    # cartpole.draw()
    # print cartpole.sense()
    # print x_ref
    # print u_ref
    raw_input("reset state")


    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-20, 20), ylim=(-2, 2))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text


    def animate(i):
        thisx = [x1[i], x2[i]]
        thisy = [y1[i], y2[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        return line, time_text


    #ani.save('double_pendulum.mp4', fps=15)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    while running:
        reps += 1

        # # Set action to -1, 1, or 0 by pressing lef or right arrow or nothing.
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT: 
        #         running = False
        #     elif event.type == KEYDOWN:
        #         if event.key == K_RIGHT:
        #             action = 1
        #         elif event.key == K_LEFT:
        #             action = -1
        #     elif event.type == KEYUP:
        #         if event.key == K_RIGHT or event.key == K_LEFT:
        #             action = 0

        # # Apply action to cartpole simulation
        # cartpole.act(action)
        # # Redraw cartpole in new state
        # cartpole.draw()
        #x = np.asarray(cartpole.sense(),np.float64)
        #x.shape = (4,1)
        u =  (K_inf.dot(x - x_ref)) + u_ref;
        print "U:",u
        x = sim_cartpole(cartpole,x,u,dt)

        print x

        # find the end point
        endy = 0.5 * np.sin(x[1,:]-np.pi/2)
        endx = 0.5 * np.cos(x[1,:]-np.pi/2)

        # plot the points
        #ax.plot([x[0,:], x[0,:]+endx], [0, endy])
        x1.append(x[0,:])
        x2.append(x[0,:]+endx)
        y1.append(0)
        y2.append(0.5)

        #fig.show()
        #cartpole.actforce(u)
        #cartpole.setState(x)
        #cartpole.draw()
        print "Angle ",x[1,:]
        print "Points",endx,endy
#        plt.pause(1/60.)
#        plt.cla()
#    pygame.quit()
        if reps==1000:
            break

    import matplotlib.animation as animation

    ani = animation.FuncAnimation(fig, animate, np.arange(1,reps),
                                  interval=10, blit=True, init_func=init)

    plt.show()