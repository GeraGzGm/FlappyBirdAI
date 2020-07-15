import pygame
import random
import numpy as np
import time

pygame.font.init()

clock = pygame.time.Clock()

def salir():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True

Birds_img = [pygame.transform.scale2x(pygame.image.load("imgs/bird1.png")),pygame.transform.scale2x(pygame.image.load("imgs/bird2.png")),pygame.transform.scale2x(pygame.image.load("imgs/bird3.png"))]
Birdsr_img = [pygame.transform.scale2x(pygame.image.load("imgs/bird1r.png")),pygame.transform.scale2x(pygame.image.load("imgs/bird2r.png")),pygame.transform.scale2x(pygame.image.load("imgs/bird3r.png"))]

Pipe_img = pygame.transform.scale2x(pygame.image.load("imgs/pipe.png"))
Floor_img = pygame.transform.scale2x(pygame.image.load("imgs/base.png"))
BG_img = pygame.transform.scale2x(pygame.image.load("imgs/bg.png"))

STAT_FONT = pygame.font.SysFont("comicsans",50)
SCORE = 0
GEN = 0

class Bird():
    IMGS = Birds_img
    Max_rotation = 25
    Rot_vel = 20
    Anim_time = 5
    NEURONS = 5
    NEURONS2 = 3
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.im_count = 0
        self.img = self.IMGS[0]
        self.fit = 0
        self.alive = True
        
        self.wh = -1 + (1+1) * np.random.random((self.NEURONS,4))
        self.bh = -1 + (1+1) * np.random.random((self.NEURONS,1))
        
        self.wh2 = -1 + (1+1) * np.random.random((self.NEURONS2,self.NEURONS))
        self.bh2 = -1 + (1+1) * np.random.random((self.NEURONS2,1))
        
        self.wo = -1 + (1+1) * np.random.random((1,self.NEURONS2))
        self.bo = -1 + (1+1) * np.random.random((1,1))
        
    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y
    
    def move(self):
        self.tick_count += 1
        
        d = self.vel*self.tick_count + 1.5*self.tick_count**2
        
        if d >= 16: 
            d = 16
        elif d < 0:
            d -= 2
        
        self.y += d
        
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.Max_rotation:
                self.tilt = self.Max_rotation
        else:
            if self.tilt > -90:
                self.tilt -= self.Rot_vel
    
    def draw(self,win):
        self.im_count += 1
        
        if self.im_count < self.Anim_time:
            self.img = self.IMGS[0]
        elif self.im_count < self.Anim_time*2:
            self.img = self.IMGS[1]
        elif self.im_count < self.Anim_time*3:
            self.img = self.IMGS[2]
        elif self.im_count < self.Anim_time*4:
            self.img = self.IMGS[1]
        elif self.im_count < self.Anim_time*4 +1:
            self.img = self.IMGS[0]
            self.im_count = 0
        
        if self.tilt <= -80:
            self.img = self.IMGS[0]
            self.im_count = self.Anim_time*2
        
        rotated_image = pygame.transform.rotate(self.img,self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x,self.y)).center)
        
        win.blit(rotated_image,new_rect.topleft)
    
    def get_mask(self):
        return pygame.mask.from_surface(self.img)
    
    def Brain(self,pipes):
        inp = np.array([[self.y],[self.y - abs(pipes.top)],[self.y - pipes.bottom],[self.vel]])
        yh = sigmoid(np.dot(self.wh,inp) + self.bh)
        yh2 = sigmoid(np.dot(self.wh2,yh) + self.bh2)

        
        self.yo = sigmoid(np.dot(self.wo,yh2) + self.bo)
        
        
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
            
class Pipe():
    GAP = 200
    VEL = 5
    
    def __init__(self,x):
        self.x = x
        self.height = 0
        
        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(Pipe_img,False,True)
        self.PIPE_BOTTOM = Pipe_img
        
        self.passed = False
        self.set_height()
        
    def set_height(self):
        self.height = random.randrange(50,450)
        self.top  = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
        
    def move(self):
        self.x -= self.VEL
        
    def draw(self,win):
        win.blit(self.PIPE_TOP,(self.x,self.top))
        win.blit(self.PIPE_BOTTOM,(self.x,self.bottom))
    
    def collide(self,bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        
        top_offset = (self.x - bird.x,self.top - round(bird.y))
        bottom_offset = (self.x - bird.x,self.bottom - round(bird.y))
        
        b_point = bird_mask.overlap(bottom_mask,bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)
        
        if t_point or b_point:
            return True
        
        return False
    
class Ground():
    VEL = 5
    WIDTH = Floor_img.get_width()
    IMG = Floor_img
    
    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
    
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        elif self.x2 + self.WIDTH < 0:
            self.x1 = self.x1 + self.WIDTH

    def draw(self,win):
        win.blit(self.IMG,(self.x1,self.y))
        win.blit(self.IMG,(self.x2,self.y))
    
def draw_wind(win,birds,pipes,base,SCORE,GEN,DrawLines = False):
    win.blit(BG_img,(0,0))
    
    text = STAT_FONT.render("Score: " + str(SCORE),1,(255,255,255))
    textgen = STAT_FONT.render("Gen: " + str(GEN),1,(255,255,255))
    
    for pipe in pipes:
        pipe.draw(win)
    
    # base.draw(win)
    
    for bird in birds:
        if bird.alive:
            bird.draw(win)
            if DrawLines:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[-1].x + pipes[-1].PIPE_TOP.get_width()/2, pipes[-1].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[-1].x + pipes[-1].PIPE_BOTTOM.get_width()/2, pipes[-1].bottom), 5)
        
    
    win.blit(text,(490 - text.get_width(),10))
    win.blit(textgen,(10,10))
    pygame.display.update()
    
def move(birds,pipes,base,SCORE):
    add_pipe = False
    
    for bird in birds:
        if bird.alive:
            bird.move()
            bird.fit += 0.1
          
        if pipes[0].x + pipes[0].PIPE_BOTTOM.get_width() > bird.x:
            bird.Brain(pipes[0])
        else:
            bird.Brain(pipes[-1])

    
    for pipe in pipes:
        for bird in birds:
            if bird.yo > 0.5:
                bird.jump()
                
            if pipe.collide(bird) or bird.y + bird.img.get_height() >= 730 or bird.y < 1:
                bird.fit -= 1
                bird.alive = False

            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True
        pipe.move()
        
    if add_pipe:
        SCORE += 1
        bird.fit += 1
        pipes.append(Pipe(600))
      
    if len(pipes)>2:
        pipes.pop(0)
    
    base.move()
    return SCORE

def Seleccion(birds):
    aptitud = []
    for bird in birds:
        if bird.fit >= 0:
            aptitud.append( 1/(1+bird.fit))
        else:
            aptitud.append(1 + abs(bird.fit))
    
    ap_tot = np.sum(np.array(aptitud))
    probs = [bird/ap_tot for bird in aptitud]
    
    P1,P2 = Padres(birds,probs),Padres(birds,probs)
    while P1 == P2: P2 = Padres(birds,probs)
      
    return P2,P2
    
def Padres(birds,probs):
    r,psum = np.random.rand(),0
    for i in range(len(birds)):
        psum += probs[i]
        if psum > r:
            return i

class Hijos():
    def __init__(self,wh,wh2,wo,bh,bh2,bo):
        self.wh = wh
        self.wo = wo
        self.bh = bh
        self.bo = bo
        self.wh2 = wh2
        self.bh2 = bh2

def Cruza(birds):
    # P1,P2 = Seleccion(birds)
    
    allf = np.array([bird.fit for bird in birds])
    P1 = np.argmax(allf)
    allf[P1] = -9999
    P2 = np.argmax(allf)
    
    a,b = birds[P1].wh.shape 
    wh = np.zeros((a,b,2))
    Pc = np.random.randint(0,a*b)
    suma = 0
    for i in range(a):
        for j in range(b):
            if suma < Pc:
                wh[i,j,0] = birds[P1].wh[i,j]
                wh[i,j,1] = birds[P2].wh[i,j]
                suma+=1
            else:
                wh[i,j,0] = birds[P2].wh[i,j]
                wh[i,j,1] = birds[P1].wh[i,j]
    
    a,b = birds[P1].wh2.shape 
    wh2 = np.zeros((a,b,2))
    Pc = np.random.randint(0,a*b)
    suma = 0
    for i in range(a):
        for j in range(b):
            if suma < Pc:
                wh2[i,j,0] = birds[P1].wh2[i,j]
                wh2[i,j,1] = birds[P2].wh2[i,j]
                suma+=1
            else:
                wh2[i,j,0] = birds[P2].wh2[i,j]
                wh2[i,j,1] = birds[P1].wh2[i,j]
         
    a,b = birds[P1].bh.shape 
    bh = np.zeros((a,b,2))
    Pc = np.random.randint(0,a*b)
    suma = 0
    for i in range(a):
        for j in range(b):
            if suma < Pc:
                bh[i,j,0] = birds[P1].bh[i,j]
                bh[i,j,1] = birds[P2].bh[i,j]
                suma+=1
            else:
                bh[i,j,0] = birds[P2].bh[i,j]
                bh[i,j,1] = birds[P1].bh[i,j]
    
    a,b = birds[P1].bh2.shape 
    bh2 = np.zeros((a,b,2))
    Pc = np.random.randint(0,a*b)
    suma = 0
    for i in range(a):
        for j in range(b):
            if suma < Pc:
                bh2[i,j,0] = birds[P1].bh2[i,j]
                bh2[i,j,1] = birds[P2].bh2[i,j]
                suma+=1
            else:
                bh2[i,j,0] = birds[P2].bh2[i,j]
                bh2[i,j,1] = birds[P1].bh2[i,j]
    
    a,b = birds[P1].wo.shape 
    wo = np.zeros((a,b,2))
    Pc = np.random.randint(0,a*b)
    suma = 0
    for i in range(a):
        for j in range(b):
            if suma < Pc:
                wo[i,j,0] = birds[P1].wo[i,j]
                wo[i,j,1] = birds[P2].wo[i,j]
                suma+=1
            else:
                wo[i,j,0] = birds[P2].wo[i,j]
                wo[i,j,1] = birds[P1].wo[i,j]
    
    a,b = birds[P1].bo.shape 
    bo = np.zeros((a,b,2))
    Pc = np.random.randint(0,a*b)
    suma = 0
    for i in range(a):
        for j in range(b):
            if suma < Pc:
                bo[i,j,0] = birds[P1].bo[i,j]
                bo[i,j,1] = birds[P2].bo[i,j]
                suma+=1
            else:
                bo[i,j,0] = birds[P2].bo[i,j]
                bo[i,j,1] = birds[P1].bo[i,j]
    
    return wh,wh2,wo,bh,bh2,bo
            
def Mutacion(Pm,birds,hijos):
    
    for h in range(len(hijos)-2):
        a,b = birds[0].wh.shape 
        birds[h].wh = hijos[h].wh
        for i in range(a):
            for j in range(b):
                ra =np.random.rand()
                if ra < Pm:
                    hijos[h].wh[i,j] = -1 + (1+1) * np.random.rand()
                
        a,b = birds[0].wh2.shape 
        birds[h].wh2= hijos[h].wh2
        for i in range(a):
            for j in range(b):
                ra =np.random.rand()
                if ra < Pm:
                    hijos[h].wh2[i,j] = -1 + (1+1) * np.random.rand()
        
        a,b = birds[0].wo.shape
        birds[h].wo =  hijos[h].wo
        for i in range(a):
            for j in range(b):
                ra =np.random.rand()
                if ra < Pm:
                    hijos[h].wo[i,j] = -1 + (1+1) * np.random.rand()
                
        
        a,b = birds[0].bh.shape 
        birds[h].bh =  hijos[h].bh
        for i in range(a):
            for j in range(b):
                ra =np.random.rand()
                if ra < Pm:
                    hijos[h].bh[i,j] = -1 + (1+1) * np.random.rand()
                
        a,b = birds[0].bh2.shape 
        birds[h].bh2 =  hijos[h].bh2
        for i in range(a):
            for j in range(b):
                ra =np.random.rand()
                if ra < Pm:
                    hijos[h].bh2[i,j] = -1 + (1+1) * np.random.rand()
        
        a,b = birds[0].bo.shape 
        birds[h].bo =  hijos[h].bo
        for i in range(a):
            for j in range(b):
                ra =np.random.rand()
                if ra < Pm:
                    hijos[h].bo[i,j] = -1 + (1+1) * np.random.rand()
                
#####################################################        
N = 20
Pm = 0.001
load = True
Max_Gen = 100
Max_Score = 20
#####################################################  

birds = [Bird(200,250) for i in range(N)]

wh1,wh2,wo = np.load("pesos/wh1.npy"),np.load("pesos/wh2.npy"),np.load("pesos/wo.npy")
bh1,bh2,bo = np.load("pesos/bh1.npy"),np.load("pesos/bh2.npy"),np.load("pesos/bo.npy")
fitness = np.array(np.load("pesos/fitness.npy"))

if load:
    for i in range(N):
        if i < wh1.shape[0]:
            best = np.argmax(fitness)
            birds[i].wh1 = wh1[i].reshape((wh1[0].shape[1],wh1[0].shape[2]))
            birds[i].wh2 = wh2[i].reshape((wh2[0].shape[1],wh2[0].shape[2]))
            birds[i].wo = wo[i].reshape((wo[0].shape[1],wo[0].shape[2]))
            birds[i].bh1 = bh1[i].reshape((bh1[0].shape[1],bh1[0].shape[2]))
            birds[i].bh2 = bh2[i].reshape((bh2[0].shape[1],bh2[0].shape[2]))
            birds[i].bo = bo[i].reshape((bo[0].shape[1],bo[0].shape[2]))
            fitness[best] = -9999

base = Ground(730) 
pipes = [Pipe(400)]

win = pygame.display.set_mode((500,800))
moridos = 0

moving = True



while not salir():
    while moving:
        draw_wind(win,birds,pipes,base,SCORE,GEN)
        SCORE = move(birds,pipes,base,SCORE)
        
        for bird in birds:
            if bird.alive == False: 
                moridos += 1
            
        if moridos == N: 
            moving = False 
        moridos = 0
        
        if SCORE >= Max_Score:
            moving = False 
    
    if GEN >= Max_Gen or SCORE >= Max_Score:    break
    
    for bird in birds: 
        bird.x,bird.y = 200,250
        bird.alive = True
            
    hijos = []
    while len(hijos) <= N:
          wh,wh2,wo,bh,bh2,bo = Cruza(birds)
          for i in range(2):
              hijos.append(Hijos(wh[:,:,i],wh2[:,:,i],wo[:,:,i],bh[:,:,i],bh2[:,:,i],bo[:,:,i]))
            
    Mutacion(Pm,birds,hijos)

    pipes = [Pipe(400)]
    SCORE = 0
    moving = True
    GEN+=1
    pygame.time.wait(300)
    
    
wh1,wh2,wo,bh1,bh2,bo,fitness = [],[],[],[],[],[],[]
for i in range(N):
    wh1.append([birds[i].wh])
    wh2.append([birds[i].wh2])
    wo.append([birds[i].wo])
    bh1.append([birds[i].bh])
    bh2.append([birds[i].bh2])
    bo.append([birds[i].bo])
    fitness.append([birds[i].fit])
    

np.save("pesos/wh1",wh1),np.save("pesos/wh2",wh2),np.save("pesos/wo",wo)
np.save("pesos/bh1",bh1),np.save("pesos/bh2",bh2),np.save("pesos/bo",bo)
np.save("pesos/fitness",fitness)
pygame.quit()
# quit()