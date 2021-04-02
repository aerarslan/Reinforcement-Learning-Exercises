"""
RL GAME
"""
# pygame template
import pygame
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# window size
WIDTH = 180
HEIGHT = 180
FPS = 30 # how fast game is

# colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 0) # RGB
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (128, 0, 128)

class Player(pygame.sprite.Sprite):
    # sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
#        self.radius = 10
#        pygame.draw.circle(self.image, RED, self.rect.center,self.radius)
        self.rect.centerx = WIDTH
        self.rect.bottom = HEIGHT
        self.speedx = 0
        self.speedy = 0
        
    def update(self, action):
        self.speedx = 0
        self.speedy = 0
        keystate = pygame.key.get_pressed()
        
        if keystate[pygame.K_LEFT] or action == 0:
            self.speedx = -4
        elif keystate[pygame.K_RIGHT] or action == 1:
            self.speedx = 4
        elif keystate[pygame.K_UP] or action == 2:
            self.speedy = -4
        else:
            self.speedy = 4
         
        self.rect.x +=self.speedx
        self.rect.y +=self.speedy
        
        if self.rect.right > WIDTH:
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
            
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT
        if self.rect.top < 0:
            self.rect.top = 0
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)   
    
class Enemy(pygame.sprite.Sprite):
    
    def __init__(self, width, height, corx, cory):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((width,height))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
#        self.radius = 5
#        pygame.draw.circle(self.image, WHITE, self.rect.center,self.radius)
        self.rect.x = corx
        self.rect.y = cory
        
#        self.speedx = 0
#        self.speedy = 3
    
#    def update(self):
#        
#        self.rect.x += self.speedx
#        self.rect.y += self.speedy
#        
#        if self.rect.top > HEIGHT + 10:
#            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
#            self.rect.y = random.randrange(2,6)
#            self.speedy = 3
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)  

class FinalDest(pygame.sprite.Sprite):
    
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20))
        self.image.fill(PURPLE)
        self.rect = self.image.get_rect()
#        self.radius = 5
#        pygame.draw.circle(self.image, WHITE, self.rect.center,self.radius)
        self.rect.x = 160
        self.rect.y = 0
        
#        self.speedx = 0
#        self.speedy = 3
    
#    def update(self):
#        
#        self.rect.x += self.speedx
#        self.rect.y += self.speedy
#        
#        if self.rect.top > HEIGHT + 10:
#            self.rect.x = random.randrange(0, WIDTH - self.rect.width)
#            self.rect.y = random.randrange(2,6)
#            self.speedy = 3
            
    def getCoordinates(self):
        return (self.rect.x, self.rect.y)  
    
class DQLAgent:
    def __init__(self):
        # parameter / hyperparameter
        self.state_size = 8 # distance [(playerx-m1x),(playery-m1y),(playerx-m2x),(playery-m2y)]
        self.action_size = 4 # right, left, up, down
        
        self.gamma = 0.95
        self.learning_rate = 0.001 
        
        self.epsilon = 1  # explore
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model = self.build_model()
        
        
    def build_model(self):
        # neural network for deep q learning
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "relu"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory,batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
            if done:
                target = reward 
            else:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target, verbose = 0)
            
    def adaptiveEGreedy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
 
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.finalDest = pygame.sprite.Group() # ******
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy(40, 5, 110, 40)
        self.m2 = Enemy(5, 40, 50, 100)
        self.m3 = Enemy(40,5,130,130)
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.dest1 = FinalDest() # **********
        self.all_sprite.add(self.dest1) # *********
        self.finalDest.add(self.dest1) # **********
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
        self.agent = DQLAgent()
        
    def findDistance(self, a, b):
        d = a-b
        return d
    
    def step(self, action):
        state_list = []
        
        # update
        self.player.update(action)
#        self.enemy.update()
        
        # get coordinate
        next_player_state = self.player.getCoordinates()
        next_m1_state = self.m1.getCoordinates()
        next_m2_state = self.m2.getCoordinates()
        next_m3_state = self.m3.getCoordinates()
        next_dest1_state = self.dest1.getCoordinates() # *****
        
        # find distance
        state_list.append(self.findDistance(next_player_state[0],next_m1_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m1_state[1]))
        state_list.append(self.findDistance(next_player_state[0],next_m2_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m2_state[1]))
        state_list.append(self.findDistance(next_player_state[0],next_dest1_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_dest1_state[1]))
        state_list.append(self.findDistance(next_player_state[0],next_m3_state[0]))
        state_list.append(self.findDistance(next_player_state[1],next_m3_state[1]))        
        
        return [state_list]
         
    # reset
    def initialStates(self):
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.finalDest = pygame.sprite.Group() # ******
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy(40, 5, 100, 40)
        self.m2 = Enemy(5, 40, 50, 100)
        self.m3 = Enemy(40,5,130,130)        
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.all_sprite.add(self.m3)        
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.enemy.add(self.m3)
        self.dest1 = FinalDest() # **********
        self.all_sprite.add(self.dest1) # *********
        self.finalDest.add(self.dest1) # **********
        
        self.reward = 0
        self.total_reward = 0
        self.done = False
    
        state_list = []
        
        # get coordinate
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()
        m2_state = self.m2.getCoordinates()
        m3_state = self.m3.getCoordinates()
        dest1_state = self.dest1.getCoordinates()
        
        # find distance
        state_list.append(self.findDistance(player_state[0],m1_state[0]))
        state_list.append(self.findDistance(player_state[1],m1_state[1]))
        state_list.append(self.findDistance(player_state[0],m2_state[0]))
        state_list.append(self.findDistance(player_state[1],m2_state[1]))
        state_list.append(self.findDistance(player_state[0],dest1_state[0]))
        state_list.append(self.findDistance(player_state[1],dest1_state[1]))
        state_list.append(self.findDistance(player_state[0],m3_state[0]))
        state_list.append(self.findDistance(player_state[1],m3_state[1]))        
        return [state_list]
        
    def run(self):
        # game loop
        state = self.initialStates()
        running = True
        batch_size = 24
        while running:
            self.reward = -0.1
            # keep loop running at the right speed
            clock.tick(FPS) 
            # process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False   
            # update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward 
                
            hits = pygame.sprite.spritecollide(self.player,self.enemy,False, pygame.sprite.collide_rect)   
            if hits:
                self.reward = -100
                self.total_reward += self.reward 
                self.done = True
                running = False
                print("Total reward: ",self.total_reward) 
            
#            arrives = pygame.sprite.spritecollide(self.player,self.finalDest,False, pygame.sprite.collide_rect)
            if self.dest1.getCoordinates() == self.player.getCoordinates():
                self.reward = 500
                self.total_reward += self.reward 
                self.done = True
                running = False
                print("Total reward: ",self.total_reward) 
                
                print("Yes Baby!")                
#            if arrives:
#                self.reward = 500
#                self.total_reward += self.reward 
#                self.done = True
#                running = False
#                print("Total reward: ",self.total_reward) 
#                
#                print("Yes Baby!")
                
            # storage
            self.agent.remember(state, action,self.reward, next_state, self.done)
            
            # update state
            state = next_state
            
            # training
            self.agent.replay(batch_size)
            
            # epsilon greedy
            self.agent.adaptiveEGreedy()
            
            # draw / render(show)
            screen.fill(GREEN)
            self.all_sprite.draw(screen)
            # after drawing flip display
            pygame.display.flip()
    
        pygame.quit()  

if __name__ == "__main__":
    env = Env()
    liste = []
    t = 0
    while True:
        t += 1
        print("Episode: ",t)
        liste.append(env.total_reward)
                
        # initialize pygame and create window
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()
        
        env.run()
  
