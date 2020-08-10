import random
import pygame
import numpy as np
import math

class EnhancedSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, velocityX, velocityY):
        super().__init__()
        self.width = width
        self.height = height
        self.x = 0.0
        self.y = 0.0
        self.velocityX = velocityX
        self.velocityY = velocityY

    def setPosition(self, position):
        self.x = position[0]
        self.y = position[1]
        self.rect.x = round(self.x)
        self.rect.y = round(self.y)

    def setX(self, x):
        self.x = float(x)
        self.rect.x = round(x)

    def setY(self, y):
        self.y = float(y)
        self.rect.y = round(y)

    def confineVertical(self, minY, maxY, flipVel=False):
        if self.rect.top < minY:
            self.setY(minY)
            if flipVel:
                self.velocityY *= -1.0
        elif self.rect.bottom > maxY:
            self.setY(maxY - self.height)
            if flipVel:
                self.velocityY *= -1.0

    def deltaX(self, xDelta):
        self.x += xDelta
        self.rect.x = round(self.x)

    def deltaY(self, yDelta):
        self.y += yDelta
        self.rect.y = round(self.y)

    def applyVelocities(self):
        self.deltaX(self.velocityX)
        self.deltaY(self.velocityY)

class Player(EnhancedSprite):
    def __init__(self, color, width, height, speed, isLeftPaddle, EDGE_OFFSET, CANVAS_WIDTH, CANVAS_HEIGHT, velocityX=0, velocityY=0):
        super().__init__(width, height, velocityX, velocityY)
        # Create image to show sprite
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.EDGE_OFFSET = EDGE_OFFSET
        self.isLeftPaddle = isLeftPaddle

        # Set initial position
        if isLeftPaddle:
            self.setPosition((EDGE_OFFSET, (CANVAS_HEIGHT-self.height)//2))
            # self.rect.left = EDGE_OFFSET
            # self.rect.centery = CANVAS_HEIGHT // 2
        else:
            self.setPosition((CANVAS_WIDTH - EDGE_OFFSET - self.width, (CANVAS_HEIGHT-self.height)//2))
            # self.rect.right = CANVAS_WIDTH - EDGE_OFFSET
            # self.rect.centery = CANVAS_HEIGHT // 2

        self.speed = speed
        self.score = 0
        self.timestep_reward = 0

    def performAction(self, action):
        '''
        action: Action to perform
        -------------------------
        0: Don't move paddle
        1: Paddle up
        2: Paddle down
        '''
        if action == 0: # Do nothing
            self.velocityY = 0
        elif action == 1: # Move up
            self.velocityY = -self.speed
        elif action == 2: # Move down
            self.velocityY = self.speed
        else:
            print("ERROR: Action invalid")
            assert(False)
    
    def reset(self, CANVAS_WIDTH, CANVAS_HEIGHT):
        # Set reset position
        if self.isLeftPaddle:
            self.setPosition((self.EDGE_OFFSET, (CANVAS_HEIGHT-self.height)//2))
            # self.rect.left = EDGE_OFFSET
            # self.rect.centery = CANVAS_HEIGHT // 2
        else:
            self.setPosition((CANVAS_WIDTH - self.EDGE_OFFSET - self.width, (CANVAS_HEIGHT-self.height)//2))
            # self.rect.right = CANVAS_WIDTH - EDGE_OFFSET
            # self.rect.centery = CANVAS_HEIGHT // 2
        
        self.velocityY = 0

class Ball(EnhancedSprite):
    def __init__(self, color, width, height, MAX_INITIAL_VEL, CANVAS_WIDTH, CANVAS_HEIGHT, velocityX=0, velocityY=0):
        super().__init__(width, height, velocityX, velocityY)
        # Check inputs
        assert(MAX_INITIAL_VEL != 0)
        self.MAX_INITIAL_VEL = MAX_INITIAL_VEL

        # Create image to show sprite
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        self.rect = self.image.get_rect()
    
    def reset(self, CANVAS_WIDTH, CANVAS_HEIGHT):
        self.setPosition((CANVAS_WIDTH//2, CANVAS_HEIGHT//2))

        self.velocityX = random.uniform(-self.MAX_INITIAL_VEL*1.25, self.MAX_INITIAL_VEL*1.25)
        self.velocityY = random.uniform(-self.MAX_INITIAL_VEL*0.75, self.MAX_INITIAL_VEL*0.75)

        # Make sure ball doesn't get stuck bouncing up and down
        while abs(self.velocityX) < 0.1:
            self.velocityX = random.uniform(-self.MAX_INITIAL_VEL, self.MAX_INITIAL_VEL)

class Game:

    def __init__(self, CANVAS_WIDTH, CANVAS_HEIGHT, framerate):
        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.CANVAS_HEIGHT = CANVAS_HEIGHT
        self.framerate = framerate
        self.clock = pygame.time.Clock()
        self.EDGE_OFFSET = 10
        self.TOP_WALL_Y = 10
        self.BOT_WALL_Y = self.CANVAS_HEIGHT - 10
        self.BG_COLOR = (0,0,0)
        self.canvas = pygame.display.set_mode([self.CANVAS_WIDTH, self.CANVAS_HEIGHT])

        self.allSprites = pygame.sprite.Group()
        self.playerSprites = pygame.sprite.Group()
        self.ball = pygame.sprite.Group()

        # Create both players
        self.LeftPlayer = Player(color=(255,80,80), width=10, height=50, speed=1, isLeftPaddle=True, EDGE_OFFSET=self.EDGE_OFFSET, CANVAS_WIDTH=self.CANVAS_WIDTH, CANVAS_HEIGHT=self.CANVAS_HEIGHT)
        self.RightPlayer = Player(color=(80,80,255), width=10, height=50, speed=1, isLeftPaddle=False, EDGE_OFFSET=self.EDGE_OFFSET, CANVAS_WIDTH=self.CANVAS_WIDTH, CANVAS_HEIGHT=self.CANVAS_HEIGHT)
        self.allSprites.add(self.LeftPlayer)
        self.allSprites.add(self.RightPlayer)
        self.playerSprites.add(self.LeftPlayer)
        self.playerSprites.add(self.RightPlayer)
        # Create ball
        self.Ball = Ball(color=(255,255,255), width=7, height=7, MAX_INITIAL_VEL=1, CANVAS_WIDTH=CANVAS_WIDTH, CANVAS_HEIGHT=CANVAS_HEIGHT)
        self.allSprites.add(self.Ball)

    def reset(self):
        # Call reset method of each player and ball
        for sprite in self.allSprites:
            sprite.reset(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)
    
    def step(self, leftAction, rightAction):
        # Clear timestep rewards
        self.LeftPlayer.timestep_reward = 0
        self.RightPlayer.timestep_reward = 0

        # Perform Actions
        self.LeftPlayer.performAction(leftAction)
        self.RightPlayer.performAction(rightAction)
        
        # Handle collisions
        playersHit = pygame.sprite.spritecollide(self.Ball, self.playerSprites, dokill=False)

        for player in playersHit:
            if len(playersHit) > 1:
                print("Error: Ball hit both players at same time")
                assert(False)
            if self.Ball.velocityX > 10:
                print("--------------------------")
            if player.isLeftPaddle:
                # Ball bounces off left paddle
                self.Ball.setX(player.x + player.width)
                self.Ball.velocityX *= -1.0 if abs(self.Ball.velocityX) > 0.2 else -2.0
            else:
                # Ball bounces off right paddle
                self.Ball.setX(player.x - self.Ball.width)
                self.Ball.velocityX *= -1.0 if abs(self.Ball.velocityX) > 0.2 else -2.0

            # Figure out where on paddle ball hit
            ball_center_y = self.Ball.y + self.Ball.height/2
            paddle_center_y = player.y + player.height/2

            # If ball hits top of paddle, go down (+vel), if ball hits bottom of paddle go up (-vel)
            # ball_vel_y_modifier has abs val <= 1 + small value (assuming ball height << paddle height)
            ball_vel_y_modifier = (paddle_center_y - ball_center_y) / (float(player.height) / 2.0)
            self.Ball.velocityY = (-1.0 if self.Ball.velocityY < 0 else 1.0) * \
                                  0.5 * math.sqrt(self.Ball.velocityY**2 + self.Ball.velocityX**2) * \
                                  (1 + 0.8*ball_vel_y_modifier) + player.velocityY
            self.Ball.velocityY = min(self.Ball.velocityY, 3.5)

        # if self.Ball.rect.top >= self.LeftPlayer.rect.top and self.Ball.rect.bottom <= self.LeftPlayer.rect.bottom and self.Ball.rect.left <= self.LeftPlayer.rect.right:
        #     self.Ball.rect.left = self.LeftPlayer.rect.right
        #     self.Ball.velocityX *= -1  # Flip X velocity
        #     self.Ball.velocityY = (self.LeftPlayer.velocityY + self.Ball.velocityY) / 2
        #     if self.Ball.velocityY == 0:
        #         self.Ball.velocityY = random.randint(0, 1) * 2 - 1  # Get a 1 or -1
        # if self.Ball.rect.top >= self.RightPlayer.rect.top and self.Ball.rect.bottom <= self.RightPlayer.rect.bottom and self.Ball.rect.right >= self.RightPlayer.rect.left:
        #     self.Ball.rect.right = self.RightPlayer.rect.left
        #     self.Ball.velocityX *= -1  # Flip X velocity
        #     self.Ball.velocityY = (self.RightPlayer.velocityY + self.Ball.velocityY) / 2
        #     if self.Ball.velocityY == 0:
        #         self.Ball.velocityY = random.randint(0, 1) * 2 - 1  # Get a 1 or -1
        
        # Ball bounces off top and bottom walls
        self.Ball.confineVertical(self.TOP_WALL_Y, self.BOT_WALL_Y, flipVel=True)
        # if self.Ball.rect.top <= self.TOP_WALL_Y:
        #     self.Ball.setY()
        #     self.Ball.rect.top = self.TOP_WALL_Y
        #     self.Ball.velocityY *= -1.0
        #
        # if self.Ball.rect.top <= self.TOP_WALL_Y:
        #     self.Ball.rect.top = self.TOP_WALL_Y
        #     self.Ball.velocityY *= -1.0

        # Make sure players don't leave confines of screen
        self.LeftPlayer.confineVertical(self.TOP_WALL_Y, self.BOT_WALL_Y)
        self.RightPlayer.confineVertical(self.TOP_WALL_Y, self.BOT_WALL_Y)
        # if self.LeftPlayer.rect.top < self.TOP_WALL_Y:
        #     self.LeftPlayer.rect.top = self.TOP_WALL_Y
        # elif self.LeftPlayer.rect.bottom > self.BOT_WALL_Y:
        #     self.LeftPlayer.rect.bottom = self.BOT_WALL_Y
        # if self.RightPlayer.rect.top < self.TOP_WALL_Y:
        #     self.RightPlayer.rect.top = self.TOP_WALL_Y
        # elif self.RightPlayer.rect.bottom > self.BOT_WALL_Y:
        #     self.RightPlayer.rect.bottom = self.BOT_WALL_Y

        # Check if point is scored
        done = False
        if(self.Ball.rect.right > self.RightPlayer.rect.right):
            self.RightPlayer.score -= 1
            self.LeftPlayer.score += 1
            self.RightPlayer.timestep_reward = -1
            self.LeftPlayer.timestep_reward = -1
            done = True
        if(self.Ball.rect.left < self.LeftPlayer.rect.left) :
            self.LeftPlayer.score -= 1
            self.RightPlayer.score += 1
            self.LeftPlayer.timestep_reward = -1
            self.RightPlayer.timestep_reward = 1
            done = True

        # Apply velocities
        for sprite in self.allSprites:
            sprite.applyVelocities()
        
        return dict(done=done, leftPlayerRew=self.LeftPlayer.timestep_reward, rightPlayerRew=self.RightPlayer.timestep_reward)

    def render(self):
        self.canvas.fill(self.BG_COLOR)
        self.allSprites.draw(self.canvas)
        pygame.display.flip()

        if self.framerate > 0:
            self.clock.tick(self.framerate)

    def getScreenRGB(self):
        """
        Returns the current game screen in RGB format.
        Taken from https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/base/pygamewrapper.py
        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).
        """
        frame = pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)
        frame = np.rot90(frame, 1, axes=(0, 1))
        frame = np.flipud(frame)

        return frame
    
    def getScreenGrayscale(self):
        """
        Gets the current game screen in Grayscale format. Converts from RGB using relative lumiance.
        Taken from https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/ple.py
        Returns
        --------
        numpy uint8 array
                Returns a numpy array with the shape (width, height).
        """
        frame = self.getScreenRGB()
        frame = 0.21 * frame[:, :, 0] + 0.72 * frame[:, :, 1] + 0.07 * frame[:, :, 2]
        frame = np.round(frame).astype(np.uint8)

        return frame

    def getScreenBlackWhite(self):
        frame = self.getScreenRGB()
        frame = 0.21 * frame[:, :, 0] + 0.72 * frame[:, :, 1] + 0.07 * frame[:, :, 2]
        frame = frame > 80.0

        return frame

    def nonVisualObs(self):
        info = {'ballCenter': self.Ball.rect.center,
                'leftPlayerCenter': self.LeftPlayer.rect.center,
                'rightPlayerCenter': self.RightPlayer.rect.center}
        return info
