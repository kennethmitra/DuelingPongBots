import random
import pygame
import numpy as np

class EnhancedSprite(pygame.sprite.Sprite):
    def __init__(self, width, height, velocityX, velocityY):
        super().__init__()
        self.width = width
        self.height = height
        self.velocityX = velocityX
        self.velocityY = velocityY

    def setPosition(self, position):
        self.rect.x = position[0]
        self.rect.y = position[1]

    def deltaX(self, xDelta):
        self.rect.x += xDelta

    def deltaY(self, yDelta):
        self.rect.y += yDelta

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
            self.rect.left = EDGE_OFFSET
            self.rect.centery = CANVAS_HEIGHT // 2
        else:
            self.rect.right = CANVAS_WIDTH - EDGE_OFFSET
            self.rect.centery = CANVAS_HEIGHT // 2

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
            self.rect.left = self.EDGE_OFFSET
            self.rect.centery = CANVAS_HEIGHT // 2
        else:
            self.rect.right = CANVAS_WIDTH - self.EDGE_OFFSET
            self.rect.centery = CANVAS_HEIGHT // 2
        
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
        print("RESET BALL!!!!")
        self.setPosition((CANVAS_WIDTH//2, CANVAS_HEIGHT//2))

        self.velocityX = random.randint(-self.MAX_INITIAL_VEL, self.MAX_INITIAL_VEL)
        self.velocityY = random.randint(-self.MAX_INITIAL_VEL, self.MAX_INITIAL_VEL)

        # Make sure ball doesn't get stuck bouncing up and down
        while self.velocityX == 0:
            self.velocityX = random.randint(-self.MAX_INITIAL_VEL, self.MAX_INITIAL_VEL)

class Game:

    def __init__(self, CANVAS_WIDTH, CANVAS_HEIGHT):
        self.CANVAS_WIDTH = CANVAS_WIDTH
        self.CANVAS_HEIGHT = CANVAS_HEIGHT
        self.EDGE_OFFSET = 10
        self.TOP_WALL_Y = 10
        self.BOT_WALL_Y = self.CANVAS_HEIGHT - 10
        self.BG_COLOR = (255,255,255)
        self.canvas = pygame.display.set_mode([self.CANVAS_WIDTH, self.CANVAS_HEIGHT])

        self.allSprites = pygame.sprite.Group()
        self.playerSprites = pygame.sprite.Group()
        self.ball = pygame.sprite.Group()

        # Create both players
        self.LeftPlayer = Player(color=(255,0,0), width=10, height=50, speed=2, isLeftPaddle=True, EDGE_OFFSET=self.EDGE_OFFSET, CANVAS_WIDTH=self.CANVAS_WIDTH, CANVAS_HEIGHT=self.CANVAS_HEIGHT)
        self.RightPlayer = Player(color=(0,0,255), width=10, height=50, speed=2, isLeftPaddle=False, EDGE_OFFSET=self.EDGE_OFFSET, CANVAS_WIDTH=self.CANVAS_WIDTH, CANVAS_HEIGHT=self.CANVAS_HEIGHT)
        self.allSprites.add(self.LeftPlayer)
        self.allSprites.add(self.RightPlayer)
        self.playerSprites.add(self.LeftPlayer)
        self.playerSprites.add(self.RightPlayer)
        # Create ball
        self.Ball = Ball(color=(0,255,0), width=10, height=10, MAX_INITIAL_VEL=3, CANVAS_WIDTH=CANVAS_WIDTH, CANVAS_HEIGHT=CANVAS_HEIGHT)
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
        
        # Check collisions
        playersHit = pygame.sprite.spritecollide(self.Ball, self.playerSprites, dokill=False)
        
        # Handle collisions
        for player in playersHit: # Should only loop at most once
            self.Ball.velocityX *= -1 # Flip X velocity
           # self.Ball.velocityY += player.velocityY
        
        # Check wall collisions
        if self.Ball.rect.top < self.TOP_WALL_Y or self.Ball.rect.bottom > self.BOT_WALL_Y:
            self.Ball.velocityY *= -1

        # Make sure players don't leave confines of screen
        if self.LeftPlayer.rect.top < self.TOP_WALL_Y:
            self.LeftPlayer.rect.top = self.TOP_WALL_Y
        elif self.LeftPlayer.rect.bottom > self.BOT_WALL_Y:
            self.LeftPlayer.rect.bottom = self.BOT_WALL_Y
        if self.RightPlayer.rect.top < self.TOP_WALL_Y:
            self.RightPlayer.rect.top = self.TOP_WALL_Y
        elif self.RightPlayer.rect.bottom > self.BOT_WALL_Y:
            self.RightPlayer.rect.bottom = self.BOT_WALL_Y

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

    def getScreenRGB(self):
        """
        Returns the current game screen in RGB format.
        Taken from https://github.com/ntasfi/PyGame-Learning-Environment/blob/master/ple/games/base/pygamewrapper.py
        Returns
        --------
        numpy uint8 array
            Returns a numpy array with the shape (width, height, 3).
        """

        return pygame.surfarray.array3d(pygame.display.get_surface()).astype(np.uint8)
    
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

    def getExtraInfo(self):
        info = {'ballPos':self.Ball.rect, 'leftPlayerPos':self.LeftPlayer.rect, 'rightPlayerPos':self.RightPlayer.rect}
        return info
