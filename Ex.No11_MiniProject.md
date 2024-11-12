# Ex.No: 11  Mini Project 
### DATE:                                                                            
### REGISTER NUMBER : 212222240117
### AIM: 
To write a python program to simulate a tower defense game using deep learning algorithms.
### Algorithm:
#### 1.Initialize Game Components

Import necessary libraries (Pygame, Numpy, TensorFlow, Pandas, Matplotlib).
Set up Pygame screen, colors, and basic game settings (FPS, screen size, tile size).

#### 2.Define Game Objects

Tower Class: Towers have a range, damage, and can attack enemies within range.
Enemy Class: Enemies have health, speed, and AI-driven movement towards the closest tower.
EnemyAI Model: Simple neural network that determines movement direction based on enemy-tower positioning.

#### 3.Gameplay Loop

Towers: Display on screen and check for enemies within range to attack.
Enemies: Use EnemyAI to locate and move towards the nearest tower. Record each move.
Damage Handling: Towers inflict damage on enemies within range. Remove enemies when health reaches zero.
Update Screen: Refresh at specified FPS.

#### 4.End-of-Game Data Logging and Visualization

Save enemies’ movement paths to an Excel file.
Plot and save movement visualization using Matplotlib.

### Program:

```
import pygame 
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Tower Defense")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game settings
FPS = 60
TILE_SIZE = 20
ENEMY_SPEED = 2

# Tower Class
class Tower:
    def __init__(self, x, y, range_radius, damage):
        self.x = x
        self.y = y
        self.range_radius = range_radius
        self.damage = damage
        self.upgrades = 0

    def shoot(self, enemy):
        # Calculate distance between tower and enemy
        dist = np.sqrt((enemy.x - self.x) ** 2 + (enemy.y - self.y) ** 2)
        if dist <= self.range_radius:
            enemy.take_damage(self.damage)

    def upgrade(self):
        self.range_radius += 10
        self.damage += 5
        self.upgrades += 1

    def draw(self, surface):
        pygame.draw.circle(surface, BLUE, (self.x, self.y), self.range_radius, 1)
        pygame.draw.rect(surface, GREEN, (self.x - 15, self.y - 15, 30, 30))

# Enemy Class
class Enemy:
    def __init__(self, x, y, health, ai_model):
        self.x = x
        self.y = y
        self.health = health
        self.speed = ENEMY_SPEED
        self.ai = ai_model
        self.moves = []  # Store movement history

    def move(self, towers):
        # Move towards the closest tower
        closest_tower = self.ai.find_closest_tower(self.x, self.y, towers)
        self.x, self.y = self.ai.move_towards(self.x, self.y, closest_tower)

        # Record the current position after movement
        self.moves.append((self.x, self.y))

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            return True  # Enemy is dead
        return False

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.x, self.y, 30, 30))

# AI Model for Enemy behavior
class EnemyAI:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        # Define the input layer with the shape (2,) for the 2D input (x, y)
        inputs = Input(shape=(2,))
        x = Dense(16, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(4, activation='softmax')(x)  # 4 possible actions (directions)
        
        # Create the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def find_closest_tower(self, enemy_x, enemy_y, towers):
        # Find the closest tower based on Euclidean distance
        closest_tower = min(towers, key=lambda tower: np.sqrt((tower.x - enemy_x) ** 2 + (tower.y - enemy_y) ** 2))
        return closest_tower

    def move_towards(self, enemy_x, enemy_y, tower):
        # Move towards the tower based on their relative positions
        if tower.x > enemy_x:
            enemy_x += ENEMY_SPEED
        elif tower.x < enemy_x:
            enemy_x -= ENEMY_SPEED
        
        if tower.y > enemy_y:
            enemy_y += ENEMY_SPEED
        elif tower.y < enemy_y:
            enemy_y -= ENEMY_SPEED

        return enemy_x, enemy_y

# Save enemy movement to Excel
def save_movement_to_excel(enemies):
    movement_data = []
    for enemy_id, enemy in enumerate(enemies):
        for step, (x, y) in enumerate(enemy.moves):
            movement_data.append({'Enemy': enemy_id, 'Step': step, 'X Position': x, 'Y Position': y})

    # Create a DataFrame
    df = pd.DataFrame(movement_data)
    # Save to Excel
    df.to_excel("enemy_movements.xlsx", index=False)

# Visualize enemy movements from the Excel sheet
def visualize_movement_from_excel():
    df = pd.read_excel("enemy_movements.xlsx")
    
    plt.figure(figsize=(10, 6))
    
    for enemy_id in df['Enemy'].unique():
        enemy_data = df[df['Enemy'] == enemy_id]
        plt.plot(enemy_data['X Position'], enemy_data['Y Position'], marker='o', label=f'Enemy {enemy_id}')

    plt.title('Enemy Movement Paths')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid()
    plt.savefig("enemy_movements.png")  # Save the plot as an image
    plt.show()  # Show the plot

# Main Game Loop
def game_loop():
    running = True
    clock = pygame.time.Clock()

    # Towers and enemies
    towers = [Tower(200, 300, 100, 10), Tower(400, 500, 150, 15)]
    ai = EnemyAI()
    
    # Add 10 enemies with random start positions
    enemies = [Enemy(random.randint(0, WIDTH), random.randint(0, HEIGHT), 100, ai) for _ in range(10)]

    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update and draw towers
        for tower in towers:
            tower.draw(screen)

        # Update and draw enemies
        for enemy in enemies:
            enemy.move(towers)  # Enemies move towards the closest tower
            for tower in towers:
                tower.shoot(enemy)
            enemy.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

    # Save movement history to Excel after the game loop ends
    save_movement_to_excel(enemies)
    # Visualize movement after saving
    visualize_movement_from_excel()


if __name__ == "__main__":
    game_loop()
```

### Output:

Tower defense:

![Screenshot 2024-11-12 102131](https://github.com/user-attachments/assets/5b0e1c2a-8c4c-48de-ad62-4e4887bbf244)

Path Finding Algorithm:

![Screenshot 2024-11-12 102156](https://github.com/user-attachments/assets/6682e840-99b5-4aa2-875f-c4deb36915f2)

### Result:
Thus the simple tower defense game was implemented using deep learning algorithms.
