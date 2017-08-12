# Self Driving Car - Environment (Map)

import time
import numpy as np
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty

from ai import DQN # AI using PyTorch
# from ai_tf import DQN # AI using TensorFlow
# from ai_tf_alt import DQN # AI using TensorFlow contrib.layers (more efficient)


# Add this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')


# Introduce last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
length = 0 # the length of the last drawing
n_points = 0 # the total number of points in the last drawing

# Get our AI, which we call 'brain', and that contains our Neural Network that represent our Q-function
brain = DQN(input_sz=5, output_sz=3, hidden_layer_sizes=[60], gamma=0.9) # 5 inputs (dimensionality), 3 outputs (actions), gamma = 0.9
action2rotation = [0, 20, -20]

scores = [] # sliding window of rewards with respect to time
last_reward = 0
last_signal = [0, 0, 0, 0, 0]
last_distance = 0

# Constant reward
PUNISHMENT = -6.0
GOAL_REWARD = 2.0
STEP_COST = -0.3
CLOSER_REWARD = 0.2

# Constant speed
FAST_SPEED = 6
SLOW_SPEED = 1


# Initialize the map
first_update = True # use this trick to initialize the map only once
def init_map():
	global sand # sand is an array that has as many cells as our GUI has pixels. Each cell is 1 if there is sand else 0.
	global goal_x # x-coordinate of the goal
	global goal_y # y-coordinate of the goal
	global first_update # specify the first_update flag
	sand = np.zeros((longueur, largeur)) # initialize the sand array with only zeros
	goal_x = 20 # the goal to reach is at the upper left of the map (x is 20 not 0 because the car gets bad reward if it touches the wall)
	goal_y = largeur - 20 # y is (largeur - 20) not largeur because the car gets bad reward if it touches the wall
	first_update = False


# Create the Car class
class Car(Widget):

	angle = NumericProperty(0) # the angle of the car (angle between the x-axis of the map and the axis of the car)
	rotation = NumericProperty(0) # the last rotation of the car (after taking the action, the car does a rotation of 0, 20 or -20 degrees)
	velocity_x = NumericProperty(0) # the x-coordinate of the velocity vector
	velocity_y = NumericProperty(0) # the y-coordinate of the velocity vector
	velocity = ReferenceListProperty(velocity_x, velocity_y) # the velocity vector
	sensor1_x = NumericProperty(0) # the x-coordinate of the first sensor (the one that looks forward)
	sensor1_y = NumericProperty(0) # the y-coordinate of the first sensor (the one that looks forward)
	sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # the first sensor vector
	sensor2_x = NumericProperty(0) # the x-coordinate of the second sensor (the one that looks 30 degrees to the left)
	sensor2_y = NumericProperty(0) # the y-coordinate of the second sensor (the one that looks 30 degrees to the left)
	sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) # the second sensor vector
	sensor3_x = NumericProperty(0) # the x-coordinate of the third sensor (the one that looks 30 degrees to the right)
	sensor3_y = NumericProperty(0) # the y-coordinate of the third sensor (the one that looks 30 degrees to the right)
	sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # the third sensor vector
	signal1 = NumericProperty(0) # the signal received by sensor 1
	signal2 = NumericProperty(0) # the signal received by sensor 2
	signal3 = NumericProperty(0) # the signal received by sensor 3

	def get_signal(self, sensor_x, sensor_y):
		x, y = int(sensor_x), int(sensor_y)
		signal = int(np.sum(sand[x - 10: x + 10, y - 10: y + 10])) / 400.0 # density of sand around sensor
		# if the sensor is out of the map (the car is facing one edge of the map)
		if sensor_x > longueur - 10 or sensor_x < 10 or sensor_y > largeur - 10 or sensor_y < 10:
			signal = 1.0 # sensor detects full sand
		return signal

	def move(self, rotation):
		self.pos = Vector(*self.velocity) + self.pos # update the position of the car according to its last position and velocity
		self.rotation = rotation # get the rotation of the car
		self.angle = self.angle + self.rotation # update the angle
		self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos # update the position of the sensor 1
		self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos # update the position of the sensor 2
		self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos # update the position of the sensor 3
		self.signal1 = self.get_signal(self.sensor1_x, self.sensor1_y) # signal 1: density of sand around sensor 1
		self.signal2 = self.get_signal(self.sensor2_x, self.sensor2_y) # signal 2: density of sand around sensor 2
		self.signal3 = self.get_signal(self.sensor3_x, self.sensor3_y) # signal 3: density of sand around sensor 3


class Ball1(Widget):
	pass # sensor 1

class Ball2(Widget):
	pass # sensor 2

class Ball3(Widget):
	pass # sensor 3

class Goal1(Widget):
	pass # goal 1

class Goal2(Widget):
	pass # goal 2


# Create the Game class
class Game(Widget):

	car = ObjectProperty(None) # get the car object from our kivy file
	ball1 = ObjectProperty(None) # get the sensor 1 object from our kivy file
	ball2 = ObjectProperty(None) # get the sensor 2 object from our kivy file
	ball3 = ObjectProperty(None) # get the sensor 3 object from our kivy file
	goal1 = ObjectProperty(None) # get the goal 1 object from our kivy file
	goal2 = ObjectProperty(None) # get the goal 2 object from our kivy file

	def serve_car(self):
		# start the car when we launch the application
		self.car.center = self.center # the car will start at the center of the map
		self.car.velocity = Vector(FAST_SPEED, 0) # the car will start to go horizontally to the right with a speed of 6

	def update(self, dt):
		# the big update function that updates everything that needs to be updated at each discrete time t
		# when reaching a new state (getting new signals from sensors)

		global brain # specify the global variables (the brain of the car, that is our AI)
		global scores # specify the global variables (the sliding window of rewards)
		global last_reward # specify the global variables (the last reward)
		global last_signal # specify the global variables (the last signal)
		global last_distance # specify the global variables (the last distance)
		global goal_x # specify the global variables (x-coordinate of the goal)
		global goal_y # specify the global variables (y-coordinate of the goal)
		global longueur
		global largeur

		longueur = self.width # width of the map (horizontal edge)
		largeur = self.height # height of the map (vertical edge)

		# trick to initialize the map only once
		if first_update:
			init_map()

			self.goal1.pos = [0, largeur - 100]
			self.goal2.pos = [longueur - 100, 0]

			dx = goal_x - self.car.x # difference of x-coordinate between the goal and the car
			dy = goal_y - self.car.y # difference of y-coordinate between the goal and the car

			orientation = Vector(*self.car.velocity).angle((dx, dy)) / 180.0 # direction of the car with respect to the goal (if the car is heading perfectly towards the goal, then orientation = 0)
			last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] # our input state vector to NN, composed of the three signals received by the three sensors, plus the orientation and -orientation

			diff = np.array([dx, dy])
			last_distance = np.sqrt(diff.dot(diff))

		last_action = brain.select_action(last_signal, temperature=75)
		rotation = action2rotation[last_action] # convert the action played (0, 1 or 2) into the rotation angle (0, 20 or -20)
		self.car.move(rotation) # move the car according to this rotation angle

		diff = np.array([goal_x - self.car.x, goal_y - self.car.y])
		distance = np.sqrt(diff.dot(diff)) # get the new distance between the car and the goal after the car moved

		self.ball1.pos = self.car.sensor1 # update the position of the first sensor (ball1) after the car moved
		self.ball2.pos = self.car.sensor2 # update the position of the second sensor (ball2) after the car moved
		self.ball3.pos = self.car.sensor3 # update the position of the third sensor (ball3) after the car moved

		if sand[int(self.car.x), int(self.car.y)] > 0:
			# if the car is on the sand
			self.car.velocity = Vector(SLOW_SPEED, 0).rotate(self.car.angle) # it is slowed down (speed = 1)
			last_reward = PUNISHMENT # it gets a bad reward (PUNISHMENT)
		else:
			self.car.velocity = Vector(FAST_SPEED, 0).rotate(self.car.angle) # it goes to a normal speed (spped = 6)
			last_reward = STEP_COST # it gets a step cost reward
			if distance < last_distance:
				last_reward = CLOSER_REWARD # if it gets closer to the goal, it will get a slightly positive reward

		# if the car is in the edge of the frame
		# it will not be slowed down, but gets a bad reward (PUNISHMENT)
		if self.car.x < 10:
			self.car.x = 10
			last_reward = PUNISHMENT
		if self.car.x > self.width - 10:
			self.car.x = self.width - 10
			last_reward = PUNISHMENT
		if self.car.y < 10:
			self.car.y = 10
			last_reward = PUNISHMENT
		if self.car.y > self.height - 10:
			self.car.y = self.height - 10
			last_reward = PUNISHMENT

		# when the car reaches its goal
		if distance < 75:
			goal_x = self.width - goal_x # the goal becomes the bottom right corner of the map, and vice versa
			goal_y = self.height - goal_y # the goal becomes the bottom right corner of the map, and vice versa
			last_reward = GOAL_REWARD # reaches the goal, and gets a big reward

		orientation = Vector(*self.car.velocity).angle((goal_x - self.car.x, goal_y - self.car.y)) / 180.0 # direction of the car with respect to the goal (if the car is heading perfectly towards the goal, then orientation = 0)
		next_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation] # our input state vector to NN, composed of the three signals received by the three sensors, plus the orientation and -orientation

		# # update the brain (let AI learn)
		brain.update(last_signal, last_action, last_reward, next_signal)

		# append the score (mean of the last 100 rewards to the reward window)
		scores.append(brain.score())

		# update the last signal and last distance
		last_signal = next_signal
		last_distance = distance


# Painting for graphic interface
class MyPaintWidget(Widget):

	def on_touch_down(self, touch):
		# put some sand when we do a left click
		global length, n_points, last_x, last_y
		with self.canvas:
			Color(0.8, 0.7, 0)
			touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
			last_x = int(touch.x)
			last_y = int(touch.y)
			n_points = 0
			length = 0
			x = np.clip(touch.x, 0, longueur - 1)
			y = np.clip(touch.y, 0, largeur - 1)
			sand[int(x), int(y)] = 1

	def on_touch_move(self, touch):
		# put some sand when we move the mouse while pressing left
		global length, n_points, last_x, last_y
		if touch.button == 'left':
			touch.ud['line'].points += [touch.x, touch.y]
			x, y = int(touch.x), int(touch.y)
			length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
			n_points += 1
			density = n_points / length
			touch.ud['line'].width = int(20 * density + 1)
			row1 = int(np.clip(x - 10, 0, longueur - 2))
			row2 = int(np.clip(x + 10, 1, longueur - 1))
			col1 = int(np.clip(y - 10, 0, largeur - 2))
			col2 = int(np.clip(y + 10, 1, largeur - 1))
			sand[row1 : row2, col1 : col2] = 1
			last_x = x
			last_y = y


# API and switches interface
class CarApp(App):

	def build(self):
		# build the App
		parent = Game()
		parent.serve_car()
		Clock.schedule_interval(parent.update, 1.0 / 60.0)
		self.painter = MyPaintWidget()
		clearbtn = Button(text='clear')
		savebtn = Button(text='save', pos=(parent.width, 0))
		loadbtn = Button(text='load', pos=(2*parent.width, 0))
		clearbtn.bind(on_release=self.clear_canvas)
		savebtn.bind(on_release=self.save)
		loadbtn.bind(on_release=self.load)
		parent.add_widget(self.painter)
		parent.add_widget(clearbtn)
		parent.add_widget(savebtn)
		parent.add_widget(loadbtn)
		return parent

	def clear_canvas(self, obj):
		global sand
		self.painter.canvas.clear()
		sand = np.zeros((longueur, largeur))

	def save(self, obj):
		print('Saving brain...')
		brain.save()
		plt.plot(scores)
		plt.show()

	def load(self, obj):
		print('Loading last saved brain...')
		brain.load()


# Run the App
if __name__ == '__main__':
	CarApp().run()

