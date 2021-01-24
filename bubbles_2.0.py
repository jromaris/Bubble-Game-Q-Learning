import time
import random
from objs.constants import *
from objs.bubble_file import *
from objs.grid_file import *
from objs.shooter_file import *
from objs.game_objects import *
from objs.q_learning import ReplayBuffer, DQN, select_epsilon_greedy_action, train_step
import pygame as pg
import tensorflow as tf
import pprint


pg.init()


def reset_game():
	# Create background
	background = Background()

	# Initialize gun, position at bottom center of the screen
	gun = Shooter(pos=BOTTOM_CENTER)
	gun.putInBox()

	print(gun.loaded.color)
	print(gun.reload1.color)
	print(gun.reload2.color)
	print(gun.reload3.color)
	grid_manager = GridManager()
	game = Game()
	# cheat_manager = CheatManager(grid_manager, gun)
	# Check collision with bullet and update grid as needed
	grid_manager.view(gun, game)

	# print(len(grid_manager.grid))
	# print("FILA 0 COLUMNA 0")
	# print(grid_manager.grid[0][0].pos)
	# print("FILA 0 COLUMNA 1")
	# print(grid_manager.grid[0][1].pos)
	# print("FILA 0 COLUMNA 19")
	# print(grid_manager.grid[0][19].pos)
	# print("FILA 1 COLUMNA 0")
	# print(grid_manager.grid[1][0].pos)
	# print("FILA 1 COLUMNA 1")
	# print(grid_manager.grid[1][1].pos)
	# print("FILA 1 COLUMNA 19")
	# print(grid_manager.grid[1][19].pos)

	# print(grid_manager.grid[10][0].color)
	# print(grid_manager.grid[10][1].color)

	# print(BG_COLOR)
	# print("CURR OK")
	# print(grid_manager.grid_curr_ok)
	# print("CURR NOK")
	# print(grid_manager.grid_curr_nok)
	# Starting mouse position
	mouse_pos = (DISP_W / 2, DISP_H / 2)

	return game, background, grid_manager, gun, mouse_pos


def step(game, grid_manager, action):
	grid_state, reward = grid_manager.gameInfo(game)
	pass


def main():

	num_actions = 14
	num_episodes = 1000
	epsilon = 1.0
	buffer = ReplayBuffer(100000)
	batch_size = 32
	discount = 0.99
	cur_frame = 0

	main_nn = DQN(num_actions=num_actions)
	target_nn = DQN(num_actions=num_actions)
	optimizer = tf.keras.optimizers.Adam(1e-4)
	mse = tf.keras.losses.MeanSquaredError()

	# Start training. Play game once and then train with a batch.
	last_100_ep_rewards = []
	for episode in range(num_episodes + 1):
		game, background, grid_manager, gun, mouse_pos = reset_game()

		ep_reward, done = 0, False
		while not game.over:	 # or won game
			state_in = tf.expand_dims(state, axis=0)
			action = select_epsilon_greedy_action(state_in, epsilon)


			grid_state, reward = grid_manager.gameInfo(game)
			# next_state, reward, done, info = env.step(action)

			ep_reward += reward
			# Save to experience replay.
			buffer.add(state, action, reward, next_state, done)
			state = next_state
			cur_frame += 1
			# Copy main_nn weights to target_nn.
			if cur_frame % 2000 == 0:
				target_nn.set_weights(main_nn.get_weights())

			# Train neural network.
			if len(buffer) >= batch_size:
				states, actions, rewards, next_states, dones = buffer.sample(batch_size)
				loss = train_step(states, actions, rewards, next_states, dones)

		if episode < 950:
			epsilon -= 0.001

		if len(last_100_ep_rewards) == 100:
			last_100_ep_rewards = last_100_ep_rewards[1:]
		last_100_ep_rewards.append(ep_reward)

		if episode % 50 == 0:
			print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
				  f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')

	# env.close()


	# pretty self-explanatory
	while not game.over:		

		# quit when you press the x
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				quit()

			# get mouse position
			if event.type == pg.MOUSEMOTION:
				mouse_pos = pg.mouse.get_pos()
				
			# if you click, fire a bullet
			if event.type == pg.MOUSEBUTTONDOWN:
				gun.fire()
			
			if event.type == pg.KEYDOWN:
				# cheat_manager.view(event) # if a key is pressed, the cheat manager should know about it

				# Ctrl+C to quit
				if event.key == pg.K_c and pg.key.get_mods() & pg.KMOD_CTRL:
					pg.quit()
					quit()

		# Draw BG first
		background.draw()

		# Check collision with bullet and update grid as needed
		grid_manager.view(gun, game)

		# ACA IRIA LA PARTE DE LA RED NEURONAL
		# print(grid_manager.grid[0][0].__dict__)

		gun.rotate(mouse_pos)			# Rotate the gun if the mouse is moved
		gun.draw_bullets()				# Draw and update bullet and reloads

		game.drawScore()				# draw score

		pg.display.update()
		clock.tick(60)					# 60 FPS

		# print(gun.angle)
	game.gameOverScreen(grid_manager, background)


if __name__ == '__main__': 
	while True:
		main()
