import pygame as pg
import random
from objs.grid_file import GridManager
from objs.shooter_file import Shooter
from objs.game_objects import Game, Background
from objs.q_learning import ReplayBuffer, DQN, select_epsilon_greedy_action, train_step, do_trained_action
import numpy as np
import tensorflow as tf
from objs.plotter import genlogistic, genlog_func
from objs.constants import *


def reset_game():
	# Create background
	if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
		background = Background()
	else:
		background = None

	# Initialize gun, position at bottom center of the screen
	gun = Shooter(pos=BOTTOM_CENTER)
	print(BOTTOM_CENTER)
	if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
		gun.putInBox()

	# print(gun.loaded.color)
	# print(gun.reload1.color)
	# print(gun.reload2.color)
	# print(gun.reload3.color)
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

	return game, background, grid_manager, gun


def handle_game_events():

	for event in pg.event.get():
		if event.type == pg.QUIT:
			pg.quit()
			quit()

		# get mouse position
		if event.type == pg.MOUSEMOTION:
			pass

		# if you click, fire a bullet
		if event.type == pg.MOUSEBUTTONDOWN:
			pass

		if event.type == pg.KEYDOWN:
			# Ctrl+C to quit
			if event.key == pg.K_c and pg.key.get_mods() & pg.KMOD_CTRL:
				pg.quit()
				quit()


def train_logic():
	num_episodes = 1000
	epsilon = 1
	buffer = ReplayBuffer(100000)
	batch_size = 32
	discount = 0.85
	cur_frame = 0

	gun_fired = True

	# Start training. Play game once and then train with a batch.
	last_100_ep_rewards = []
	limit_a, limit_b = 15, 165
	angle_step = 0.5
	angles = [i * angle_step for i in range(int(limit_a / angle_step), int(limit_b / angle_step))]

	action = random.randint(0, len(angles) - 1)
	num_actions = len(angles)

	main_nn = DQN(num_actions=num_actions)
	target_nn = DQN(num_actions=num_actions)
	optimizer = tf.keras.optimizers.Adam(1e-4)
	mse = tf.keras.losses.MeanSquaredError()

	for episode in range(num_episodes):
		first = True

		game, _, grid_manager, gun = reset_game()

		ep_reward, done = 0, False
		while not done:  # or won game
			if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
				handle_game_events()

			state = grid_manager.grid_state

			# Check collision with bullet and update grid as needed
			grid_manager.view(gun, game)

			if not first:
				gun_fired = gun.fire()
				if not gun_fired:
					state_in = tf.expand_dims(state, axis=0)
					action = select_epsilon_greedy_action(main_nn, state_in,
														  genlog_func(epsilon), num_actions)
			# print('\tAction: ', action)
			else:
				action = random.randint(0, len(angles) - 1)
				first = False

			gun.rotate(angles[action])  # Rotate the gun if the mouse is moved

			gun.draw_bullets()  # Draw and update bullet and reloads

			next_state, reward = grid_manager.gameInfo(game)
			ep_reward += reward

			game.drawScore()  # draw score

			done = game.over  # or game.won
			# Save to experience replay.
			buffer.add(state, action, reward, next_state, done)
			state = next_state
			cur_frame += 1
			# Copy main_nn weights to target_nn.
			if cur_frame % 2000 == 0:
				target_nn.set_weights(main_nn.get_weights())

			# Train neural network.
			if len(buffer) >= batch_size and not first and not gun_fired:
				# print('Reward: ', reward)
				states, actions, rewards, next_states, dones = buffer.sample(batch_size)
				loss = train_step(main_nn=main_nn, target_nn=target_nn, mse=mse, optimizer=optimizer,
								  states=states, actions=actions, rewards=rewards,
								  next_states=next_states, dones=dones, discount=discount,
								  num_actions=num_actions)

		print(f'Episode {episode}/{num_episodes}')
		print('\tgenlog_func(Epsilon): ', genlog_func(epsilon))
		epsilon -= 0.001
		if episode == 998:
			print('Ante-último')
		if len(last_100_ep_rewards) == 100:
			last_100_ep_rewards = last_100_ep_rewards[1:]
		last_100_ep_rewards.append(ep_reward)

		if episode % 50 == 0:
			print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
				  f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')

	main_nn.save(MODELS_PATH)
	del main_nn


def train_graphic():
	num_episodes = 1000
	epsilon = 1
	buffer = ReplayBuffer(100000)
	batch_size = 32
	discount = 0.99
	cur_frame = 0

	gun_fired = True

	# Start training. Play game once and then train with a batch.
	last_100_ep_rewards = []
	limit_a, limit_b = 15, 165
	angle_step = 0.5
	angles = [i * angle_step for i in range(int(limit_a/angle_step), int(limit_b/angle_step))]

	action = random.randint(0, len(angles)-1)
	num_actions = len(angles)

	main_nn = DQN(num_actions=num_actions)
	target_nn = DQN(num_actions=num_actions)
	optimizer = tf.keras.optimizers.Adam(1e-4)
	mse = tf.keras.losses.MeanSquaredError()

	for episode in range(num_episodes):
		first = True

		game, background, grid_manager, gun = reset_game()

		ep_reward, done = 0, False
		while not done:	 # or won game
			handle_game_events()

			state = grid_manager.grid_state

			# Draw BG first
			background.draw()

			# Check collision with bullet and update grid as needed
			grid_manager.view(gun, game)

			if not first:
				gun_fired = gun.fire()
				if not gun_fired:
					state_in = tf.expand_dims(state, axis=0)
					action = select_epsilon_greedy_action(main_nn, state_in,
														  genlog_func(epsilon), num_actions)
					# print('\tAction: ', action)
			else:
				action = random.randint(0, len(angles) - 1)
				first = False

			gun.rotate(angles[action])  # Rotate the gun if the mouse is moved

			gun.draw_bullets()  # Draw and update bullet and reloads

			next_state, reward = grid_manager.gameInfo(game)
			ep_reward += reward

			game.drawScore()  # draw score

			pg.display.update()

			clock.tick(60)  # 60 FPS

			done = game.over 	 # or game.won
			# Save to experience replay.
			buffer.add(state, action, reward, next_state, done)
			state = next_state
			cur_frame += 1
			# Copy main_nn weights to target_nn.
			if cur_frame % 2000 == 0:
				target_nn.set_weights(main_nn.get_weights())

			# Train neural network.
			if len(buffer) >= batch_size and not first and not gun_fired:
				print('Reward: ', reward)
				states, actions, rewards, next_states, dones = buffer.sample(batch_size)
				loss = train_step(main_nn=main_nn, target_nn=target_nn, mse=mse, optimizer=optimizer,
								states=states, actions=actions, rewards=rewards,
								next_states=next_states, dones=dones, discount=discount,
								num_actions=num_actions)

		print('genlog_func(Epsilon): ', genlog_func(epsilon))
		epsilon -= 0.001
		if episode == 998:
			print('Ante-último')
		if len(last_100_ep_rewards) == 100:
			last_100_ep_rewards = last_100_ep_rewards[1:]
		last_100_ep_rewards.append(ep_reward)

		if episode % 50 == 0:
			print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
				  f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')

	main_nn.save(MODELS_PATH)
	del main_nn


def test():

	main_nn = tf.keras.models.load_model('models', compile=False)
	limit_a, limit_b = 15, 165
	angle_step = 0.5
	angles = [i * angle_step for i in range(int(limit_a/angle_step), int(limit_b/angle_step))]

	action = random.randint(0, len(angles)-1)
	num_actions = len(angles)

	first = True
	gun_fired = True

	game, background, grid_manager, gun = reset_game()

	ep_reward, done = 0, False
	while not done:  # or won game
		handle_game_events()

		state = grid_manager.grid_state

		background.draw()

		grid_manager.view(gun, game)

		if not first:
			gun_fired = gun.fire()
			if not gun_fired:
				state_in = tf.expand_dims(state, axis=0)
				action = do_trained_action(main_nn, state_in)
				print('\tAction: ', action)
		else:
			action = random.randint(0, len(angles) - 1)
			first = False

		gun.rotate(angles[action])  # Rotate the gun if the mouse is moved

		gun.draw_bullets()  # Draw and update bullet and reloads

		game.drawScore()  # draw score

		pg.display.update()

		clock.tick(20)  # 60 FPS

		done = game.over  # or game.won


def main():
	if TRAIN_TEST:
		if TRAIN_TYPE == 'logic':
			train_logic()
		elif TRAIN_TYPE == 'graphic':
			train_graphic()
	else:
		test()


if __name__ == '__main__':
	main()
