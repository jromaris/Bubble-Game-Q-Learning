import time
import random
from objs.constants import *
from objs.bubble_file import *
from objs.grid_file import *
from objs.shooter_file import *
from objs.game_objects import *
from objs.q_learning import ReplayBuffer, DQN, select_epsilon_greedy_action, train_step
import pygame as pg
import numpy as np
import tensorflow as tf
import pprint
from objs.plotter import gudermannian, genlogistic, genlog_func

TEST = False
MODELS_PATH = 'new_models'

pg.init()


def reset_game():
	# Create background
	background = Background()

	# Initialize gun, position at bottom center of the screen
	gun = Shooter(pos=BOTTOM_CENTER)
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
	mouse_pos = (DISP_W / 2, DISP_H / 2)

	return game, background, grid_manager, gun, mouse_pos


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


def train():
	num_actions = 180 - 30
	num_episodes = 1000
	epsilon = 1
	buffer = ReplayBuffer(100000)
	batch_size = 32
	discount = 0.99
	cur_frame = 0

	main_nn = DQN(num_actions=num_actions)
	target_nn = DQN(num_actions=num_actions)
	optimizer = tf.keras.optimizers.Adam(1e-4)
	mse = tf.keras.losses.MeanSquaredError()

	gun_fired = True

	# Start training. Play game once and then train with a batch.
	last_100_ep_rewards = []
	angles = [i for i in range(15, 165)]
	action = random.randint(15, 164)

	for episode in range(num_episodes + 1):
		first = True

		game, background, grid_manager, gun, mouse_pos = reset_game()

		ep_reward, done = 0, False
		while not done:	 # or won game
			handle_game_events()

			state = grid_manager.grid_state

			# Draw BG first
			background.draw()

			# Check collision with bullet and update grid as needed
			grid_manager.view(gun, game)
			state_in = tf.expand_dims(state, axis=0)

			if not first:
				gun_fired = gun.fire()
				if not gun_fired:
					action = select_epsilon_greedy_action(main_nn, state_in,
														  genlog_func(epsilon), num_actions)
					# print('\tAction: ', action)
			else:
				action = random.randint(0, len(angles) - 1)
				first = False

			gun.rotate(angles[action])  # Rotate the gun if the mouse is moved

			gun.draw_bullets()  # Draw and update bullet and reloads

			game.drawScore()  # draw score

			pg.display.update()
			clock.tick(60)  # 60 FPS

			next_state, reward = grid_manager.gameInfo(game)

			ep_reward += reward

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
				# print('Reward: ', reward)
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
	num_actions = 180 - 30

	main_nn = tf.keras.models.load_model(MODELS_PATH)
	angles = [i for i in range(15, 165)]
	action = random.randint(15, 164)

	first = True
	gun_fired = False

	game, background, grid_manager, gun, mouse_pos = reset_game()

	ep_reward, done = 0, False
	while not done:  # or won game
		handle_game_events()

		background.draw()

		state = grid_manager.grid_state
		grid_manager.view(gun, game)

		if not first:
			gun_fired = gun.fire()
		if not gun_fired:
			state_in = tf.expand_dims(state, axis=0)
			action = select_epsilon_greedy_action(main_nn, state_in,
												  genlog_func(0), num_actions)

		gun.rotate(angles[action])  # Rotate the gun if the mouse is moved

		gun.draw_bullets()  # Draw and update bullet and reloads

		game.drawScore()  # draw score

		pg.display.update()
		clock.tick(60)  # 60 FPS

		done = game.over  # or game.won


def main():
	if TEST:
		test()
	else:
		train()


if __name__ == '__main__':
	main()
