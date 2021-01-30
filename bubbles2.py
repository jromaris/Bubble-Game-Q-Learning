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
import matplotlib.pyplot as plt
from collections import deque
import pickle

def reset_game(reward_paras, initial_grid):
    # Create background
    if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
        background = Background()
    else:
        background = None

    # Initialize gun, position at bottom center of the screen
    gun = Shooter(pos=BOTTOM_CENTER)
    # print(BOTTOM_CENTER)
    if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
        gun.putInBox()

    grid_manager = GridManager(initial_grid)
    game = Game()

    # Check collision with bullet and update grid as needed
    grid_manager.view(gun, game, reward_paras)

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


# reward_params : {'game over': -200, 'no hit': -2, 'hit': 1, 'balls_down_positive': True}
# epsilon_params : {constant: (False, 0), 'a': 0, 'k': 1, 'b': 1.5, 'q': 0.5, 'v': 0.12, 'm': 0, 'c': 1}
def train_logic(epsilon_paras, reward_paras, num_episodes=1000, batch_size=32, discount=0.92,
                amount_frames=2000, activation='tanh', model_n=0):

    epsilon = 1

    buffer = ReplayBuffer(100000+1)
    cur_frame = 0

    # Start training. Play game once and then train with a batch.
    last_100_ep_rewards = []
    limit_a, limit_b = 15, 165
    angle_step = 0.5
    angles = [i * angle_step for i in range(int(limit_a / angle_step), int(limit_b / angle_step))]
    print('angles:', angles)

    action = random.randint(0, len(angles) - 1)
    num_actions = len(angles)

    main_nn = DQN(num_actions=num_actions, activate=activation)
    target_nn = DQN(num_actions=num_actions, activate=activation)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    mse = tf.keras.losses.MeanSquaredError()
    if SAVE_SAMPLES:
        last_plays = deque(maxlen=5)
        to_save = []

    if USE_SAMPLES:
        saved = pickle.load(open('drive/Shareddrives/Redes/savedmodel' + str(model_n) +'.p', 'rb'))

    for episode in range(num_episodes):

        if USE_SAMPLES and len(saved) > episode:
            game, background, grid_manager, gun = reset_game(reward_paras, initial_grid=saved[episode])
        else:
            game, background, grid_manager, gun = reset_game(reward_paras, initial_grid=None)

        # Check collision with bullet and update grid as needed
        state = grid_manager.grid_state

        ep_reward, done = 0, False

        while not done:  # or won game

            if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
                handle_game_events()
                background.draw()

            gun.rotate(angles[action])  # Rotate the gun if the mouse is moved

            if not gun.fired.exists:
                # has to be in this order !!
                state = grid_manager.grid_state
                last_plays.append(grid_manager.grid)
                gun.fire()
                state_in = tf.expand_dims(state, axis=0)
                action = select_epsilon_greedy_action(main_nn, state_in,
                                                      genlog_func(epsilon, epsilon_paras), num_actions)
                # gun.rotate(angles[action])  # Rotate the gun if the mouse is moved
            else:
                next_state = grid_manager.view(gun, game, reward_paras)
                if next_state is not None:
                    reward = grid_manager.gameInfo(game, reward_paras)
                    # Save to experience replay.
                    buffer.add(state, action, reward, next_state, done)
                    ep_reward += reward
                    if len(buffer) >= batch_size:
                        # print('Reward: ', reward)
                        # print('Episode Reward: ', ep_reward)
                        # Train neural network.
                        # if ep_reward < -200:
                        # plt.imshow(state)
                        # plt.colorbar()
                        # plt.show()
                        # plt.imshow(next_state)
                        # plt.colorbar()
                        # plt.show()

                        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                        train_step(main_nn=main_nn, target_nn=target_nn, mse=mse, optimizer=optimizer,
                                   states=states, actions=actions, rewards=rewards,
                                   next_states=next_states, dones=dones, discount=discount,
                                   num_actions=num_actions)
                        cur_frame += 1
                        # Copy main_nn weights to target_nn.
                        if cur_frame % amount_frames == 0:
                            target_nn.set_weights(main_nn.get_weights())

            gun.draw_bullets()   # Draw and update bullet and reloads

            game.drawScore()  # draw score
            if not (TRAIN_TYPE == 'logic' and TRAIN_TEST):
                pg.display.update()
                clock.tick(60)  # 60 FPS

            done = game.over  # or game.won
        if SAVE_SAMPLES:
            to_save.append(last_plays.popleft())

        print(f'Episode {episode}/{num_episodes}')
        print('\tgenlog_func(Epsilon): ', genlog_func(epsilon, epsilon_paras))
        if not epsilon_paras['constant'][0]:
            epsilon -= 1 / num_episodes

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)

        if episode % 50 == 0:
            print(f'Episode {episode}/{num_episodes}. Epsilon: {genlog_func(epsilon, epsilon_paras):.3f}. '
                  f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
            print('len buffer: ', len(buffer))

    if SAVE_SAMPLES:
        pickle.dump(obj=to_save, file=open('drive/Shareddrives/Redes/savedmodel' + str(model_n) +'.p', 'wb'))
    main_nn.save(MODELS_PATH + '/model' + str(model_n))
    del main_nn


def test(reward_paras):
    main_nn = tf.keras.models.load_model('models/model7', compile=False)
    limit_a, limit_b = 15, 165
    angle_step = 0.5
    angles = [i * angle_step for i in range(int(limit_a / angle_step), int(limit_b / angle_step))]

    game, background, grid_manager, gun = reset_game(reward_paras, initial_grid=None)

    ep_reward, done = 0, False
    while not done:  # or won game
        handle_game_events()

        background.draw()

        grid_manager.view(gun, game, reward_paras)
        state = grid_manager.grid_state

        if not gun.fired.exists:
            # plt.imshow(state)
            # plt.colorbar()
            # plt.show()
            state_in = tf.expand_dims(state, axis=0)
            action = do_trained_action(main_nn, state_in)
            # print('\tAction: ', action)

            gun.rotate(angles[action])  # Rotate the gun if the mouse is moved
            gun.fire()

        gun.draw_bullets()  # Draw and update bullet and reloads

        game.drawScore()  # draw score

        pg.display.update()

        clock.tick(20)  # 60 FPS

        done = game.over  # or game.won


def main(epsilon_pars, reward_pars, num_episodes=1000, batch_size=32, discount=0.92, amount_frames=2000,
         activation='tanh', mod_n=0):
    # genlogistic(epsilon_pars)
    if TRAIN_TEST:
        train_logic(epsilon_pars, reward_pars, num_episodes=num_episodes, batch_size=batch_size, discount=discount,
                    amount_frames=amount_frames, activation=activation, model_n=mod_n)
    else:
        test(reward_pars)


# if __name__ == '__main__':
#     reward_params = {'game over': -200, 'no hit': -2, 'hit': 1, 'balls_down_positive': True}
#     epsilon_params = {'constant': (False, 0.7), 'a': 0, 'k': 0.75, 'b': 1.5, 'q': 0.5, 'v': 0.55, 'm': 0, 'c': 1}
#     main(epsilon_params, reward_params, num_episodes=1000, batch_size=32, discount=0.92, amount_frames=2000,
#          activation='tanh', mod_n=0)
