import time, random
from objs.constants import *
from objs.bubble_file import *
from objs.grid_file import *
from objs.shooter_file import *
from objs.game_objects import *
import pygame as pg
import pprint
pg.init()


def main():

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
	cheat_manager = CheatManager(grid_manager, gun)
	grid_manager.view(gun, game)	# Check collision with bullet and update grid as needed	
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
	mouse_pos = (DISP_W/2, DISP_H/2)
	
	# pretty self-explanatory
	while not game.over:		

		# quit when you press the x
		for event in pg.event.get():
			if event.type == pg.QUIT:
				pg.quit()
				quit()

			# get mouse position
			if event.type == pg.MOUSEMOTION: mouse_pos = pg.mouse.get_pos()
				
			# if you click, fire a bullet
			if event.type == pg.MOUSEBUTTONDOWN: gun.fire()
			
			if event.type == pg.KEYDOWN:
				cheat_manager.view(event) # if a key is pressed, the cheat manager should know about it

				# Ctrl+C to quit
				if event.key == pg.K_c and pg.key.get_mods() & pg.KMOD_CTRL:
					pg.quit()
					quit()

		
		background.draw()				# Draw BG first		

		grid_manager.view(gun, game)	# Check collision with bullet and update grid as needed		

		#ACA IRIA LA PARTE DE LA RED NEURONAL
		#print(grid_manager.grid[0][0].__dict__)

		gun.rotate(mouse_pos)			# Rotate the gun if the mouse is moved		
		gun.draw_bullets()				# Draw and update bullet and reloads	

		game.drawScore()				# draw score

		pg.display.update()		
		clock.tick(60)					# 60 FPS

		#print(gun.angle)
	game.gameOverScreen(grid_manager, background)

	return

if __name__ == '__main__': 
	while True: main()