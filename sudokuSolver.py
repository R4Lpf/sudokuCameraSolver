import numpy as np

grid = np.array([ [3, 0, 6, 5, 0, 8, 4, 0, 0], 
                  [5, 2, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 8, 7, 0, 0, 0, 0, 3, 1], 
                  [0, 0, 3, 0, 1, 0, 0, 8, 0], 
                  [9, 0, 0, 8, 6, 3, 0, 0, 5], 
                  [0, 5, 0, 0, 9, 0, 6, 0, 0], 
                  [1, 3, 0, 0, 0, 0, 2, 5, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 7, 4], 
                  [0, 0, 5, 2, 0, 6, 3, 0, 0] ])

def myBox(x,y):
    if x < 3 and y < 3:
        return  grid[0:3,0:3]
    elif 3<=x<6 and y < 3:
        return grid[0:3,3:6]
    elif 4<=x<9 and y < 3:
        return grid[0:3,6:9]
    elif x < 3 and 3<=x<6:
        return grid[3:6,0:3]
    elif 3<=x<6 and 3<=x<6:
        return grid[3:6,3:6]
    elif 4<=x<9 and 3<=x<6:
        return grid[3:6,6:9]
    elif x < 3 and 4<=x<9:
        return grid[6:9,0:3]
    elif 3<=x<6 and 4<=x<9:
        return grid[6:9,3:6]
    elif 4<=x<9 and 4<=x<9:
        return grid[6:9,6:9]
    else:
        return False




def valid(x,y,n,grid):
    # n = grid[y][x]
    row = grid[y]
    column = grid[0:9,x]
    box = myBox(x,y)
    if n in row or n in column or n in box:
        return True
    else:
        return False

print(valid(2,1,7,grid))