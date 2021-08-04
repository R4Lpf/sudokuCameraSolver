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



def valid(col,row,num,grid):
 
    for x in range(9):
        if grid[row][x] == num:
            return False

    for x in range(9):
        if grid[x][col] == num:
            return False
  
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True




def solve(x, y, grid):
    if y == 8 and x == 9:
        return True
    if x == 9:
        y += 1
        x = 0
    if grid[y][x]>0:
        return solve(x+1,y,grid)
    for i in range(1,10):
        if valid(x,y,i,grid):
            grid[y][x] = i
            if solve(x+1,y,grid):
                return True
        grid[y][x] = 0
    return False

print(grid)
solve(0,0,grid)
print (grid)
















