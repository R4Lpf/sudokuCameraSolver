# Sudoku Image Solver
This is the first step toward an attempt to make a sudoku camera solver taking a lot from Murtaza's guide.

## Example Sudoku Puzzle:
![4](https://user-images.githubusercontent.com/37660959/129197883-af4553d5-88c2-4833-a847-8b4fdb44c9fd.jpg)
## Example Solution: 
![solved sudoku](https://user-images.githubusercontent.com/37660959/129197896-ce2afd29-9e30-4c23-bc2b-f14fc98a9f5e.jpg)

## 1. Train your digit recognition model:
Go to digitTesseract/cnnTraining.py on your prefered IDE and change "epochsVal" to how many eppochs you want to train the model (recommended minimum of 10) and the name of the model when you have to save it (which is the last non-commented line of code), then run the file on spyder if you use spyder or directly from the command promt ("python cnnTraining.py")

## 2. Upload sudoku images and use the model trained:
If you want to solve different sudokus from the ones on the assets directory, just put some jpg files of sudoku in it and change the path name in sudokuMain.py then change the path of the model when the function "initializePredictionModel" is called to the name you gave your model. Then just run the sudokuMain.py file.

## TO DO:
Make it so that points 1 and 2 are as easy as possible in an application or automatically done.

## Installation
Before you run the program install the required libraries if you don't have them installed already:
* Open up your comand promt, cd to the main directory with the requirements.txt file in it, and type:
    * _**-pip install -r requirements.txt**_
           
