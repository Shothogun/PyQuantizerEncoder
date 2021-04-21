import numpy as np
import sys
import bitstream as bstr
from PIL import Image

class CodeBook:
  def __init__(self):
    self.image_file = sys.argv[1]

    # Codevector size
    self.L = int(input("Enter the L:\t"))

    # Codebook size
    self.M = int(input("Enter the M:\t"))

    self.image_matrix = []

    # Reconstruction vector(Codebook)
    self.y = []

    # Training set
    self.x = []
    self.x_train_size = 512

    # Stop condition on algorithm
    self.epsilon = 0

  def initialize_image_matrix(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    self.image_matrix = np.zeros((heigth,width))

    for i in range(heigth):
      for j in range(width):
        self.image_matrix[i][j] = image_object.getpixel((i,j))

  def initialize_code_book(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    n_blocks = (heigth*width)//(self.L)
    
    pace = n_blocks//self.x_train_size
    block_side = round(self.L**0.5)

    offset_x = 0
    offset_y = 0
    # Initialize train set
    for i in range(self.x_train_size):
      offset_x += (pace*block_side)
      if offset_x == width:
        offset_x = 0
        offset_y += block_side 

      block = self.image_matrix[offset_x:offset_x+block_side] \
                               [offset_y:offset_y+block_side]
      self.x.append(block)

    pace = self.x_train_size//self.M

    # Initialize Codebook
    for i in range(self.M):
      self.y.append(self.x[i*pace])

  def lbg_algorithm(self):
    self.initialize_image_matrix()
    self.initialize_code_book()

    # LBG algorithm

def main():
  if len(sys.argv) < 2:
    print("Missing image file path parameter!")
    return
  
  code_book = CodeBook()
  code_book.lbg_algorithm()


if __name__ == "__main__":
  main()