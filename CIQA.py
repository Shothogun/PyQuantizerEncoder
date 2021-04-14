import sys
from PIL import Image
import numpy as np
from scipy import integrate

class CIQA:
  def __init__(self):
    self.image_file = sys.argv[1]
    self.N = int(input("Enter the N:\t"))
    self.M = int(input("Enter the M:\t"))
    self.image = Image.open(self.image_file)
    self.pdf = np.zeros(256)
    self.step = 1
    self.b = []
    self.y = []
    self.optimal_delta = []
    self.x = np.arange(0, 256, self.step)
    self.deltas = np.arange(0, 256/self.M, self.step, dtype = np.int32)
    self.deltas[0] = 0.000001     # Delta value must not be zero!

  def sigma_square_compute(self, delta):
    i0 = (np.floor(len(self.x)//2) + 1)
    sigma_square = 0

    for i in range((self.M//2) - 1):
      ia = i0 + (np.round((i-1)*delta)//self.step)
      ib = i0 + (np.round((i)*delta)//self.step)
      v = 2*i

      sigma_square += self.compute_integral(delta,ia,ib,v) 
    
    ia = i0 + round(((self.M//2)-1)*delta//self.step);
    ib = len(self.x) - 1 
    v = self.M
    sigma_square += self.compute_integral(delta,ia,ib,v)

    return sigma_square

  def compute_integral(self,delta,ia,ib,v):
    stepx = self.step
    x = self.x

    k = -(v - 1);

    f = (self.x - ((v - 1)/2)*delta)*self.pdf

    f_integral = self.step * integrate.cumtrapz(f,initial=0)
    # print(ib,ia)
    ib = np.int32(ib)
    ia = np.int32(ia)
    integral = k * (f_integral[ib] - f_integral[ia]);    

    return integral

  def compute_delta(self):
    N = self.N
    heigth, width = self.image.size
    # print(heigth, width)
    n_blocks = (heigth*width)//(N*N)

    # Iterate each NxN block from image
    for block in range(n_blocks):
      # Image NxN Block
      for i in range(N):
        for j in range(N):
          pixel_value = self.image.getpixel((i,j))
          self.pdf[pixel_value] += 1
      
      self.pdf /= N*N 

      error = np.zeros(len(self.deltas))
      for i, delta in np.ndenumerate(self.deltas):
        index = i[0]
        error[index] = self.sigma_square_compute(delta)

      # self.optimal_delta.append(min_error)

def main():
  if len(sys.argv) < 2:
    print("Missing image file path parameter!")
    return
  
  quantizer = CIQA()
  quantizer.compute_delta()

if __name__ == "__main__":
  main()