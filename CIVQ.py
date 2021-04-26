import numpy as np
import sys
import bitstream as bstr
from PIL import Image
import math as mth


def convert2bool(x):
  if x == '1':
    return True

  else:
    return False

def convert_from_bool(x):
  if x:
    return '1'

  else:
    return '0'

class CIVQEncoder:
  def __init__(self):
    self.image_file = sys.argv[1]

    # Codevector size
    self.L = int(input("Enter the L:\t"))

    # Codebook size
    self.M = int(input("Enter the M:\t"))

    self.compressed_file = self.image_file[:-4:]+".CIVQ"

    self.image_matrix = []

    # Reconstruction vector(Codebook)
    self.y = []

    # Training set
    self.x = []
    self.x_train_size = 512

    # Quantization Regions
    self.v = [[] for i in range (self.M)]

    # Stop condition on algorithm
    self.epsilon = 0.005

    # LBG algorithm
    self.D = [0,0]
    self.D_measure = 1000

    self.bitstream = bstr.BitStream()

    self.convert_bin = {
      32:'{0:05b}',
      64:'{0:06b}',
      128:'{0:07b}',
      256:'{0:08b}'
    }

  def initialize_image_matrix(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    self.image_matrix = np.zeros((heigth,width))

    for i in range(heigth):
      for j in range(width):
        self.image_matrix[i][j] = image_object.getpixel((i,j))

    self.image_matrix = np.array(self.image_matrix, dtype=np.int32)

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
      block = np.array(self.image_matrix[offset_x:offset_x+block_side, \
                               offset_y:offset_y+block_side], dtype=np.int32)
      self.x.append(block)

      offset_x += (pace*block_side)
      if offset_x == width:
        offset_x = 0
        offset_y += block_side 

    pace = self.x_train_size//self.M

    # Initialize Codebook
    for i in range(self.M):
      self.y.append(self.x[i*pace])

  def classify_v(self):
    self.v = [[] for i in range (self.M)]
    for x in self.x:
      dmin = float('inf')
      jmin = 0
      for j in range(self.M):
        d = np.sum((self.y[j] - x)**2)
        if d < dmin:
          jmin = j
          dmin = d
      self.v[jmin].append(x)

  def compute_d(self):
    # Computes Dk
    for i in range(self.M):
      for x in self.v[i]:
        self.D[1] += np.sum((x-self.y[i])**2)

    if self.D[1] != 0:  
      self.D_measure = (self.D[1] - self.D[0])/self.D[1]
      self.D[0] = self.D[1]

  def update_codebook(self):
    block_side = round(self.L**0.5)
    for i in range(self.M):
      y = np.zeros((block_side,block_side), dtype=np.float32)
      for x in self.v[i]:
        y += x

      if len(self.v[i]) != 0:
        y /= len(self.v[i])
        self.y[i] = y

  def lbg_algorithm(self):
    self.initialize_image_matrix()
    self.initialize_code_book()

    while self.D_measure >= self.epsilon:
      self.classify_v()
      self.compute_d()

      if self.D_measure >= self.epsilon:
        self.update_codebook()
  
  def set_header(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size

    bin_heigth = '{0:016b}'.format(heigth)
    bin_width = '{0:016b}'.format(width)
    bin_L = '{0:08b}'.format(self.L)
    bin_M = '{0:016b}'.format(self.M)

    self.bitstream.write(list(map(convert2bool, bin_heigth)))
    self.bitstream.write(list(map(convert2bool, bin_width)))
    self.bitstream.write(list(map(convert2bool, bin_L)))
    self.bitstream.write(list(map(convert2bool, bin_M)))

    block_side = round((self.L)**0.5)
 
    # Codebook writing
    for code_vector in self.y:
      for i in range(block_side):
        for j in range(block_side):
          pixel_value = round(code_vector[i][j])
          bin_pixel_value = "{0:08b}".format(pixel_value)
          self.bitstream.write(list(map(convert2bool, bin_pixel_value)))

  def best_code_vector(self,block):
    d = 0
    best_cv_index = 0
    dmin=float('inf')
    block_code=0

    for i in range(self.M):
      d = np.sum((self.y[i] - block)**2)
      if d < dmin:
        best_cv_index = i
        dmin = d
      
    bin_code = self.convert_bin[self.M].format(best_cv_index)
    self.bitstream.write(list(map(convert2bool, bin_code)))

  def quantization(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    n_blocks = (heigth*width)//(self.L)
    
    block_side = round(self.L**0.5)

    offset_x = 0
    offset_y = 0

    self.set_header()

    # Iterate each L size block from image
    for i in range(n_blocks):
      block = np.array(self.image_matrix[offset_x:offset_x+block_side, \
                               offset_y:offset_y+block_side], dtype=np.int32)

      self.best_code_vector(block)

      offset_x += (block_side)
      if offset_x == width:
        offset_x = 0
        offset_y += block_side 

  def write_file(self):
    f = open(self.compressed_file, "wb")
    self.unused_bits = len(self.bitstream)%8

    # Header: 0xE_ , number of unused bits
    first_byte = "1110" + '{0:04b}'.format(self.unused_bits)
    first_byte = int(first_byte,2).to_bytes(1, 'little')
    f.write(first_byte)

    byte = ""
    for _ in range(len(self.bitstream)):
      byte += convert_from_bool(self.bitstream.read(bool,1)[0])

      if len(byte) == 8:
        byte = int(byte,2).to_bytes(1, 'little')
        f.write(byte)
        byte = ""
        
    # Complete last byte
    if len(byte)%8 != 0:
      for _ in range(8 - len(byte)%8):
        byte += '0'

    f.close()

class CIVQDecoder:
  def __init__(self):
    self.bitstream = bstr.BitStream()
    self.compressed_file = ""
    self.L = 0
    self.M = 0
    self.n_blocks = 0
    self.unused_bits = 0
    self.code_book = []
    self.image_codes = []

    self.n_bits = {
      32:5,
      64:6,
      128:7,
      256:8
    }


  def read_bits(self, size):
    bits = ""    
    for _ in range(size):
      bits += convert_from_bool(self.bitstream.read(bool,1)[0])
    return bits

  def read_file(self, compressed_file):
    self.compressed_file = compressed_file
    with open(compressed_file, "rb") as f:
      byte = f.read(1)
      self.bitstream.write(byte)
      while byte:
          byte = f.read(1)
          self.bitstream.write(byte)
    f.close()

    # Read first byte
    bits = self.read_bits(8)
    # Erases first nible '0xE'
    bits = bits[-4::]
    self.unused_bits = int(bits,2)

    # Read image heigth
    bits = self.read_bits(16)
    self.heigth = int(bits,2)

    # Read image width
    bits = self.read_bits(16)
    self.width = int(bits,2)

    # Read L
    bits = self.read_bits(8)
    self.L = int(bits,2)

    # Read M
    bits = self.read_bits(16)
    self.M = int(bits,2)

    block_side = round((self.L)**0.5)

    # Codebook
    for i in range(self.M):
      line = 0
      column = 0
      y = np.zeros((block_side,block_side), dtype=np.int32)
      for j in range(self.L):
        bits = self.read_bits(8)
        y[line][column] = int(bits,2)
        column += 1
        if column == block_side:
          column = 0
          line += 1
      self.code_book.append(y)

    # Content    
    code=""
    for _ in range(len(self.bitstream) - self.unused_bits):
      code += convert_from_bool(self.bitstream.read(bool,1)[0])

      if len(code) == self.n_bits[self.M]:
        code = int(code,2)
        self.image_codes.append(code)
        code = ""

    return

  def decode(self):
    offset_x = 0
    offset_y = 0
    reconstructed_image = Image.new('P',(self.width,self.heigth))

    block_side = round((self.L)**0.5)
    n_blocks = (self.heigth*self.width)//(self.L)

    # Iterate each L block from image
    for nth_block in range(n_blocks):
      code_vector = self.code_book[self.image_codes[nth_block]]
      
      # Image NxN Block
      for i in range(block_side):
        for j in range(block_side):
          pixel_value = int(code_vector[i][j])
          reconstructed_image.putpixel((i+offset_x,j+offset_y), pixel_value)

      offset_x += block_side
      if offset_x == self.width:
        offset_x = 0
        offset_y += block_side
      
    reconstructed_image.save(self.compressed_file[:-5:]+"_reconstructed.bmp")
  
    return

  def MSE(self, original_file, decompressed_file):
    image_test = Image.open(decompressed_file)
    image_reference = Image.open(original_file)

    mse = 0

    height, width = image_test.size

    for i in range(height):
      for j in range(width):
        I = image_test.getpixel((i,j))
        R = image_reference.getpixel((i,j))
        mse += (I-R)**2

    mse /= (height*width)

    return mse

  def PSNR(self, mse):
    psnr =  255*255
    psnr /= mse
    psnr = 10*mth.log(psnr,10)
    return psnr

def main():
  if len(sys.argv) < 2:
    print("Missing image file path parameter!")
    return
  
  encoder = CIVQEncoder()
  encoder.lbg_algorithm()
  encoder.quantization()
  encoder.write_file()

  decoder = CIVQDecoder()
  decoder.read_file(encoder.compressed_file)
  decoder.decode()

  original_file = encoder.image_file
  decompressed_file = decoder.compressed_file[:-5:]+"_reconstructed.bmp"

  mse = decoder.MSE(original_file, decompressed_file)
  print("MSE  value:", mse)
  psnr = decoder.PSNR(mse)
  print("PSNR value:", psnr)

if __name__ == "__main__":
  main()