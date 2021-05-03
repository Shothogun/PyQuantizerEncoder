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

class CIMapEncoder:
  def __init__(self):
    self.image_file = sys.argv[2]

    # Codevector size
    self.L = 3

    # Codebook size
    self.M = int(sys.argv[1]) #int(input("Enter the M:\t"))

    self.compressed_file = self.image_file[:-4:]+".CIMap"

    self.image_matrix = []

    # Reconstruction vector(Codebook)
    self.y = []

    # Training set
    self.x = []
    self.x_train_size = 2048

    # Quantization Regions
    self.v = [[] for i in range (self.M)]

    # Stop condition on algorithm
    self.epsilon = 0.005

    # LBG algorithm
    self.D = [0,0]
    self.D_measure = 1000

    self.bitstream = bstr.BitStream()

    self.convert_bin = {
      16:'{0:04b}', 
      32:'{0:05b}',
      64:'{0:06b}',
      128:'{0:07b}',
      256:'{0:08b}'
    }

  def initialize_image_matrix(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    self.image_matrix = np.zeros((3,heigth,width), dtype="object")

    for i in range(heigth):
      for j in range(width):
        self.image_matrix[:,i,j] = list(image_object.getpixel((i,j)))

    self.image_matrix = np.array(self.image_matrix, dtype="object")

  def initialize_code_book(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    pace = (heigth*width)//self.x_train_size
    
    line = 0
    column = 0

    # Initialize train set
    for i in range(self.x_train_size):
      pixel_value = np.array(self.image_matrix[:, line, column],dtype="object")

      self.x.append(pixel_value)

      column += (pace)
      if column >= width:
        column = column - width
        line += 1

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
    for i in range(self.M):
      y = np.zeros(3, dtype="object")
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
    bin_M = '{0:016b}'.format(self.M)

    self.bitstream.write(list(map(convert2bool, bin_heigth)))
    self.bitstream.write(list(map(convert2bool, bin_width)))
    self.bitstream.write(list(map(convert2bool, bin_M)))

    # Codebook writing
    for code_vector in self.y:
      for i in range(self.L):
          pixel_value = round(code_vector[i])
          bin_pixel_value = "{0:08b}".format(pixel_value)
          self.bitstream.write(list(map(convert2bool, bin_pixel_value)))

  def best_code_vector(self,code_vect):
    d = 0
    best_cv_index = 0
    dmin=float('inf')

    for i in range(self.M):
      d = np.sum((self.y[i] - code_vect)**2)
      if d < dmin:
        best_cv_index = i
        dmin = d
      
    bin_code = self.convert_bin[self.M].format(best_cv_index)
    self.bitstream.write(list(map(convert2bool, bin_code)))

  def quantization(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size

    n_pixels = heigth*width

    column = 0
    line = 0

    self.set_header()

    # Iterate each L size block from image
    for line in range(heigth):
      for column in range(width):          
        pixel_value = np.array(self.image_matrix[:, line, column],dtype="object")
        self.best_code_vector(pixel_value)

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

class CIMapDecoder:
  def __init__(self):
    self.bitstream = bstr.BitStream()
    self.compressed_file = ""
    self.L = 3
    self.M = 0
    self.n_blocks = 0
    self.unused_bits = 0
    self.code_book = []
    self.image_codes = []

    self.n_bits = {
      16:4,
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

    # Read M
    bits = self.read_bits(16)
    self.M = int(bits,2)

    block_side = round((self.L)**0.5)

    # Codebook
    for i in range(self.M):
      y = np.zeros(3, dtype="object")
      for j in range(self.L):
        bits = self.read_bits(8)
        y[j] = int(bits,2)
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
    reconstructed_image = Image.new('RGB',(self.width,self.heigth))

    # Iterate each Pixel
    for i in range(self.heigth):
      for j in range(self.width):
        code_vector = self.code_book[self.image_codes[i*self.width+j]]
        pixel_value = (code_vector[0],code_vector[1],code_vector[2])
        reconstructed_image.putpixel((i,j), pixel_value)
      
    reconstructed_image.save(self.compressed_file[:-6:]+"_reconstructed.bmp")
  
    return

  def MSE(self, original_file, decompressed_file):
    image_test = Image.open(decompressed_file)
    image_reference = Image.open(original_file)
  
    mse = 0

    height, width = image_test.size
    for i in range(height):
      for j in range(width):
        for dim in range(3):
          I = image_test.getpixel((i,j))[dim]
          R = image_reference.getpixel((i,j))[dim]
          mse += (I-R)**2

    mse /= (3*height*width)

    return mse

  def PSNR(self, mse):
    psnr =  255*255
    psnr /= mse
    psnr = 10*mth.log(psnr,10)
    return psnr

def main():
  if len(sys.argv) < 3:
    print("Missing image file path parameter!")
    return
  
  encoder = CIMapEncoder()
  encoder.lbg_algorithm()
  encoder.quantization()
  encoder.write_file()

  decoder = CIMapDecoder()
  decoder.read_file(encoder.compressed_file)
  decoder.decode()

  original_file = encoder.image_file
  decompressed_file = decoder.compressed_file[:-6:]+"_reconstructed.bmp"

  mse = decoder.MSE(original_file, decompressed_file)
  print("MSE  value:", mse)
  psnr = decoder.PSNR(mse)
  print("PSNR value:", psnr)


if __name__ == "__main__":
  main()