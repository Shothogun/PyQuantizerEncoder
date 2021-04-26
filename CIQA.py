import sys
import bitstream as bstr
from PIL import Image
import numpy as np
from scipy import integrate
import math as mth

def convert_from_bool(x):
  if x:
    return '1'

  else:
    return '0'


class CIQAEncoder:
  def __init__(self):
    self.image_file = sys.argv[1]
    self.compressed_file = self.image_file[:-4:]+".CIQA"

    self.N = int(input("Enter the N:\t"))
    self.M = int(input("Enter the M:\t"))
    self.image = Image.open(self.image_file)

    heigth, width = self.image.size
    self.n_blocks = (heigth*width)//(self.N*self.N)

    self.block = np.zeros([self.N,self.N], dtype = np.int32)
    self.b = []
    self.y = []
    self.min = np.zeros(self.n_blocks, dtype = np.int32)
    self.max = np.zeros(self.n_blocks, dtype = np.int32)
    self.optimal_delta = np.zeros(self.n_blocks, dtype = np.int32)
    self.bitstream = bstr.BitStream()
    self.unused_bits = 0

    self.convert_bin = {
      2:'{0:01b}',
      4:'{0:02b}',
      8:'{0:03b}',
      16:'{0:04b}'
    }

  def compute_delta(self):
    offset_x = 0
    offset_y = 0

    heigth, width = self.image.size

    # Iterate each NxN block from image
    for i_block in range(self.n_blocks):
      # Image NxN Block
      for i in range(self.N):
        for j in range(self.N):
          self.block[i][j] = np.int32(self.image.getpixel((i+offset_x,j+offset_y)))

      offset_x += self.N

      if offset_x == width:
        offset_x = 0
        offset_y += self.N

      self.min[i_block] = np.min(self.block)
      # print(np.min(self.block), np.max(self.block))
      self.max[i_block] = np.max(self.block)
  
  def quantize(self):
    convert2bool = lambda x: True if x == '1' else False
    offset_x = 0
    offset_y = 0

    heigth, width = self.image.size

    # Header writing: N,M, blocks number, (min, max), content
    bin_heigth = '{0:016b}'.format(heigth)
    bin_width = '{0:016b}'.format(width)
    bin_N = '{0:08b}'.format(self.N)
    bin_M = '{0:08b}'.format(self.M)
    bin_n_blocks = '{0:016b}'.format(self.n_blocks)
    self.bitstream.write(list(map(convert2bool, bin_heigth)))
    self.bitstream.write(list(map(convert2bool, bin_width)))
    self.bitstream.write(list(map(convert2bool, bin_N)))
    self.bitstream.write(list(map(convert2bool, bin_M)))
    self.bitstream.write(list(map(convert2bool, bin_n_blocks)))

    for i in range(self.n_blocks):
      bin_min_value = '{0:08b}'.format(self.min[i])
      self.bitstream.write(list(map(convert2bool, bin_min_value)))
      bin_max_value = '{0:08b}'.format(self.max[i])
      self.bitstream.write(list(map(convert2bool, bin_max_value)))

    # Iterate each NxN block from image
    for i_block in range(self.n_blocks):
      min_value = self.min[i_block]
      max_value = self.max[i_block]
      delta = round((max_value - min_value)/self.M)

      y = np.arange(self.M)*delta + min_value

      # Image NxN Block
      for i in range(self.N):
        for j in range(self.N):
          pixel_value = np.int32(self.image.getpixel((i+offset_x,j+offset_y)))
          code = np.where(y<=pixel_value)[0][-1]
          bin_code = self.convert_bin[self.M].format(code)
          self.bitstream.write(list(map(convert2bool, bin_code)))

      offset_x += self.N
      if offset_x == width:
        offset_x = 0
        offset_y += self.N

    return
  
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

class CIQADecoder:
  def __init__(self):
    self.bitstream = bstr.BitStream()
    self.compressed_file = ""
    self.N = 0
    self.M = 0
    self.n_blocks = 0
    self.unused_bits = 0
    self.image_codes = []
    self.min = []
    self.max = []

    self.n_bits = {
      2:1,
      4:2,
      8:3,
      16:4
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

    # Read N
    bits = self.read_bits(8)
    self.N = int(bits,2)

    # Read M
    bits = self.read_bits(8)
    self.M = int(bits,2)

    # Read N blocks
    bits = self.read_bits(16)
    self.n_blocks = int(bits,2)

    for i in range(self.n_blocks):
      bits = self.read_bits(8)
      self.min.append(int(bits,2))
      bits = self.read_bits(8)
      self.max.append(int(bits,2))

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
    n_pixel = 0
    reconstructed_image = Image.new('P',(self.width,self.heigth))

    # Iterate each NxN block from image
    for i_block in range(self.n_blocks):
      min_value = self.min[i_block]
      max_value = self.max[i_block]
      delta = round((max_value - min_value)/self.M)

      y = np.arange(self.M)*delta + min_value

      # Image NxN Block
      for i in range(self.N):
        for j in range(self.N):
          pixel_value = int(y[self.image_codes[n_pixel]])
          reconstructed_image.putpixel((i+offset_x,j+offset_y), pixel_value)
          n_pixel +=1

      offset_x += self.N
      if offset_x == self.width:
        offset_x = 0
        offset_y += self.N

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
    psnr = 255*255
    psnr /= mse
    psnr = 10*mth.log(psnr,10)
    return psnr

def main():
  if len(sys.argv) < 2:
    print("Missing image file path parameter!")
    return
  
  quantizer = CIQAEncoder()
  quantizer.compute_delta()
  quantizer.quantize()
  quantizer.write_file()

  decoder = CIQADecoder()
  decoder.read_file(quantizer.compressed_file)
  decoder.decode()

  original_file = quantizer.image_file
  decompressed_file = decoder.compressed_file[:-5:]+"_reconstructed.bmp"

  mse = decoder.MSE(original_file, decompressed_file)
  print("MSE  value:", mse)
  psnr = decoder.PSNR(mse)
  print("PSNR value:", psnr)

if __name__ == "__main__":
  main()