import numpy as np
import sys
import bitstream as bstr
from PIL import Image
import math as mth

def convert_from_bool(x):
  if x:
    return '1'

  else:
    return '0'

class FlyStgEncoder:
  def __init__(self):
    self.image_file = sys.argv[1]
    self.compressed_file = self.image_file[:-4:]+".FSA"

    self.M = int(input("Enter the M:\t"))
    self.dithering_flag = int(input("Use dithering(1 or 0)?\t"))
    self.image = Image.open(self.image_file)

    heigth, width = self.image.size
    
    self.bitstream = bstr.BitStream()
    self.unused_bits = 0

    self.convert_bin = {
      8:'{0:03b}',
      16:'{0:04b}'
    }

  def initialize_image_matrix(self):
    image_object = Image.open(self.image_file)
    heigth, width = image_object.size
    self.image_matrix = np.zeros((3,heigth,width), dtype="object")

    for i in range(heigth):
      for j in range(width):
        self.image_matrix[:,i,j] = list(image_object.getpixel((i,j)))

    self.image_matrix = np.array(self.image_matrix, dtype="object")

  def dithering(self):
    self.initialize_image_matrix()
    heigth, width = self.image.size

    delta = round(256/self.M)
    y = np.arange(self.M)*delta

    for i in range(heigth-1):
      for j in range(width-1):
        oldpixel = self.image_matrix[:,i,j].copy()
        newpixel = np.array([y[np.where(y<=oldpixel[0])[0][-1]],
                    y[np.where(y<=oldpixel[1])[0][-1]],
                    y[np.where(y<=oldpixel[2])[0][-1]]], dtype="object")
        self.image_matrix[:,i,j] = newpixel.copy()
        if self.dithering_flag:
          quant_error = np.array(oldpixel - newpixel, dtype="object")
          self.image_matrix[:, i,j + 1] = self.image_matrix[:, i,j + 1] + quant_error * (7 / 16)
          self.image_matrix[:, i + 1,j - 1] = self.image_matrix[:, i + 1,j - 1] + quant_error * (3 / 16)
          self.image_matrix[:, i + 1,j    ] = self.image_matrix[:, i + 1,j] + quant_error * (5 / 16)
          self.image_matrix[:, i + 1,j + 1] = self.image_matrix[:, i + 1,j + 1] + quant_error * (1 / 16)

  def quantize(self):
    convert2bool = lambda x: True if x == '1' else False
    heigth, width = self.image.size

    # Header writing: heigth, width, M
    bin_heigth = '{0:016b}'.format(heigth)
    bin_width = '{0:016b}'.format(width)
    bin_M = '{0:08b}'.format(self.M)
    self.bitstream.write(list(map(convert2bool, bin_heigth)))
    self.bitstream.write(list(map(convert2bool, bin_width)))
    self.bitstream.write(list(map(convert2bool, bin_M)))

    delta = round(256/self.M)
    y = np.arange(self.M)*delta

    # Image encoded
    for i in range(heigth):
      for j in range(width):
        for dim in range(3):
          pixel_value = self.image_matrix[dim,i,j]
          code = np.where(y<=pixel_value)[0][-1]
          bin_code = self.convert_bin[self.M].format(code)
          self.bitstream.write(list(map(convert2bool, bin_code)))

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

class FlyStgDecoder:
  def __init__(self):
    self.bitstream = bstr.BitStream()
    self.compressed_file = ""
    self.M = 0
    self.unused_bits = 0
    self.image_codes = []

    self.n_bits = {
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

    # Read M
    bits = self.read_bits(8)
    self.M = int(bits,2)

    code=""
    for _ in range(len(self.bitstream) - self.unused_bits):
      code += convert_from_bool(self.bitstream.read(bool,1)[0])

      if len(code) == self.n_bits[self.M]:
        code = int(code,2)
        self.image_codes.append(code)
        code = ""

    return

  def decode(self):
    reconstructed_image = Image.new('RGB',(self.width,self.heigth))

    delta = round(256/self.M)

    y = np.arange(self.M)*delta

    pixel_value=[0,0,0]
    n_pixel = 0

    # Image NxN Block
    for i in range(self.heigth):
      for j in range(self.width):
        for dim in range(3):
          pixel_value[dim] = int(y[self.image_codes[n_pixel]])
          reconstructed_image.putpixel((i,j), tuple(pixel_value))
          n_pixel +=1

    reconstructed_image.save(self.compressed_file[:-4:]+"_reconstructed.bmp")

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
  if len(sys.argv) < 2:
    print("Missing image file path parameter!")
    return
  
  quantizer = FlyStgEncoder()
  quantizer.dithering()
  quantizer.quantize()
  quantizer.write_file()

  decoder = FlyStgDecoder()
  decoder.read_file(quantizer.compressed_file)
  decoder.decode()

  original_file = quantizer.image_file
  decompressed_file = decoder.compressed_file[:-4:]+"_reconstructed.bmp"

  mse = decoder.MSE(original_file, decompressed_file)
  print("MSE  value:", mse)
  psnr = decoder.PSNR(mse)
  print("PSNR value:", psnr)

if __name__ == "__main__":
  main()