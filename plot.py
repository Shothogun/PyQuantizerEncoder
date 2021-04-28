import matplotlib.pyplot as plt
from PIL import Image


x1=[2,4,8,16]
x2=[2,4,8,16]
x3=[2,4,8,16]
x4=[2,4,8,16]

y1=[123.98,30.10,7.57,4.92]
y2=[304.11,75.00,18.37,4.95]
y3=[646.48,161.66,  40.39,9.71]
y4=[1171.46,295.20,75.21,17.75]


plt.plot(x1,y1,'b.')
plt.plot(x2,y2,'r.')
plt.plot(x3,y3,'g.')
plt.plot(x4,y4,'m.')
plt.show()

reconstructed_image = Image.new('RGB',(512,512))

# Iterate each Pixel
for i in range(512):
  for j in range(512):
    pixel_value = (0,0,255)
    reconstructed_image.putpixel((i,j), pixel_value)
  
reconstructed_image.save("test.bmp")
  