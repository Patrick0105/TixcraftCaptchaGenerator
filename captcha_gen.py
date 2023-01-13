from PIL import Image, ImageDraw, ImageFont
import csv
import random

img_width = 120
img_height = 100


code_chars = 'abcdefghijklmnopqrstuvwxyz'
font = ImageFont.truetype('Training/ReenieBeanie.ttf', 72)
img_quantity = 20
df = ('Training/TrainingSet.csv')

with open(df, 'w', newline ='') as csvFile:
    field = ['file_name','code']
    dictWriter = csv.DictWriter(csvFile, fieldnames = field)
    dictWriter.writeheader()
    
    for k in range(img_quantity):
        img = Image.new('RGB', (img_width, img_height), (51, 156, 216))
        code_str = ''.join(random.sample(code_chars, 4))
        draw = ImageDraw.Draw(img)
        for i, c in enumerate(code_str):
            x = i * 23 + random.randint(0, 10)
            y = random.randint(0, 10)
            draw.text((x, y), c, font=font, fill=(255,255,255))
            
        img_name = f'img{k}.jpg'
        img.save(f'Training/Training_Img/{img_name}')
        dictWriter.writerow({'file_name':img_name,'code':code_str})
csvFile.close()
