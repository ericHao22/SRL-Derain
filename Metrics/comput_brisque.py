from brisque import BRISQUE
import os 
import skimage.io

path = 'path/to/images'
folder = os.listdir(path)
total = 0
sum = 0

for i in folder:
    if 'Thumbs.db' in i:
        continue
    obj = BRISQUE(url=False)
    img = skimage.io.imread(os.path.join(path,i)) # H, W, C
    score = obj.score(img)
    
    total+= score
    print(i,": ", score)
    sum+=1

print("Result: ",total/sum)