import cv2
import os

counts={}

for (i,imagepath) in enumerate(os.listdir("downloads")):

	print(f'[INFO] processing image {i+1}/{len(imagepath)}')
    
	try:
		path='downloads/'+imagepath
		image=cv2.imread(path)

		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		gray=cv2.copyMakeBorder(gray,8,8,8,8,cv2.BORDER_REPLICATE)
		thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
		
		cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cnts=cnts[0]
		cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:4]

		for c in cnts:
			print(c)
			(x,y,w,h)=cv2.boundingRect(c)
			roi=gray[y-5:y+h+5,x-5:x+w+5]
			cv2.imshow("ROI",cv2.resize(roi,(28,28)))
			key=cv2.waitKey(0)

			if key==ord("'"):
				print("[INFO] Ignoring character... ")
				continue

			key=chr(key).upper()
			dirpath="dataset/"+key
			if not os.path.exists(dirpath):
				os.mkdir(dirpath)

			count=counts.get(key,1)
			cv2.imwrite(dirpath+f"/{str(count).zfill(6)}.png",roi)

			counts[key]=count+1

	except KeyboardInterrupt:
		print(f'[INFO] Manually leaving script...')
		break

	except:
		print(f'[INFO] Skipping Image... ')

	