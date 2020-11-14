import cv2
import os

counts={}

#can access pixels as numpy arrays
#image channels are in (b,g,r) format

for (i,imagepath) in enumerate(os.listdir("downloads")):

	print(f'[INFO] processing image {i+1}/{len(imagepath)}')
    
	try:
		path='downloads/'+imagepath
		image=cv2.imread(path)
		
        #convert to grayscale.Just 1 channels
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                
		#increase borders of image by 8 pixels on each side.
		gray=cv2.copyMakeBorder(gray,8,8,8,8,cv2.BORDER_REPLICATE)


		#thresholding is simply converting grayscale image to binary 0 or 255
		#obtain thresholded image.set all pixels greater than 0 to 255 ie white. and remaining black but do its inverse
		#as method in inverse
		#it returns 2 parameters T,thresh
		#T is threshold value,thresh is image
		#In manuaal threshold we need to supply threshold value,whereas is adaptive threshold we don't
		thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
		
	
		#find cotours in image
		#destroys original image so we pass copy, RETR_EXTERNAL because we only want external contours
		#find contours works internally by applying edge-detection algorithms
		#chain-approx-simple only stores boundary points, as no need to store entire points
		cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cnts=cnts[0]
		
		#sort by decreasing area and pick 4 contours
		cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:4]

		for c in cnts:
			#approx bounding rectangles for contours
			(x,y,w,h)=cv2.boundingRect(c)
			
			#pick regin of interset
			roi=gray[y-5:y+h+5,x-5:x+w+5]
			cv2.imshow("ROI",cv2.resize(roi,(28,28)))
			key=cv2.waitKey(0)
                        
			#wait for key by user after that save image in desored directory.
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

	
