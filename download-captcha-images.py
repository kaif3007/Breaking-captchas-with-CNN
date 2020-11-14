import argparse
import requests
import time
import os


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,help="path to output directory of images")
ap.add_argument("-n", "--num-images", type=int,default=500, help="# of images to download")
args = vars(ap.parse_args())

#using this url we can download captcha images. refresh this page and we will
#get a new captcha
#captcha image is a 4 digit image conataing 0 to 9
#shape 150X36, a total of 500 images

url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

for i in range(0, args["num_images"]):

	try:
		#try to grab a new captcha image
		#r is a response object containing many attributes
		r = requests.get(url, timeout=60)


		p=f'{args["output"]}/{str(total).zfill(5)}.jpg'
		f = open(p, "wb")
		f.write(r.content)
		f.close()


		print(f'[INFO] downloaded: {p}')
		total += 1

	except:
		print("[INFO] error downloading image...")

		#insert a small sleep to be courteous to the server
	time.sleep(0.1)
