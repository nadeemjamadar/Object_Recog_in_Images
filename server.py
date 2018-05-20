import object_detection_api
import os
from PIL import Image
#flask files
if __name__ == '__main__':
	import sys
	sys.path.append('lib/')
from flask import Flask, request, Response

app = Flask(__name__)

# for CORS
@app.after_request
def after_request(response):
	response.headers.add('Access-Control-Allow-Origin', '*')
	response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
	response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
	return response


#@app.route('/')
def index():
	return Response('Tensor Flow object detection')


@app.route('/local')
def local():
	return Response(open('./static/local.html').read(), mimetype="text/html")
	
@app.route('/credits')
def credits():
	return Response(open('./static/credits.html').read(), mimetype="text/html")
@app.route('/coco')
def coco():
	return Response(open('./static/coco.html').read(), mimetype="text/html")
@app.route('/imgup')
def imgup():
	return Response(open('./static/imgup.html').read(), mimetype="text/html")
@app.route('/vidup')
def vidup():
	return Response(open('./static/vidup.html').read(), mimetype="text/html")


	
@app.route('/video')
def remote():
	return Response(open('./static/video.html').read(), mimetype="text/html")

@app.route('/')
@app.route('/ui')
def ui():
	return Response(open('./static/oi.html').read(), mimetype="text/html")


@app.route('/test')
def test():
	PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
	TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 6)]

	image = Image.open(TEST_IMAGE_PATHS[0])
	objects = object_detection_api.get_objects(image)

	return objects


@app.route('/image', methods=['POST'])
def image():
	try:
		image_file = request.files['image'].stream  # get the image
		#dobox=1 if request.args.get('withbox','no')=='yes' else 0
		#print('dobox',dobox)
		# Set an image confidence threshold value to limit returned data
		threshold = request.form.get('threshold')
		if threshold is None:
			threshold = 0.5
		else:
			threshold = float(threshold)

		# finally run the image through tensor flow object detection`
		image_object = Image.open(image_file)
		im=image_object.copy().convert('RGB')
		objects = object_detection_api.get_objects(im,threshold)
		#if dobox:
		#	img = object_detection_api.get_objects_image(im)
		#	import base64
		#	from io import BytesIO
		#	buffered = BytesIO()
		#	img.save(buffered, format="JPEG")
		#	img_str = base64.b64encode(buffered.getvalue())
		#	imgt="""<img src="data:image/jpeg;base64,{0}"/>""".format(img_str)
		#	return json.dumps(objects)+"<br><br><br>"+imgt
		return objects

	except Exception as e:
		print('POST /image error: %e' % e)
		return e


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0',port=8001)
