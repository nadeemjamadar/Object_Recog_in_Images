# Object Recognition in Images
-Final Year GTU Project

A flask based web-app which labels the objects that it can identify from either of these:

--> Camera Video stream

--> Static upload-able images

--> Whole video [remote or upload-able]

### API Doc

- Send a post request to:
[http://localhost:5001/images](http://localhost:5001/image)
With a file being uploaded as "image" parameter
Easier if form type is set to multipart/form-data

--> Sample cURL request to api:
`curl "http://localhost:8001/image" -X POST -H "Content-Type: multipart/form-data" -H "User-Agent: Mozilla/5.0" -F "image=@IMG_20180412_115639.jpg" `

-- Minimal cURL Request looks like:
`curl "http://localhost:8001/image" -X POST -F "image=@IMG_20180412_115639.jpg" `

- Expected response is a json array with always 1st element as API info
and rest are json describing each object with following attributes:
-- name as "Object"
-- class_name as object name (eg "person")
-- score as some float (eg 0.9807044863700867) telling confidence of prediction of that object
--x,y,height,width are co-ordinates to draw a bounding box (scaled to 0-1) on given image

--> Sample JSON response with two objects person and bottle in image:

[{"name": "Object Recognition in Images REST API", "version": "0.0.1", "numObjects": 2.0, "threshold": 0.4}, {"name": "Object", "class_name": "person", "score": 0.7700180411338806, "y": 0.17950987815856934, "x": 0.0, "height": 1.0, "width": 0.8535020351409912}, {"name": "Object", "class_name": "bottle", "score": 0.7336319088935852, "y": 0.45810097455978394, "x": 0.21297135949134827, "height": 0.8313612341880798, "width": 0.3277725875377655}]

### Starting Web-App:
- (Optional) Create python environment
- pip install tensorflow-gpu (also, ensure tf works, after installing CUDA if nvidia)
- pip install flask
- python server.py (u can change flask port number from server.py file last line)

### Resources Used:
- COCO dataset
- Google Inception Model / MobileNet model
- Adapted api call structure from tfObjWebrtc
