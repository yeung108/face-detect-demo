#!python3

import os
from flask import Flask, request, Response
from flask_restful import Resource, Api
import pika
from threading import Thread
import json
import base64
import time
import platform
from camera import NormalCamera

liveliness = bool(os.environ["LIVELINESS"] == 'true') if "LIVELINESS" in os.environ else False
ageWeightedAverage = bool(os.environ["AGE_WEIGHTED_AVERAGE"] == 'true') if "AGE_WEIGHTED_AVERAGE" in os.environ else False
testMode = bool(os.environ["TEST_MODE"] == 'true') if "TEST_MODE" in os.environ else False
showUrl = bool(os.environ["SHOW_URL"] == 'true') if "SHOW_URL" in os.environ else False
savePhoto = bool(os.environ["SAVE_PHOTO"] == 'true') if "SAVE_PHOTO" in os.environ else False
detectTimeout = float(os.environ["DETECT_TIMEOUT"]) if "DETECT_TIMEOUT" in os.environ else 50
rotation = int(os.environ["ROTATION"]) if "ROTATION" in os.environ else -1
rabbitmqUsername = str(os.environ["RABBITMQ_USERNAME"]) if "RABBITMQ_USERNAME" in os.environ else ""
rabbitmqPassword = str(os.environ["RABBITMQ_PASSWORD"]) if "RABBITMQ_PASSWORD" in os.environ else ""

print("LIVELINESS: "+str(liveliness))
print("AGE_WEIGHTED_AVERAGE: "+str(ageWeightedAverage))
print("TEST_MODE: "+str(testMode))
print("DETECT_TIMEOUT: "+str(detectTimeout))
print("SHOW_URL: "+str(showUrl))
print("SAVE_PHOTO: "+str(savePhoto))
print("ROTATION: "+str(rotation))

rsCamera = NormalCamera.get_instance(testMode, rotation)

app = Flask(__name__)
api = Api(app)
credentials = pika.PlainCredentials(rabbitmqUsername, rabbitmqPassword)
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost', port=5672, virtual_host="/", credentials=credentials))
channel = connection.channel()
args = {}
args["x-message-ttl"] = 5000
channel.queue_declare(queue='rpc_face_camera_queue', arguments=args)

def callback(ch, method, properties, body):
	# Load the JSON to a Python list & dump it back out as formatted JSON
	data = json.loads(body.decode('utf8').replace("'", '"'))
	payload = base64.b64decode(data['payload']).decode('utf8').replace("'", '"')
	print(data['event'])
	print(properties)
	if (data['event'] == 'START_DETECTING_FACE'):
		print('in START_DETECTING_FACE')
		#if request.args.get('full_image') is not None else False
		rsCamera.__startDetect__(liveliness, ageWeightedAverage, detectTimeout, savePhoto)
		ch.basic_ack(delivery_tag = method.delivery_tag)
		ch.basic_publish(exchange='', 
		routing_key=properties.reply_to, 
		properties=pika.BasicProperties(
			correlation_id = properties.correlation_id,
			reply_to=properties.reply_to,
			content_type="application/json"
		), 
		body=json.dumps({
			'event': data['event'],
			'payload': ""
			})
		)
	elif (data['event'] == 'END_DETECTING_FACE'):
		print("in END_DETECTING_FACE")
		photo, gender, age = rsCamera.__endDetect__()
		if (photo and gender and age):
			ch.basic_ack(delivery_tag = method.delivery_tag)
			ch.basic_publish(exchange='', 
			routing_key=properties.reply_to, 
			properties=pika.BasicProperties(
				correlation_id = properties.correlation_id,
				reply_to=properties.reply_to,
				content_type="application/json"
			), 
			body=json.dumps({
				'event': data['event'],
				'payload': str(base64.urlsafe_b64encode(json.dumps({
					'photo': photo, 
					'gender': gender, 
					'age': age, 
					'data': json.dumps(payload)
					}).encode('utf-8')), "utf-8")
				})
			)
		else:
			ch.basic_ack(delivery_tag = method.delivery_tag)
channel.basic_consume(queue='rpc_face_camera_queue', auto_ack=False, on_message_callback=callback)

def gen():
	while True:
		frame = rsCamera.get_jpeg_frame()
		if not frame:
			continue
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def genFace():
	while True:
		frame = rsCamera.video_feed(liveliness, ageWeightedAverage)
		if not frame:
			continue
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_depth():
	while True:
		frame = rsCamera.get_jpeg_depth_frame()
		if not frame:
			continue
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/detect')
def detect():
	#if request.args.get('full_image') is not None else False
	detected = rsCamera.detect(True, True, detectTimeout)
	if not detected:
		return Response(status=408)
	return Response(detected, mimetype='image/jpg')

@app.route('/video_feed')
def video_feed():
	rsCamera.__initVar__()
	return Response(genFace(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_feed')
def depth_feed():
	return Response(gen_depth(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	port = int(os.environ["PORT"]) if "PORT" in os.environ else 5000
	if (showUrl):
		app.run(host='127.0.0.1', port=port, debug=True, use_reloader=False, threaded=True)
	else:
		thread = Thread(channel.start_consuming())
		thread.start()
