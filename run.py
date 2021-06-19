from flask import Flask
import json
import flask
from flask import request, jsonify
from process import predict

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    message = ''
    if 'message' in request.args:
        message = request.args['message']
    return jsonify(predict(message))

app.run()