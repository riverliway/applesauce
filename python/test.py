#!/usr/bin/env python

import json
from websockets.sync.client import connect

def hello():
  """
  A test file to make sure the server is working correctly.
  """
  with connect("ws://127.0.0.1:4000") as websocket:
    websocket.send(json.dumps({ 'type': 'hello' }))
    message = websocket.recv()
    print(f"Received: {message}")

hello()
