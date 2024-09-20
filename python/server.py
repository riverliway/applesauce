#!/usr/bin/env python

import asyncio
import json
import traceback
from websockets.asyncio.server import serve
from simulation.orchard import OrchardSimulation2D
from simulation.simple_solver import make_simple_decision

SIMULATION_TIMEOUT = 1000

async def designator(websocket) -> None:
  """
  Handles the websocket connection.

  :param: websocket [WebSocketServerProtocol] The websocket connection.
  """
  async for msg in websocket:
    try:
      message = json.loads(json.loads(msg))
      print('~' * 80)
      print(message)

      if not isinstance(message, dict):
        return await websocket.send(error('Expected a dictionary'))

      if not 'type' in message:
        return await websocket.send(error("Expected a 'type' key"))

      if message['type'] == 'hello':
        return await websocket.send(json.dumps({'type': 'hello', 'message': 'Hello, world!'}))
      elif message['type'] == 'start-simulation':
        environment = OrchardSimulation2D(message['params']['width'], message['params']['height'], message['params']['num_bots'])
        await websocket.send(simulation_response(environment))

        while environment.time < SIMULATION_TIMEOUT and len(environment.apples) > 0:
          actions = make_simple_decision(environment)
          environment.step(actions)
          await websocket.send(simulation_response(environment))

    except:
      print(traceback.format_exc())
      await websocket.send(error('An error occurred'))

def error(message: str) -> dict:
  '''
  Creates an error response.

  :param: message [str] The error message.

  :return: dict
  '''
  return json.dumps({'type': 'error', 'message': message})

def simulation_response(simulation: OrchardSimulation2D) -> dict:
  '''
  Creates a response for a simulation.

  :param: simulation The simulation to respond with.

  :return: dict
  '''
  return json.dumps({
    'type': 'simulation',
    'simulation': simulation.to_dict()
  })

async def main():
  print('Starting server on ws://127.0.0.1:4000')
  async with serve(designator, '127.0.0.1', 4000):
    await asyncio.get_running_loop().create_future()

asyncio.run(main())
