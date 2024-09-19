#!/usr/bin/env python

import asyncio
from websockets.asyncio.server import serve
from simulation.orchard import OrchardSimulation2D

async def designator(websocket) -> None:
  """
  Handles the websocket connection.

  :param: websocket [WebSocketServerProtocol] The websocket connection.
  """
  async for message in websocket:
    try:
      print('~' * 80)
      print(message)

      if not isinstance(message, dict):
        await websocket.send(error('Expected a dictionary'))

      if not 'type' in message:
        await websocket.send(error("Expected a 'type' key"))

      if message.type == 'hello':
        await websocket.send('Hello world!')
      elif message.type == 'start-simulation':
        await websocket.send(simulation_response(OrchardSimulation2D(20, 10, 1)))

    except:
      await websocket.send(error('An error occurred'))

def error(message: str) -> dict:
  '''
  Creates an error response.

  :param: message [str] The error message.

  :return: dict
  '''
  return {'type': 'error', 'message': message}

def simulation_response(simulation: OrchardSimulation2D) -> dict:
  '''
  Creates a response for a simulation.

  :param: simulation The simulation to respond with.

  :return: dict
  '''
  return {
    'type': 'simulation',
    'simulation': simulation.to_dict()
  }

async def main():
  async with serve(designator, '127.0.0.1', 4000):
    await asyncio.get_running_loop().create_future()

asyncio.run(main())
