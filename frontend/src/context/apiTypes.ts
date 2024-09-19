
export type Coord = [number, number]

export interface OrchardSimulation2D {
  width: number
  height: number
  num_bot: number
  seed: number
  bot_locations: Coord[]
  starting_bot_locations: Coord[]
  trees: Coord[]
  starting_tree_locations: Coord[]
  apples: Coord[]
  starting_apple_locations: Coord[]
  time: number
}

export interface WsError {
  type: 'error'
  message: string
}

export interface WsSimulationUpdate {
  type: 'simulation'
  simulation: OrchardSimulation2D
}

export type WsMessage = WsError | WsSimulationUpdate
export type WsMessageType = WsMessage['type']
export interface WsMessageTypeMap {
  error: WsError
  simulation: WsSimulationUpdate
}
