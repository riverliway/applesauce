
export type Coord = [number, number]

export interface OrchardSimulation2D {
  width: number
  height: number
  num_bot: number
  seed: number
  bot_locations: Coord[]
  trees: Coord[]
  apples: Coord[]
  time: number
}

export interface WsError {
  type: 'error'
  message: string
}

export interface WsSimulationUpdate {
  type: 'simulation'
  data: OrchardSimulation2D
}

export type WsMessage = WsError | WsSimulationUpdate
export type WsMessageType = WsMessage['type']
export interface WsMessageTypeMap {
  error: WsError
  simulation: WsSimulationUpdate
}
