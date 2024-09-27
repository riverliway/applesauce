
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

export interface OrchardComplex2D {
  width: number
  height: number
  num_picker_bots: number
  num_pusher_bots: number
  seed: number
  time: number
  trees: Array<{
    x: number
    y: number
    diameter: number
    fertility: number
  }>
  bots: Array<{
    x: number
    y: number
    job: 'picker' | 'pusher'
    holding: boolean
    diameter: number
  }>
  apples: Array<{
    x: number
    y: number
    diameter: number
  }>
  starting_bots: Array<{
    x: number
    y: number
    job: 'picker' | 'pusher'
    diameter: number
  }>
  starting_trees: Array<{
    x: number
    y: number
    diameter: number
    fertility: number
  }>
  starting_apples: Array<{
    x: number
    y: number
    diameter: number
  }>
}

export interface WsError {
  type: 'error'
  message: string
}

export interface WsSimulationUpdate {
  type: 'simulation'
  simulation: OrchardComplex2D
}

export type WsMessage = WsError | WsSimulationUpdate
export type WsMessageType = WsMessage['type']
export interface WsMessageTypeMap {
  error: WsError
  simulation: WsSimulationUpdate
}
