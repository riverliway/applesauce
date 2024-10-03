
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

export interface Tree {
  x: number
  y: number
  diameter: number
  fertility: number
}

export interface Bot {
  x: number
  y: number
  job: 'picker' | 'pusher'
  holding: number | null
  diameter: number
  orientation: number
}

export interface Basket {
  x: number
  y: number
  diameter: number
}

export interface Apple {
  x: number
  y: number
  diameter: number
  held: boolean
  collected: boolean
}

export interface OrchardComplex2D {
  width: number
  height: number
  num_picker_bots: number
  num_pusher_bots: number
  seed: number
  time: number
  trees: Tree[]
  bots: Bot[]
  baskets: Basket[]
  apples: Apple[]
  starting_bots: Bot[]
  starting_baskets: Basket[]
  starting_trees: Tree[]
  starting_apples: Apple[]
  TICK_SPEED: number
}

export interface WsError {
  type: 'error'
  message: string
}

export interface WsSimulationUpdate {
  type: 'simulation'
  simulation: OrchardComplex2D
}

export interface WsPathUpdate {
  type: 'path'
  paths: Coord[]
  time: number
}

export type WsMessage = WsError | WsSimulationUpdate | WsPathUpdate
export type WsMessageType = WsMessage['type']
export interface WsMessageTypeMap {
  error: WsError
  simulation: WsSimulationUpdate
  path: WsPathUpdate
}
