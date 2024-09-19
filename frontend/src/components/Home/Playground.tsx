import React, { useEffect, useState } from 'react'
import { useWebsocketContext } from '../../context/WSContext'
import { OrchardSimulation2D, WsSimulationUpdate } from '../../context/apiTypes'
import { CircularProgress } from '@mui/material'
import { Orchard } from './Orchard'

export const Playground: React.FC = () => {
  const api = useWebsocketContext()
  const [width, setWidth] = useState(21)
  const [height, setHeight] = useState(11)
  const [numBots, setNumBots] = useState(2)
  const [orchard, setOrchard] = useState<OrchardSimulation2D | undefined>()
  const [stateUpdateQueue, setStateUpdateQueue] = useState<OrchardSimulation2D[]>([])

  useEffect(() => {
    const callback = (m: WsSimulationUpdate) => setStateUpdateQueue(q => [...q, m.simulation])

    api.register('simulation', callback)
    api.startSimulation(width, height, numBots)

    return () => api.unregister('simulation', callback)
  }, [])

  useEffect(() => {
    const clear = setInterval(() => {
      if (stateUpdateQueue.length > 0) {
        setOrchard(stateUpdateQueue[0])
        setStateUpdateQueue(q => q.slice(1))
      }
    }, 1000)

    return () => clearInterval(clear)
  }, [stateUpdateQueue])

  if (orchard === undefined) {
    return <CircularProgress />
  }

  return (
    <div className='w-full h-full flex flex-col justify-center items-center'>
      <div className='flex flex-col justify-center items-center w-5/6 h-5/6'>
        <div className='flex flex-row w-full justify-between items-center'>
          <div>Time: {orchard.time}</div>
          <div>Apples: {orchard.starting_apple_locations.length - orchard.apples.length}/{orchard.starting_apple_locations.length}</div>
        </div>
        <Orchard data={orchard} />
      </div>
    </div>
  )
}
