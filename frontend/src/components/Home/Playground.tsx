import React, { useEffect, useState } from 'react'
import { useWebsocketContext } from '../../context/WSContext'
import { OrchardComplex2D, WsSimulationUpdate } from '../../context/apiTypes'
import { CircularProgress } from '@mui/material'
import { OrchardComplex } from './OrchardComplex'

export const Playground: React.FC = () => {
  const api = useWebsocketContext()
  const [width, _setWidth] = useState(1000)
  const [height, _setHeight] = useState(800)
  const [numBots, _setNumBots] = useState(2)
  const [orchard, setOrchard] = useState<OrchardComplex2D | undefined>()
  const [stateUpdateQueue, setStateUpdateQueue] = useState<OrchardComplex2D[]>([])

  useEffect(() => {
    const callback = (m: WsSimulationUpdate) => setStateUpdateQueue(q => [...q, m.simulation])

    api.register('simulation', callback)
    api.startSimulation('complex', width, height, numBots)

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
          <div>Apples: {orchard.starting_apples.length - orchard.apples.length}/{orchard.starting_apples.length}</div>
        </div>
        <OrchardComplex data={orchard} />
      </div>
    </div>
  )
}
