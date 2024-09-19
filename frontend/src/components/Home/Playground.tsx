import React, { useEffect, useState } from 'react'
import { useWebsocketContext } from '../../context/WSContext'
import { WsSimulationUpdate } from '../../context/apiTypes'

export const Playground: React.FC = () => {
  const api = useWebsocketContext()
  const [width, setWidth] = useState(20)
  const [height, setHeight] = useState(10)
  const [numBots, setNumBots] = useState(1)

  useEffect(() => {
    const callback = (m: WsSimulationUpdate) => console.log(m)

    api.register('simulation', callback)
    api.startSimulation(width, height, numBots)

    return () => api.unregister('simulation', callback)
  }, [])

  return (
    <div>
      <h1 className='text-3xl font-bold underline'>Playground</h1>
    </div>
  )
}
