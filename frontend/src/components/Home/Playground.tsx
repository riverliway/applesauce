import React, { useEffect, useRef, useState } from 'react'
import { useWebsocketContext } from '../../context/WSContext'
import { OrchardComplex2D, WsSimulationUpdate } from '../../context/apiTypes'
import { CircularProgress } from '@mui/material'
import { OrchardComplex } from './OrchardComplex'

const VIEW_WIDTH = 1000
const VIEW_HEIGHT = 800
const SCALE = 2

export const Playground: React.FC = () => {
  const api = useWebsocketContext()
  const [width, _setWidth] = useState(VIEW_WIDTH * SCALE)
  const [height, _setHeight] = useState(VIEW_HEIGHT * SCALE)
  const [numBots, _setNumBots] = useState(1)
  const [orchard, setOrchard] = useState<OrchardComplex2D | undefined>()
  const stateUpdateRef = useRef<OrchardComplex2D[]>([])

  useEffect(() => {
    const callback = (m: WsSimulationUpdate) => stateUpdateRef.current = [...stateUpdateRef.current, m.simulation]

    api.register('simulation', callback)
    api.startSimulation('complex', width, height, numBots)

    return () => api.unregister('simulation', callback)
  }, [])

  const updateState = (): void => {
    if (stateUpdateRef.current.length > 0) {
      setOrchard(stateUpdateRef.current[0])
      stateUpdateRef.current = stateUpdateRef.current.slice(1)
    }
  }

  useEffect(() => {
    const clear = setInterval(updateState, 10)

    return () => clearInterval(clear)
  }, [])

  if (orchard === undefined) {
    return <CircularProgress />
  }

  const timeSeconds = orchard.time / orchard.TICK_SPEED

  return (
    <div className='w-full h-full flex flex-col justify-center items-center'>
      <div className='flex flex-col justify-center items-center w-5/6 h-5/6'>
        <div className='flex flex-row w-full justify-between items-center'>
          <div>Time: {Math.floor(timeSeconds)}.{(timeSeconds * 10) % 10}s</div>
          <div>Apples: {orchard.apples.filter(a => a.collected).length}/{orchard.apples.length}</div>
        </div>
        <OrchardComplex data={orchard} scale={SCALE} />
      </div>
    </div>
  )
}
