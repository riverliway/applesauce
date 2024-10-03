import React, { useEffect, useRef, useState } from 'react'
import { useWebsocketContext } from '../../context/WSContext'
import { Coord, OrchardComplex2D, WsPathUpdate, WsSimulationUpdate } from '../../context/apiTypes'
import { CircularProgress } from '@mui/material'
import { OrchardComplex } from './OrchardComplex'
import { Timer } from '../Common/Timer'

const VIEW_WIDTH = 1000
const VIEW_HEIGHT = 800
const SCALE = 2
const SIM_DISPLAY_SPEED = 0.05

export const Playground: React.FC = () => {
  const api = useWebsocketContext()
  const [tickSpeed, setTickSpeed] = useState<number | undefined>(undefined)
  const [width, _setWidth] = useState(VIEW_WIDTH * SCALE)
  const [height, _setHeight] = useState(VIEW_HEIGHT * SCALE)
  const [numBots, _setNumBots] = useState(4)
  const [numBaskets, _setNumBaskets] = useState(4)
  const [orchard, setOrchard] = useState<OrchardComplex2D | undefined>()
  const [pathDots, setPathDots] = useState<{ point: Coord[], time: number }[]>([])
  const stateUpdateRef = useRef<OrchardComplex2D[]>([])
  const [startTime, _setStartTime] = useState(Date.now())

  useEffect(() => {
    const callback = (m: WsSimulationUpdate) => {
      stateUpdateRef.current = [...stateUpdateRef.current, m.simulation]
      if (tickSpeed === undefined) {
        setTickSpeed(m.simulation.TICK_SPEED)
      }
    }

    const pathsCallback = (m: WsPathUpdate) => {
      setPathDots(pd => [...pd, { point: m.paths, time: m.time }])
    }

    api.register('simulation', callback)
    api.register('path', pathsCallback)
    api.startSimulation('complex', width, height, numBots, numBaskets, 13334)

    return () => {
      api.unregister('simulation', callback)
      api.unregister('path', pathsCallback)
    }
  }, [])

  const updateState = (): void => {
    if (stateUpdateRef.current.length > 0) {
      setOrchard(stateUpdateRef.current[0])
      stateUpdateRef.current = stateUpdateRef.current.slice(1)
    }
  }

  useEffect(() => {
    if (tickSpeed === undefined) {
      return
    }

    const clear = setInterval(updateState, 1000 / tickSpeed * SIM_DISPLAY_SPEED)

    return () => clearInterval(clear)
  }, [tickSpeed])

  if (orchard === undefined) {
    return <CircularProgress />
  }

  const timeSeconds = orchard.time / orchard.TICK_SPEED

  return (
    <div className='w-full h-full flex flex-col justify-center items-center'>
      <div className='flex flex-col justify-center items-center'>
        <div className='flex flex-row justify-between items-center' style={{ width: 1000 }}>
          <div>
            <div className='flex flex-row gap-2'>Real Time: <Timer time={(Date.now() - startTime) / 1000} /></div>
            <div className='flex flex-row gap-2'>Simulated Time: <Timer time={timeSeconds} /></div>
          </div>
          <div>Apples: {orchard.apples.filter(a => a.collected).length}/{orchard.apples.length}</div>
        </div>
        <OrchardComplex data={orchard} scale={SCALE} pathDots={pathDots.find(pd => pd.time === orchard.time)?.point} />
      </div>
    </div>
  )
}
