import React, { useEffect, useState } from 'react'
import { OrchardComplex2D } from '../../context/apiTypes'
import { TextField } from '@mui/material'
import { OrchardComplex } from './OrchardComplex'
import { Timer } from '../Common/Timer'
import { ArrowLeft, ArrowRight, FirstPage, LastPage, Pause, PlayArrow } from '@mui/icons-material'

const VIEW_WIDTH = 500
const VIEW_HEIGHT = 500
const SCALE = 1
const SIM_DISPLAY_SPEED = 0.05

export const Playback: React.FC = () => {
  const [history, setHistory] = useState<OrchardComplex2D[]>([])
  const [timestep, setTimestep] = useState(0)
  const [startTime, setStartTime] = useState(0)
  const [playing, setPlaying] = useState(true)

  const updateState = (): void => {
    setTimestep(t => t < history.length - 1 && playing ? t + 1 : t)
  }

  useEffect(() => {
    if (history.length === 0) {
      return
    }

    setStartTime(Date.now())
    const clear = setInterval(updateState, 1000 / history[0].TICK_SPEED * SIM_DISPLAY_SPEED)

    return () => clearInterval(clear)
  }, [history.length])

  if (history.length === 0) {
    return (
      <TextField
        label="Multiline"
        multiline
        maxRows={4}
        onChange={e => {
          let content = e.target.value.trim()
          content = content.endsWith(',') ? content.slice(0, -1) : content
          content = content.startsWith('[') ? content : `[${content}]`
          setHistory(JSON.parse(content))
        }}
      />
    )
  }

  const orchard = history[timestep]
  const timeSeconds = orchard.time / orchard.TICK_SPEED

  return (
    <div className='w-full h-full flex flex-col justify-center items-center'>
      <div className='flex flex-col justify-center items-center'>
        <div className='flex flex-row justify-between items-center' style={{ width: 1000 }}>
          <div>
            <div className='flex flex-row gap-2'>Real Time: <Timer time={(Date.now() - startTime) / 1000} /></div>
            <div className='flex flex-row gap-2'>Simulated Time: <Timer time={timeSeconds} /></div>
          </div>
          <div className='flex flex-row gap-2'>
            <FirstPage onClick={() => {
              setPlaying(false)
              setTimestep(t => t > 0 ? t - 1 : 0)
            }} />
            {playing ? (
              <Pause onClick={() => setPlaying(false)} />
            ) : (
              <PlayArrow onClick={() => setPlaying(true)} />
            )}
            <LastPage onClick={() => {
              setPlaying(false)
              setTimestep(t => t < history.length - 1 ? t + 1 : t)
            }} />
          </div>
          <div>Apples: {orchard.apples.filter(a => a.collected).length}/{orchard.apples.length}</div>
        </div>
        <OrchardComplex data={orchard} scale={SCALE} />
      </div>
    </div>
  )
}
