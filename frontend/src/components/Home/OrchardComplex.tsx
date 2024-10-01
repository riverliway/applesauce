import React from 'react'
import { OrchardComplex2D } from '../../context/apiTypes'

interface OrchardComplexProps {
  data: OrchardComplex2D
  scale: number
}

export const OrchardComplex: React.FC<OrchardComplexProps> = props => {
  return (
    <div
      className='relative bg-orange-100'
      style={{
        width: props.data.width / props.scale,
        height: props.data.height / props.scale
      }}
    >
      {props.data.trees.map((tree, i) => (
        <div
          key={i}
          className='absolute bg-green-300 rounded-full'
          style={{
            width: tree.diameter / props.scale,
            height: tree.diameter / props.scale,
            top: tree.y / props.scale,
            left: tree.x / props.scale
          }}
        />
      ))}
      {props.data.bots.map((bot, i) => (
        <div
          key={i}
          className='absolute bg-blue-300 rounded-full flex justify-center items-start'
          style={{
            width: bot.diameter / props.scale,
            height: bot.diameter / props.scale,
            top: bot.y / props.scale,
            left: bot.x / props.scale,
            transform: `rotate(${bot.orientation}rad)`,
          }}
        >
          <div className='bg-black' style={{ width: 2 / props.scale, height: bot.diameter / props.scale / 2 }} />
        </div>
      ))}
      {props.data.apples.map((apple, i) => (
        <div
          key={i}
          className='absolute bg-red-300 rounded-full'
          style={{
            width: apple.diameter / props.scale,
            height: apple.diameter / props.scale,
            top: apple.y / props.scale,
            left: apple.x / props.scale
          }}
        />
      ))}
    </div>
  )
}
