import React from 'react'
import { OrchardComplex2D } from '../../context/apiTypes'

interface OrchardComplexProps {
  data: OrchardComplex2D
}

export const OrchardComplex: React.FC<OrchardComplexProps> = props => {
  return (
    <div className='bg-orange-100' style={{ width: props.data.width, height: props.data.height }}>
      {props.data.trees.map((tree, i) => (
        <div
          key={i}
          className='absolute bg-green-300 rounded-full'
          style={{ width: tree.diameter, height: tree.diameter, top: tree.y, left: tree.x }}
        />
      ))}
      {props.data.bots.map((bot, i) => (
        <div
          key={i}
          className='absolute bg-blue-300 rounded-full'
          style={{ width: bot.diameter, height: bot.diameter, top: bot.y, left: bot.x }}
        />
      ))}
      {props.data.apples.map((apple, i) => (
        <div
          key={i}
          className='absolute bg-red-300 rounded-full'
          style={{ width: apple.diameter, height: apple.diameter, top: apple.y, left: apple.x }}
        />
      ))}
    </div>
  )
}
