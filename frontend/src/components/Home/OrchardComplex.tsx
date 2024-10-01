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
            top: (tree.y - tree.diameter / 2) / props.scale,
            left: (tree.x - tree.diameter / 2) / props.scale
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
            top: (bot.y - bot.diameter / 2) / props.scale,
            left: (bot.x - bot.diameter / 2) / props.scale,
            transform: `rotate(${bot.orientation + Math.PI / 2}rad)`,
          }}
        >
          <div className='bg-black' style={{ width: 2 / props.scale, height: bot.diameter / props.scale / 2 }} />
        </div>
      ))}
      {props.data.bots.filter(b => b.holding !== undefined && b.holding !== null).map((bot, i) => {
        const apple = props.data.apples[bot.holding as number]
        const noseX = bot.x + Math.cos(bot.orientation) * bot.diameter / 2
        const noseY = bot.y + Math.sin(bot.orientation) * bot.diameter / 2

        return (
          <div
            key={i}
            className='absolute bg-red-300 rounded-full'
            style={{
              width: apple.diameter / props.scale,
              height: apple.diameter / props.scale,
              top: (noseY - apple.diameter / 2) / props.scale,
              left: (noseX - apple.diameter / 2) / props.scale
            }}
          />
        )
      })}
      {props.data.baskets.map((basket, i) => (
        <div
          key={i}
          className='absolute bg-slate-400 rounded-full flex justify-center items-start'
          style={{
            width: basket.diameter / props.scale,
            height: basket.diameter / props.scale,
            top: (basket.y - basket.diameter / 2) / props.scale,
            left: (basket.x - basket.diameter / 2) / props.scale
          }}
        />
      ))}
      {props.data.apples.filter(a => !a.held).map((apple, i) => (
        <div
          key={i}
          className='absolute bg-red-300 rounded-full'
          style={{
            width: apple.diameter / props.scale,
            height: apple.diameter / props.scale,
            top: (apple.y - apple.diameter / 2) / props.scale,
            left: (apple.x - apple.diameter / 2) / props.scale
          }}
        />
      ))}
    </div>
  )
}
