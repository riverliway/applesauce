import React from 'react'
import { OrchardSimulation2D } from '../../context/apiTypes'
import { indexArray } from '../../utils/indexArray'

interface OrchardProps {
  data: OrchardSimulation2D
}

export const Orchard: React.FC<OrchardProps> = props => {
  return (
    <div className='flex flex-col w-full h-full'>
      {indexArray(props.data.height).map(y => (
        <div key={y} className='flex flex-row w-full h-full border-b border-black first:border-t'>
          {indexArray(props.data.width).map(x => {
            const cell = props.data.trees.some(tree => tree[0] === x && tree[1] === y)
              ? '🌳'
              : props.data.apples.some(apple => apple[0] === x && apple[1] === y)
              ? '🍎'
              : props.data.bot_locations.some(bot => bot[0] === x && bot[1] === y)
              ? '🤖'
              : ' '

            const backgroundColor = {'🌳': 'bg-green-300', '🍎': 'bg-red-300', '🤖': 'bg-blue-300', ' ': 'bg-orange-100'}[cell]

            return (
              <div key={x} className={`text-3xl w-full h-full border-r border-black first:border-l flex flex-col justify-center items-center ${backgroundColor}`}>
                {cell}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}
