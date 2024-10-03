import React from 'react'

/**
 * Displays the time in Xh Ym Zs format
 * @prop time - the time in seconds
 */
export const Timer: React.FC<{ time: number }> = props => {
  const hours = Math.floor(props.time / 3600)
  const minutes = Math.floor((props.time % 3600) / 60)
  const seconds = Math.floor(props.time % 60)
  const secondTenths = Math.floor((props.time * 10) % 10)

  let formattedTime = `${seconds}.${secondTenths}s`
  if (minutes > 0) {
    formattedTime = `${minutes}m ` + formattedTime
  }
  if (hours > 0) {
    formattedTime = `${hours}h ` + formattedTime
  }

  return <div>{formattedTime}</div>
}
