import React from 'react'
import { useNavigate } from 'react-router'

import { Button } from '@mui/material'

interface ErrorPageProps {
  title: string
  description?: string
  buttonText?: string
  onButtonClick?: () => void
}

/**
 * An error page
 * @prop title - the title of the error
 * @prop description - the description of the error
 * @prop buttonText - the text of the button. By default it is the return home button
 * @prop onButtonClick - the function to call when the button is clicked
 */
export const ErrorPage: React.FC<ErrorPageProps> = props => {
  const navigate = props.buttonText !== '' ? useNavigate() : (_str: string) => {}

  const buttonText = props.buttonText ?? 'Return Home'
  const onButtonClick = props.onButtonClick ?? ((): void => {
    navigate('/')
    window.location.reload()
  })

  return (
    <div className='w-full flex flex-col justify-items-center items-center'>
      <img className='w-4/5 max-w-2xl' src='/error.png' alt='computer error screen' />
      <div className='flex flex-col justify-items-center items-center'>
        <div className='text-2xl font-bold'>{props.title}</div>
        {props.description && <div>{props.description}</div>}
      </div>
      {buttonText && <Button onClick={onButtonClick}>{buttonText}</Button>}
    </div>
  )
}
