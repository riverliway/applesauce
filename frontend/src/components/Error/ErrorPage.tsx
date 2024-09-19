import React from 'react'

import './ErrorPage.css'
import { Button } from 'antd'
import { useNavigate } from 'react-router'

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
  const navigate = useNavigate()

  const buttonText = props.buttonText ?? 'Return Home'
  const onButtonClick = props.onButtonClick ?? ((): void => {
    navigate('/')
    window.location.reload()
  })

  return (
    <div className='errorPageContain'>
      <img className='errorImg' src='/error.png' alt='computer error screen' />
      <div className='errorPageContent'>
        <div className='errorPageTitle'>{props.title}</div>
        {props.description && <div>{props.description}</div>}
      </div>
      {buttonText && <Button size='large' type='primary' onClick={onButtonClick}>{buttonText}</Button>}
    </div>
  )
}
