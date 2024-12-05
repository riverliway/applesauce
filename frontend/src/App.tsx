import React, { useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { ErrorBoundary } from './components/Error/ErrorBoundary'
import { ErrorPage } from './components/Error/ErrorPage'
import { Page404 } from './components/Error/Page404'
import { Playground } from './components/Home/Playground'
import { WebsocketProvider } from './context/WSContext'
import { Playback } from './components/Home/Playback'

/**
 * The main component that wraps the entire application
 */
const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/playback' element={<Safe><Playback /></Safe>} />
        <Route path='/' element={<Safe><WebsocketProvider><Playground /></WebsocketProvider></Safe>} />
        <Route path='*' element={<Safe><Page404 /></Safe>} />
      </Routes>
    </BrowserRouter>
  )
}

interface SafeProps {
  children: React.ReactNode
}

/**
 * A route that catches any errors that occur in the element
 */
const Safe: React.FC<SafeProps> = ({ children }) => {
  const [errorMessage, setErrorMessage] = useState<string | undefined>()

    return (
      <ErrorBoundary
        catch={(error, errorInfo) => {
          setErrorMessage(error.message)
          console.error(error)
          console.error(errorInfo)
        }}
        alternate={<ErrorPage title='An Error Occured' description={errorMessage} />}
      >
        {children}
      </ErrorBoundary>
    )
}

export default App
