import React, { useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'

import { ErrorBoundary } from './components/Error/ErrorBoundary'
import { ErrorPage } from './components/Error/ErrorPage'
import { Page404 } from './components/Error/Page404'
import { Playground } from './components/Home/Playground'
import { WebsocketProvider } from './context/WSContext'

/**
 * The main component that wraps the entire application
 */
const App: React.FC = () => {
  return (
    <WebsocketProvider>
        <BrowserRouter>
        <Routes>
          <Route path='/' element={<Safe><Playground /></Safe>} />
          <Route path='*' element={<Safe><Page404 /></Safe>} />
        </Routes>
      </BrowserRouter>
    </WebsocketProvider>
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
