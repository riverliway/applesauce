import React from 'react'

import { ErrorPage } from './ErrorPage'

/**
 * An error page for 404 errors
 */
export const Page404: React.FC = () => {
  return (
    <ErrorPage
      title='404 - Unknown Page'
      description="We don't recognize that URL."
    />
  )
}
